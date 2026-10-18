"""GPU side of the dashboard. Runs the shipped ai_router code unchanged and records what each step did.

- run_batch(): M29Pipeline.run with its own tracer (spans, metrics, per-image records), plus a hook on
  route() that keeps the object vectors and scene vectors the pipeline would otherwise drop, so the
  batch page can draw the latent space, class centroids and the K-Means clusters.
- explain(): one image, every step B0..B7 with the matrices behind it (same code path as
  one_image_walkthrough/walkthrough.py). Every number we recompute by hand is compared with the repo's
  own result and the comparison is returned in "checks".
"""
from __future__ import annotations

import json
import os
import math
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from settings import AI_ROUTER, SPLITS

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
sys.path.insert(0, str(AI_ROUTER))
import torch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

from src.config import calibrated_dir, load_config  # noqa: E402
from src.inference.hungarian_matcher import active_mask, build_groups, select_anchor_objects  # noqa: E402
from src.pipeline import M29Pipeline  # noqa: E402
from src.routing.diversity_sampler import diversity_select  # noqa: E402
from src.utils.boxes import cxcywh_to_xyxy, giou_matrix_xyxy, iou_matrix_cxcywh, to_pixel_xyxy  # noqa: E402
from src.utils.eval_utils import match_to_gt  # noqa: E402
from src.utils.io_manager import _labels_of, budget_size, load_bdd_gt, make_loader  # noqa: E402
from src.monitoring.tracer import routing_rule  # noqa: E402


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    z = np.exp(x - x.max())
    return z / z.sum()


def R(x, n=4):
    """Round numbers / arrays for JSON; NaN -> None."""
    if x is None:
        return None
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 0:
        v = float(a)
        return None if not math.isfinite(v) else round(v, n)
    a = np.round(a, n)
    return [R(v, n) for v in a] if a.ndim > 1 else [None if not math.isfinite(v) else float(v) for v in a]


def find_labels(stem: str) -> Optional[Path]:
    for d in SPLITS.values():
        p = d / f"{stem}.json"
        if p.is_file():
            return p
    return None


class Engine:
    def __init__(self):
        self.cfg = load_config()
        self.lock = threading.Lock()                     # one GPU, one job at a time
        self.pipe = M29Pipeline(self.cfg)
        self.eng, self.router = self.pipe.engine, self.pipe.router
        self.gate = self.router.gate
        self.names = self.eng.class_names
        self.mask = active_mask(self.names, self.cfg["classes"].get("active"))
        self.det = self.cfg["detection"]
        self.th = self.router.t
        self.device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------ system info
    def system(self) -> Dict:
        members = []
        for m in self.cfg["ensemble"]["members"]:
            ck = Path(m.get("checkpoint") or "")
            members.append({"name": m["name"], "checkpoint": ck.name, "mb": round(ck.stat().st_size / 2**20, 1) if ck.is_file() else None,
                            "arch": self.cfg["ensemble"].get("hf_config")})
        members[0]["role"] = "Anchor"
        div = self.cfg["diversity"]
        return {"device": self.device, "members": members, "classes": self.names,
                "input_size": self.cfg["inference"]["input_size"], "detection": self.det,
                "thresholds": vars(self.th), "policy": self.cfg["policy"]["routing"],
                "diversity": {"budget_ratio": div["budget_ratio"], "budget_abs": div.get("budget_abs"),
                              "n_init": div.get("kmeans_n_init", 10), "seed": div.get("seed", 0)},
                "fcd": {"dim": int(self.gate.comps.shape[0]), "classes": [self.names[c] for c in self.gate.slot]},
                "ai_router": str(AI_ROUTER)}

    def centroid(self, cls: str) -> Dict:
        c = self.names.index(cls)
        s = self.gate.slot.get(c)
        if s is None:
            return {"class": cls, "missing": True}
        inv = self.gate.inv_cov[s].astype(np.float64)
        return {"class": cls, "mu": R(self.gate.mu[s]), "inv_cov": R(inv, 3),
                "cov_diag": R(np.diag(np.linalg.inv(inv)))}

    # ------------------------------------------------------------------ batch
    def run_batch(self, paths: Sequence[Path], job_dir: Path, progress: Callable[[int, int], None]) -> Dict:
        job_dir = Path(job_dir)
        cap: Dict[str, Dict] = {}
        pipe = self.pipe
        orig_route = pipe.route

        def hooked(results):
            results = list(results)
            for r in results:
                cap[r.filename] = {
                    "scene": np.asarray(r.image_embedding, np.float64).copy(),
                    "cls": [int(o.predicted_class) for o in r.object_scores],
                    "emb": np.stack([o.anchor_embedding for o in r.object_scores]).astype(np.float64) if r.object_scores else np.zeros((0, 256)),
                    "unc": [float(o.unc_total) for o in r.object_scores],
                    "conf": [float(o.mean_confidence) for o in r.object_scores],
                    "size": list(r.image_size)}
            out = orig_route(results)
            for r in out:
                cap[r.filename].update(zone_b5=r.zone, priority_b5=r.priority, reasons_b5=list(r.flag_reasons),
                                       img_unc=float(r.max_uncertainty), img_conf=float(r.min_confidence), n=int(r.n_objects),
                                       rule=routing_rule(r.flag_reasons, r.n_objects, r.zone))
            progress(len(cap), len(paths))
            return out

        with self.lock:
            pipe.route = hooked
            try:
                rep = pipe.run(list(paths), out_dir=job_dir / "queue", num_workers=0, trace_dir=job_dir / "trace")
            finally:
                del pipe.route                       # back to the class method
        return self._summarise(rep, cap, job_dir)

    def _summarise(self, rep: Dict, cap: Dict[str, Dict], job_dir: Path) -> Dict:
        names, gate, th = self.names, self.gate, self.th
        results = rep["results"]
        images = []
        for r in results:
            c = cap[r.filename]
            images.append({"name": r.filename, "zone": r.zone, "priority": r.priority, "reasons": list(r.flag_reasons),
                           "zone_b5": c["zone_b5"], "rule": c["rule"], "img_unc": R(c["img_unc"]), "img_conf": R(c["img_conf"]),
                           "n": c["n"], "size": c["size"], "dedup": "DIVERSITY_DEDUP" in r.flag_reasons})
        # ---- objects: FCD distance for every object (the pipeline skips GREY images; we compute it for the plots)
        obj_img, obj_cls, Z, D, unc, conf = [], [], [], [], [], []
        for i, r in enumerate(results):
            c = cap[r.filename]
            if not len(c["cls"]):
                continue
            z = gate.project(c["emb"])
            d = gate.distances(c["emb"], np.array(c["cls"]))
            Z.append(z); D.extend(d.tolist()); obj_cls.extend(c["cls"]); obj_img.extend([i] * len(c["cls"]))
            unc.extend(c["unc"]); conf.extend(c["conf"])
        Z = np.concatenate(Z) if Z else np.zeros((0, gate.comps.shape[0]))
        D = np.asarray(D, np.float64); obj_cls = np.asarray(obj_cls, int)
        for i, im in enumerate(images):
            m = np.asarray(obj_img) == i
            fin = D[m][np.isfinite(D[m])]
            im["fcd_max_all"] = R(fin.max()) if len(fin) else None
        per_class = []
        for k, nm in enumerate(names):
            m = obj_cls == k
            dk = D[m][np.isfinite(D[m])]
            per_class.append({"name": nm, "n": int(m.sum()),
                              "unc_mean": R(np.mean(np.asarray(unc)[m])) if m.any() else None,
                              "unc_var": R(np.var(np.asarray(unc)[m])) if m.any() else None,
                              "conf_mean": R(np.mean(np.asarray(conf)[m])) if m.any() else None,
                              "conf_var": R(np.var(np.asarray(conf)[m])) if m.any() else None,
                              "fcd_mean": R(dk.mean()) if len(dk) else None, "fcd_var": R(dk.var()) if len(dk) else None,
                              "fcd_over_tau": int((dk > th.tau_fcd).sum())})
        # ---- latent space of objects: 2-D PCA of the 64-D FCD space, class centroids projected the same way
        latent = None
        if len(Z) >= 3:
            mean = Z.mean(0)
            _, _, vt = np.linalg.svd(Z - mean, full_matrices=False)
            P = vt[:2]
            xy = (Z - mean) @ P.T
            rng = np.random.default_rng(0)
            keep = np.sort(rng.choice(len(Z), min(len(Z), 4000), replace=False))
            cent = [{"name": names[c], "xy": R((gate.mu[s] - mean) @ P.T, 3)} for c, s in gate.slot.items()]
            var = np.var((Z - mean) @ vt.T, axis=0)
            latent = {"x": R(xy[keep, 0], 3), "y": R(xy[keep, 1], 3), "cls": obj_cls[keep].tolist(),
                      "img": np.asarray(obj_img)[keep].tolist(), "d": R(D[keep], 3), "centroids": cent,
                      "explained": R(var[:2] / var.sum(), 4), "n_total": int(len(Z))}
        # ---- K-Means on the RED pool, replicated with the pipeline's own function and seed
        div = self.cfg["diversity"]
        budget, seed, n_init = rep["budget"], div.get("seed", 0), div.get("kmeans_n_init", 10)
        red_idx = [i for i, im in enumerate(images) if im["zone_b5"] == "RED"]
        kmeans = {"budget": budget, "red_before": len(red_idx), "ran": len(red_idx) > budget}
        if kmeans["ran"]:
            x = np.stack([cap[images[i]["name"]]["scene"] for i in red_idx])
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
            km = KMeans(n_clusters=budget, n_init=n_init, random_state=seed).fit(x)
            chosen = diversity_select(np.stack([cap[images[i]["name"]]["scene"] for i in red_idx]), budget,
                                      [cap[images[i]["name"]]["priority_b5"] for i in red_idx],
                                      seed, n_init)
            chosen_names = {images[red_idx[j]]["name"] for j in chosen}
            final_red = {im["name"] for im in images if im["zone"] == "RED"}
            mean = x.mean(0)
            _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
            xy = (x - mean) @ vt[:2].T
            cxy = (km.cluster_centers_ - mean) @ vt[:2].T
            dist = np.linalg.norm(x - km.cluster_centers_[km.labels_], axis=1)
            sizes = np.bincount(km.labels_, minlength=budget)
            for j, i in enumerate(red_idx):
                images[i]["cluster"] = int(km.labels_[j]); images[i]["dist_to_centroid"] = R(dist[j])
                images[i]["kept_by_kmeans"] = images[i]["name"] in chosen_names
            kmeans.update({"x": R(xy[:, 0], 4), "y": R(xy[:, 1], 4), "label": km.labels_.tolist(),
                           "img": red_idx, "cx": R(cxy[:, 0], 4), "cy": R(cxy[:, 1], 4), "sizes": sizes.tolist(),
                           "inertia": R(km.inertia_), "kept": sorted(red_idx[j] for j in chosen),
                           "replicates_pipeline": chosen_names == final_red,
                           "cluster_size_hist": np.bincount(sizes).tolist()})
        zones_b5 = {z: sum(im["zone_b5"] == z for im in images) for z in ("RED", "GREEN", "GREY")}
        summary = {"images": images, "n_images": len(images), "zones": rep["zones"], "zones_b5": zones_b5,
                   "budget": budget, "thresholds": vars(th), "per_class": per_class, "latent": latent, "kmeans": kmeans,
                   "objects": {"unc": R(unc), "conf": R(conf), "cls": obj_cls.tolist(), "d": R(D, 3), "img": list(obj_img)},
                   "elapsed_s": R(rep["elapsed_s"], 2), "images_per_s": R(rep["images_per_s"], 2),
                   "stage_s": {k: R(v, 3) for k, v in rep["stage_s"].items()}, "peak_vram_mb": R(rep["peak_vram_mb"], 1)}
        (job_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    # ------------------------------------------------------------------ one image, every step
    def explain(self, img_path: Path, batch_ctx: Optional[Dict] = None) -> Dict:
        with self.lock:
            return self._explain(Path(img_path), batch_ctx)

    def _explain(self, img_path: Path, batch_ctx: Optional[Dict]) -> Dict:
        from PIL import Image
        cfg, det, names, mask, gate, th, eng, pipe = self.cfg, self.det, self.names, self.mask, self.gate, self.th, self.eng, self.pipe
        checks = []

        def check(name, ours, repo, tol=1e-6):
            ok = bool(np.allclose(np.asarray(ours, float), np.asarray(repo, float), atol=tol, equal_nan=True))
            checks.append({"name": name, "ok": ok})

        out: Dict = {"image": img_path.name, "names": names, "members": eng.member_names, "thresholds": vars(th),
                     "detection": det, "policy": cfg["policy"]["routing"]}
        timings = {}
        t0 = time.perf_counter()
        # B0 -------------------------------------------------------------
        orig = Image.open(img_path).convert("RGB")
        W, H = orig.size
        inp = cfg["inference"]["input_size"]
        batch = next(iter(make_loader([img_path], 1, 0, inp)))
        px = batch["pixels"][0]
        manual = np.asarray(orig.resize((inp[1], inp[0]), resample=Image.BILINEAR), dtype=np.uint8).transpose(2, 0, 1)
        check("B0 resize == DataLoader", manual, px.numpy(), tol=0)
        pxf = px.float() / 255
        out["B0"] = {"W": W, "H": H, "in": inp, "sx": W / inp[1], "sy": H / inp[0], "bytes": img_path.stat().st_size,
                     "mean_rgb": [R(pxf[c].mean()) for c in range(3)], "std_rgb": [R(pxf[c].std()) for c in range(3)],
                     "pixel_sample": px[:, 320, 316:324].numpy().T.tolist()}
        timings["B0"] = time.perf_counter() - t0
        # B1 -------------------------------------------------------------
        t = time.perf_counter()
        eb = eng.forward_batch(batch["pixels"])
        timings["B1"] = time.perf_counter() - t
        M, Q = eb.boxes.shape[0], eb.boxes.shape[2]
        b1 = []
        for m in range(M):
            sc = sig(np.where(mask[None], eb.logits[m, 0], -np.inf))
            best, cls = sc.max(1), sc.argmax(1)
            keep = np.flatnonzero(best >= det["score_threshold"])
            keep = keep[np.argsort(-best[keep])]
            b1.append({"name": eng.member_names[m], "queries": int(Q), "kept": int(len(keep)),
                       "hist": np.histogram(best, bins=[0, .05, .1, .2, .3, .5, .7, .9, 1.0001])[0].tolist(),
                       "dets": [{"q": int(q), "cls": names[cls[q]], "score": R(best[q], 3),
                                 "box": R(to_pixel_xyxy(eb.boxes[m, 0, q], W, H), 1)} for q in keep]})
        out["B1"] = {"models": b1, "hidden_shape": list(eb.anchor_hidden.shape), "scene_dim": int(eb.scene.shape[1]),
                     "scene_norm": R(np.linalg.norm(eb.scene[0]), 3), "logit_shape": list(eb.logits.shape)}
        # B2 -------------------------------------------------------------
        t = time.perf_counter()
        boxes, logits, hidden = eb.boxes[:, 0].astype(np.float64), eb.logits[:, 0].astype(np.float64), eb.anchor_hidden[0]
        q_idx, a_cls, a_sc = select_anchor_objects(logits[0], mask, det["score_threshold"], det["max_objects"])
        groups = build_groups(boxes, logits, hidden, mask, det)
        K = len(q_idx)
        w = det["match_cost"]
        hung = []
        for m in range(1, M):
            a_xy, m_xy = cxcywh_to_xyxy(boxes[0, q_idx]), cxcywh_to_xyxy(boxes[m])
            l1 = np.abs(boxes[0, q_idx][:, None] - boxes[m][None]).sum(-1)
            gi = giou_matrix_xyxy(a_xy, m_xy)
            pc = sig(logits[m][:, a_cls]).T
            cost = w["l1"] * l1 - w["giou"] * gi - w["cls"] * pc
            chosen = np.zeros(K, int)
            if K:
                rows, cols = linear_sum_assignment(cost)
                chosen[rows] = cols
            for k in range(K):
                check(f"B2 {eng.member_names[m]} khớp object {k}", chosen[k], groups[k].member_preds[m - 1].query_index, tol=0)
            # columns shown: every chosen query plus the 3 cheapest alternatives of each row (at most 40)
            cols = list(dict.fromkeys(chosen.tolist()))
            for k in range(K):
                for c in np.argsort(cost[k])[:3]:
                    if len(cols) >= 40:
                        break
                    if int(c) not in cols:
                        cols.append(int(c))
            ious = iou_matrix_cxcywh(boxes[0, q_idx], boxes[m][chosen]).diagonal() if K else np.zeros(0)
            hung.append({"member": eng.member_names[m], "cols": cols, "l1": R(l1[:, cols]), "giou": R(gi[:, cols]),
                         "pc": R(pc[:, cols]), "cost": R(cost[:, cols]), "chosen": chosen.tolist(), "iou": R(ious),
                         "member_cls": [names[int(np.argmax(np.where(mask, logits[m][c], -np.inf)))] for c in cols],
                         "member_box": [R(to_pixel_xyxy(boxes[m][c], W, H), 1) for c in chosen]})
        out["B2"] = {"K": K, "q_idx": q_idx.tolist(), "cls": [names[c] for c in a_cls], "score": R(a_sc),
                     "weights": w, "match_iou": det["match_iou"], "hungarian": hung,
                     "anchor_box": [R(to_pixel_xyxy(boxes[0, q], W, H), 1) for q in q_idx]}
        timings["B2"] = time.perf_counter() - t
        # B3 -------------------------------------------------------------
        t = time.perf_counter()
        result = pipe.results_from_ensemble(eb, batch)[0]
        scores = result.object_scores
        objs = []
        for k, (g, s) in enumerate(zip(groups, scores)):
            preds = [g.anchor_pred, *g.member_preds]
            L = np.stack([p.logits for p in preds]).astype(np.float64); B = np.stack([p.box for p in preds]).astype(np.float64)
            P = np.stack([softmax(l) for l in L]); pbar = P.mean(0)
            c = int(np.argmax(np.where(mask, pbar, -np.inf)))
            terms = -pbar * np.log(pbar + 1e-9)
            unc_cls = float(terms.sum())
            var = B.var(axis=0); unc_loc = float(var.mean())
            unc_total = unc_cls + det["unc_loc_weight"] * unc_loc
            confs = sig(L[:, c]); mean_conf = float(confs.mean())
            in_cl = np.array([True, *[i >= det["wbf_iou"] for i in g.member_ious]])
            bxy = cxcywh_to_xyxy(B[in_cl]); wts = confs[in_cl]
            fused = (bxy * wts[:, None]).sum(0) / wts.sum()
            T = int(in_cl.sum()); wbf_conf = float(wts.mean() * min(T, M) / M)
            for nm, ours, repo in [("unc_cls", unc_cls, s.unc_cls), ("unc_loc", unc_loc, s.unc_loc), ("unc_total", unc_total, s.unc_total),
                                   ("mean_conf", mean_conf, s.mean_confidence), ("class", c, s.predicted_class),
                                   ("wbf_box", fused, cxcywh_to_xyxy(s.box_wbf)), ("wbf_conf", wbf_conf, s.wbf_confidence)]:
                check(f"B3 object {k} {nm}", ours, repo)
            objs.append({"k": k, "q": int(q_idx[k]), "L": R(L, 3), "P": R(P), "pbar": R(pbar), "terms": R(terms), "c": c, "cls": names[c],
                         "B": R(B, 5), "B_px": [R(to_pixel_xyxy(b, W, H), 1) for b in B], "var": R(var, 7),
                         "unc_cls": R(unc_cls), "unc_loc": R(unc_loc, 7), "unc_total": R(unc_total), "confs": R(confs),
                         "mean_conf": R(mean_conf), "in_cluster": in_cl.tolist(), "wts": R(wts), "fused": R(fused, 5),
                         "T": T, "wbf_conf": R(wbf_conf), "model_cls": [names[p.predicted_class] for p in preds],
                         "member_q": [int(p.query_index) for p in preds],
                         "ious": R(g.member_ious), "unmatched": bool(g.is_unmatched), "box": R(to_pixel_xyxy(s.box_wbf, W, H), 1),
                         "_emb": np.asarray(g.anchor_pred.embedding, np.float64)})
        timings["B3"] = time.perf_counter() - t
        # B4 -------------------------------------------------------------
        img_unc = max((o["unc_total"] for o in objs), default=0.0)
        img_conf = min((o["mean_conf"] for o in objs), default=0.0)
        out["B4"] = {"img_unc": R(img_unc), "img_conf": R(img_conf),
                     "k_unc": int(np.argmax([o["unc_total"] for o in objs])) if objs else None,
                     "k_conf": int(np.argmin([o["mean_conf"] for o in objs])) if objs else None,
                     "repo_img_unc": R(result.max_uncertainty), "repo_img_conf": R(result.min_confidence)}
        if objs:
            check("B4 img_unc", img_unc, result.max_uncertainty, 1e-4); check("B4 img_conf", img_conf, result.min_confidence, 1e-4)
        # B5 -------------------------------------------------------------
        t = time.perf_counter()
        slots = sorted(gate.slot.items(), key=lambda kv: kv[0])
        for o in objs:
            emb = o.pop("_emb")
            z = gate.project(emb)
            dists = []
            for c, s in slots:
                d = z - gate.mu[s]
                dists.append({"cls": names[c], "D": R(math.sqrt(max(float(d @ gate.inv_cov[s] @ d), 0.0))),
                              "euclid": R(np.linalg.norm(d))})
            s = gate.slot.get(o["c"])
            o["emb"] = R(emb, 3); o["z"] = R(z)
            o["centroid_dist"] = dists
            if s is None:
                o["dm"] = None
            else:
                d = z - gate.mu[s]
                Sd = gate.inv_cov[s] @ d
                o.update(mu=R(gate.mu[s]), dz=R(d), inv_dz=R(Sd), d2=R(float(d @ Sd)), dm=R(math.sqrt(max(float(d @ Sd), 0))),
                         contrib=R(d * Sd))
            o["fcd_flag"] = bool(o["dm"] is None and gate.missing_is_anomaly or (o["dm"] is not None and o["dm"] > th.tau_fcd))
        if objs:
            check("B5 D_M", [np.nan if o["dm"] is None else o["dm"] for o in objs],
                  gate.distances(np.stack([np.asarray(o["emb"]) for o in objs]), np.array([o["c"] for o in objs])), 1e-2)
        self.router.route(result)
        rp = cfg["policy"]["routing"]
        unmatched = any(o["unmatched"] for o in objs)
        rule_steps = []
        if not objs:
            rule_steps.append({"test": "n_objects == 0", "value": 0, "result": True, "then": f"{rp['empty_image_zone']} (NO_DETECTIONS)"})
        else:
            dis = img_unc > th.tau_unc
            rule_steps.append({"test": "img_unc > tau_unc", "value": f"{img_unc:.4f} > {th.tau_unc:.4f}", "result": bool(dis), "then": "RED P2"})
            if not dis:
                gc = img_conf >= th.tau_conf
                rule_steps.append({"test": "img_conf >= tau_conf", "value": f"{img_conf:.4f} >= {th.tau_conf:.4f}", "result": bool(gc),
                                   "then": "ứng viên Xanh → cổng FCD", "else": "GREY (LOW_CONFIDENCE)"})
            fcd_any = any(o["fcd_flag"] for o in objs)
            if dis and rp.get("fcd_on_all_objects", True) or (not dis and img_conf >= th.tau_conf):
                rule_steps.append({"test": "∃ object: D_M > tau_fcd", "value": f"max D_M = {max((o['dm'] or 0) for o in objs):.3f}, tau_fcd = {th.tau_fcd:.3f}",
                                   "result": bool(fcd_any), "then": "P1 (FCD_ANOMALY)"})
        out["B5"] = {"rule": routing_rule(result.flag_reasons, result.n_objects, result.zone), "steps": rule_steps,
                     "zone": result.zone, "priority": result.priority, "reasons": list(result.flag_reasons),
                     "unmatched_any": unmatched, "fcd_max": R(result.fcd_max_distance),
                     "gate": {"dim": int(gate.comps.shape[0]), "emb_dim": int(gate.comps.shape[1]), "classes": [names[c] for c, _ in slots]}}
        timings["B5"] = time.perf_counter() - t
        # B6 -------------------------------------------------------------
        div = cfg["diversity"]
        if batch_ctx:
            out["B6"] = {"in_batch": True, **batch_ctx}
        else:
            out["B6"] = {"in_batch": False, "budget": budget_size(1, div["budget_ratio"], div.get("budget_abs")),
                         "note": "Một ảnh lẻ: ngân sách K = ceil(0.2 × 1) = 1 ≥ số ảnh Đỏ, nên K-Means không chạy."}
        # B7 -------------------------------------------------------------
        result.image_size = (W, H)
        out["B7"] = {"zone": result.zone, "file": {"RED": "review_queue.json", "GREEN": "green_labels.json", "GREY": "grey_pool.json"}[result.zone],
                     "labels": _labels_of(result, names, with_anomaly=True)}
        # ground truth (only to judge; never used by the flow) ------------
        lp = find_labels(img_path.stem)
        if lp is not None:
            gt = load_bdd_gt(lp, cfg["classes"]["gt_map"], {n: i for i, n in enumerate(names)}, W, H)
            gt_px = [R(to_pixel_xyxy(b, W, H), 1) for b in gt["boxes"]]
            res = {"n": int(len(gt["classes"])), "boxes": gt_px, "cls": [names[c] for c in gt["classes"]], "source": lp.parent.name}
            if objs:
                tp, fn = match_to_gt(np.stack([s.box_wbf for s in scores]), np.array([o["c"] for o in objs]),
                                     np.array([o["mean_conf"] for o in objs]), gt["boxes"], gt["classes"], cfg["ltt"]["gt_match_iou"])
                for o, t_ in zip(objs, tp):
                    o["tp"] = bool(t_)
                res.update(tp=int(tp.sum()), fp=int((~tp).sum()), fn=int(fn))
            out["gt"] = res
        out["objects"] = objs
        out["checks"] = checks
        out["timings_ms"] = {k: round(v * 1000, 1) for k, v in timings.items()}
        return out
