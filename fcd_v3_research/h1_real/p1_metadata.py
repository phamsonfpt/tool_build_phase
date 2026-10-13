"""P1 — metadata-only checks (no box labels are used except the class schema).

* ride proxy: BDD100K video name = <ride-hash>-<clip-hash>; we test whether the prefix behaves like a ride id
* split overlap at ride level (A2), MOT ⊂ 100K check, class schema (C_eval)
* Evaluation pool (Val) units and m_tilde; Meta-Train = random rides of the 100K *test* split
Outputs cache/p1_units.json
"""
import collections
import json
import os
import random

from h1common import CACHE, DATA, write_json

random.seed(2026)


def names(split):
    return sorted(f[:-5] for f in os.listdir(os.path.join(DATA, "100k", split)) if f.endswith(".json"))


def ride(n):
    return n.split("-")[0]


def m_tilde(sizes):
    return sum(s * s for s in sizes) / sum(sizes)


out = {}
split_names = {s: names(s) for s in ("train", "val", "test")}
rides = {s: collections.Counter(ride(n) for n in v) for s, v in split_names.items()}
total_videos = sum(len(v) for v in split_names.values())
total_rides = len(set().union(*[set(r) for r in rides.values()]))
out["ride_proxy"] = {
    "videos": total_videos,
    "prefixes": total_rides,
    "videos_per_prefix": total_videos / total_rides,
    "per_split": {s: {"videos": len(split_names[s]), "prefixes": len(r), "max_videos_per_prefix": max(r.values()),
                      "m_tilde": m_tilde(list(r.values())),
                      "singleton_share": sum(1 for v in r.values() if v == 1) / len(r)} for s, r in rides.items()},
    "prefix_overlap": {"train&val": sorted(set(rides["train"]) & set(rides["val"])),
                       "train&test": len(set(rides["train"]) & set(rides["test"])),
                       "val&test": len(set(rides["val"]) & set(rides["test"]))},
}
# counterfactual: if the split had been random by *video* (70/10/20), how many Val rides would share a ride with train?
all_sizes = collections.Counter()
for r in rides.values():
    all_sizes.update(r)
rng = random.Random(7)
shared = []
for _ in range(5):
    val_r, tr_r = set(), set()
    for p, m in all_sizes.items():
        for _v in range(m):
            u = rng.random()
            (tr_r if u < 0.7 else val_r if u < 0.8 else set()).add(p)
    shared.append(len(val_r & tr_r) / len(val_r))
out["ride_proxy"]["counterfactual_video_split_val_rides_shared_with_train"] = sum(shared) / len(shared)

# MOT frames shipped in ai_router/data/{train,val,test} (frame #100 of each MOT clip)
mot = {}
for s in ("train", "val", "test"):
    vids = sorted(f.replace("-0000100.jpg", "") for f in os.listdir(os.path.join(DATA, s)) if f.endswith(".jpg"))
    mot[s] = {"clips": len(vids), "in_100k_same_split": len(set(vids) & set(split_names[s])),
              "rides": len({ride(v) for v in vids})}
out["mot_frames"] = mot

# class schema (labels are read only for category names)
cats = collections.Counter()
for n in split_names["val"][:2000]:
    d = json.load(open(os.path.join(DATA, "100k", "val", n + ".json")))
    for o in d["frames"][0].get("objects", []):
        if "box2d" in o:
            cats[o["category"]] += 1
out["box_categories_val_sample"] = dict(cats)
out["C_eval"] = ["car"]  # the expert detectors are single-class (car) fine-tunes

# Evaluation pool = Val minus any ride also present in train (A2) and minus the detector-training rides
detector_train_rides = {ride(f.replace("-0000100.jpg", "")) for f in os.listdir(os.path.join(DATA, "train"))}
excluded = set(rides["train"]) | detector_train_rides
eval_rides = sorted(r for r in rides["val"] if r not in excluded)
by_ride = collections.defaultdict(list)
for n in split_names["val"]:
    if ride(n) not in excluded:
        by_ride[ride(n)].append(n)
out["eval_pool"] = {"rides": len(by_ride), "images": sum(len(v) for v in by_ride.values()),
                    "excluded_rides": sorted(set(rides["val"]) & excluded),
                    "m_tilde": m_tilde([len(v) for v in by_ride.values()]),
                    "m_max": max(len(v) for v in by_ride.values())}

# Meta-Train: 3,000 random rides of the 100K test split (labelled here, never used to train the experts)
test_by_ride = collections.defaultdict(list)
for n in split_names["test"]:
    if ride(n) not in excluded:
        test_by_ride[ride(n)].append(n)
meta_rides = sorted(random.sample(sorted(test_by_ride), 3000))
out["meta_train"] = {"source": "100k/test", "rides": len(meta_rides),
                     "images": sum(len(test_by_ride[r]) for r in meta_rides),
                     "m_tilde": m_tilde([len(test_by_ride[r]) for r in meta_rides])}

units = {"val": {r: by_ride[r] for r in sorted(by_ride)}, "meta": {r: test_by_ride[r] for r in meta_rides}}
os.makedirs(CACHE, exist_ok=True)
write_json(os.path.join(CACHE, "p1_units.json"), units)
write_json(os.path.join(CACHE, "p1_report.json"), out)
print(json.dumps(out, indent=1)[:4000])
