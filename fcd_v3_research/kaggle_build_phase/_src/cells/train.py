%%writefile train.py
# ==============================================================================
# HUẤN LUYỆN MỘT CHUYÊN GIA RT-DETR-R50vd TRÊN 10 LỚP BDD100K (1 hoặc 2 GPU, DDP qua torchrun)
# Biến môi trường (đặt tự động ở notebook):
#   EXPERT_ID   : 1, 2 hoặc 3
#   BENCH=1     : chỉ chạy 40 bước để đo tốc độ, ghi bench_K.json, không lưu mô hình
#   MAX_STEPS   : số bước tối ưu hóa (tính từ tốc độ đo được và quỹ thời gian)
#   NB_START    : thời điểm notebook bắt đầu (epoch giây), dùng cho chốt chặn 11 giờ
# ==============================================================================
import os, json, math, random, hashlib, datetime, time
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import transformers
from transformers import (RTDetrForObjectDetection, RTDetrImageProcessor, TrainingArguments, Trainer,
                          TrainerCallback)
from common import *

EXPERT_ID = int(os.environ["EXPERT_ID"])
SEED = SEEDS[EXPERT_ID]
BENCH = os.environ.get("BENCH") == "1"
MAX_STEPS = int(os.environ.get("MAX_STEPS", 0))
NB_START = float(os.environ.get("NB_START", time.time()))
HARD_STOP_H = float(os.environ.get("HARD_STOP_H", 11.0))  # Kaggle dừng phiên ở 12 giờ
LR, LR_BACKBONE = 1e-4, 1e-5             # cấu hình fine-tune thường dùng cho RT-DETR
PER_GPU_BATCH, EFFECTIVE_BATCH = 4, 16
WORLD = int(os.environ.get("WORLD_SIZE", 1))
ACCUM = max(1, EFFECTIVE_BATCH // (PER_GPU_BATCH * WORLD))  # giữ batch hiệu dụng 16 với 1 hay 2 GPU
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 2))
RESUME_FROM = os.environ.get("RESUME_FROM") or None
OUT_DIR = os.path.join(WORK, f"expert_{EXPERT_ID}" if not BENCH else f"bench_tmp_{EXPERT_ID}")
transformers.set_seed(SEED)

# ------------------------------------------------------------------------------
# SỬA LỖI CỦA TRANSFORMERS (có ở 4.57 và 5.16): trong RTDetrForObjectDetectionLoss, query khử nhiễu (denoising)
# chỉ được tách khỏi các lớp decoder phụ; loss của lớp decoder CUỐI vẫn dùng logits/pred_boxes gồm cả
# query khử nhiễu (300 + 198 = 498 query). Bộ ghép Hungarian vì thế ghép GT vào query khử nhiễu (vốn nằm sát GT),
# 300 query thật của lớp cuối chỉ nhận mục tiêu "nền", và điểm phân loại khi suy luận bị ép về ~0.
# (Đây là nguyên nhân chuyên gia 2 lần đầu có 0 dự đoán trên val.) Code gốc RT-DETR tách trước khi ghép.
# ------------------------------------------------------------------------------
import transformers.loss.loss_utils as _hf_loss_utils
_HF_RTDETR_LOSS = _hf_loss_utils.LOSS_MAPPING["RTDetrForObjectDetection"]
_LOSS_FIX_SEEN = {"n": 0}

def rtdetr_loss_fixed(logits, labels, device, pred_boxes, config, outputs_class=None, outputs_coord=None,
                      *args, denoising_meta_values=None, **kwargs):
    if denoising_meta_values is not None:
        n_dn = denoising_meta_values["dn_num_split"][0]  # thứ tự: [query khử nhiễu, query thật]
        logits, pred_boxes = logits[:, n_dn:], pred_boxes[:, n_dn:]
        if _LOSS_FIX_SEEN["n"] == 0:
            assert logits.shape[1] == config.num_queries, (logits.shape, config.num_queries)
            print(f"[V] Đã tách {n_dn} query khử nhiễu khỏi loss của lớp decoder cuối "
                  f"(còn {logits.shape[1]} query thật).", flush=True)
        _LOSS_FIX_SEEN["n"] += 1
    return _HF_RTDETR_LOSS(logits, labels, device, pred_boxes, config, outputs_class, outputs_coord,
                           *args, denoising_meta_values=denoising_meta_values, **kwargs)

_hf_loss_utils.LOSS_MAPPING["RTDetrForObjectDetection"] = rtdetr_loss_fixed


class BDDDet(Dataset):
    """Augmentation ngẫu nhiên (lật ngang, độ sáng/tương phản/bão hòa) vừa tăng độ bền vừa là nguồn
    đa dạng giữa các chuyên gia; nó không loại bỏ object nào."""
    def __init__(self, items, processor, augment):
        self.items, self.processor, self.augment = items, processor, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        image = Image.open(rec["file"]).convert("RGB")
        W, _ = image.size
        boxes = [list(b) for b in rec["boxes"]]
        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                boxes = [[W - x2, y1, W - x1, y2, c] for x1, y1, x2, y2, c in boxes]
            for enh in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                image = enh(image).enhance(random.uniform(0.8, 1.2))
        anns = [{"bbox": [x1, y1, x2 - x1, y2 - y1], "category_id": int(c), "area": (x2 - x1) * (y2 - y1), "iscrowd": 0}
                for x1, y1, x2, y2, c in boxes]
        enc = self.processor(images=image, annotations={"image_id": idx, "annotations": anns}, return_tensors="pt")
        return {"pixel_values": enc["pixel_values"][0], "labels": enc["labels"][0]}


def collate_fn(batch):
    return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]), "labels": [b["labels"] for b in batch]}


def expand_rfs(items, seed):
    """Lặp ảnh theo hệ số RFS với làm tròn ngẫu nhiên (cố định theo seed của chuyên gia)."""
    rng = np.random.default_rng(seed)
    out = []
    for rec in items:
        r = rec.get("repeat", 1.0)
        out.extend([rec] * (int(math.floor(r)) + int(rng.random() < r - math.floor(r))))
    return out


def build_model():
    model = RTDetrForObjectDetection.from_pretrained(
        "PekingU/rtdetr_r50vd", id2label=ID2CLASS, label2id=CLASS2ID,
        num_labels=len(BDD_CLASSES), ignore_mismatched_sizes=True)
    # Khởi tạo hàng của đầu phân loại từ lớp COCO tương ứng để hội tụ nhanh hơn.
    # rider và traffic sign không có lớp COCO tương ứng nên giữ khởi tạo mặc định.
    # Config của PekingU dùng 'motorbike' thay cho 'motorcycle', nên mỗi lớp có danh sách tên thay thế.
    pairs = {'pedestrian': ['person'], 'car': ['car'], 'truck': ['truck'], 'bus': ['bus'], 'train': ['train'],
             'motorcycle': ['motorcycle', 'motorbike'], 'bicycle': ['bicycle'], 'traffic light': ['traffic light']}
    try:
        coco = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")
        c2i = {v: int(k) for k, v in coco.config.id2label.items()}
        done = []
        with torch.no_grad():
            for bdd, names in pairs.items():
                j = next((c2i[n] for n in names if n in c2i), None)
                if j is None:
                    continue
                i = CLASS2ID[bdd]
                for l in range(len(model.model.decoder.class_embed)):
                    model.model.decoder.class_embed[l].weight[i] = coco.model.decoder.class_embed[l].weight[j]
                    model.model.decoder.class_embed[l].bias[i] = coco.model.decoder.class_embed[l].bias[j]
                model.model.enc_score_head.weight[i] = coco.model.enc_score_head.weight[j]
                model.model.enc_score_head.bias[i] = coco.model.enc_score_head.bias[j]
                done.append(bdd)
        del coco
        print(f"[V] Đã khởi tạo đầu phân loại từ COCO cho {len(done)}/10 lớp: {done}")
    except Exception as e:  # không chặn việc train
        print(f"[!] Không khởi tạo được từ COCO ({e}); dùng khởi tạo mặc định.")
    return model


class TimeGuard(TrainerCallback):
    """Dừng an toàn (lưu checkpoint rồi thoát) khi notebook chạy gần tới giới hạn 12 giờ của Kaggle.
    Mọi tiến trình DDP thống nhất quyết định qua all_reduce để không bị treo."""
    def on_step_end(self, args, state, control, **kw):
        if state.global_step % 50 and state.global_step != 1:
            return
        flag = torch.tensor([float(time.time() - NB_START > HARD_STOP_H * 3600)],
                            device="cuda" if torch.cuda.is_available() else "cpu")
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item() > 0:
            print(f"[!] Gần hết thời gian phiên Kaggle: dừng ở bước {state.global_step} và lưu checkpoint.")
            control.should_save = True; control.should_training_stop = True


class Speed(TrainerCallback):
    """Đo tốc độ (bước tối ưu hóa mỗi giây) từ bước 10 đến bước cuối, dùng cho chế độ BENCH."""
    def __init__(self): self.t = {}
    def on_step_end(self, args, state, control, **kw):
        self.t[state.global_step] = time.time()


class NanGuard(TrainerCallback):
    """Dừng ngay nếu loss được ghi log không hữu hạn (NaN/Inf). Loss log đã được gộp giữa các GPU,
    nên mọi tiến trình nhận cùng giá trị và dừng cùng lúc."""
    def on_log(self, args, state, control, logs=None, **kw):
        v = (logs or {}).get("loss")
        if v is not None and not math.isfinite(v):
            print(f"[X] Loss = {v} ở bước {state.global_step}: dừng train.", flush=True)
            control.should_training_stop = True


PROBE_EVERY = int(os.environ.get("PROBE_EVERY", 2000))
PROBE_MIN_STEP = int(os.environ.get("PROBE_MIN_STEP", 4000))


class ScoreProbe(TrainerCallback):
    """Kiểm tra sớm: mỗi PROBE_EVERY bước, chạy suy luận trên 32 ảnh TRAIN cố định (không dùng val) và ghi
    điểm tin cậy cao nhất trung bình + số box > 0.3 vào expert_K_probe.csv. Một mô hình đang học bình thường
    giữ điểm này cao (ban đầu khoảng 0.85). Từ bước PROBE_MIN_STEP, nếu điểm < 0.1 thì mô hình đang suy biến
    (như lỗi loss đã sửa ở trên): dừng ngay để không tốn quota. Mọi GPU chạy cùng phép đo trên cùng trọng số
    nên đưa ra cùng quyết định, không cần trao đổi."""
    def __init__(self, items, processor):
        self.proc = processor
        self.ims = [Image.open(r["file"]).convert("RGB") for r in items[:32]]
        self.path = os.path.join(WORK, f"expert_{EXPERT_ID}_probe.csv")

    def on_step_end(self, args, state, control, model=None, **kw):
        if state.global_step % PROBE_EVERY:
            return
        m = model.module if hasattr(model, "module") else model
        was_training = m.training
        m.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            inp = self.proc(images=self.ims, return_tensors="pt").to(next(m.parameters()).device)
            out = m(pixel_values=inp["pixel_values"])
        s = out.logits.float().sigmoid().amax(dim=-1)                 # [ảnh, query]
        mean_max = s.amax(dim=1).mean().item(); n_box = int((s > 0.3).sum())
        if was_training:
            m.train()
        if state.is_world_process_zero:
            new = not os.path.exists(self.path)
            with open(self.path, "a") as f:
                f.write(("step,mean_max_score,boxes_over_0.3\n" if new else "") + f"{state.global_step},{mean_max:.4f},{n_box}\n")
            print(f"[probe] bước {state.global_step}: điểm cao nhất TB {mean_max:.3f}, box > 0.3: {n_box}", flush=True)
        if state.global_step >= PROBE_MIN_STEP and (not math.isfinite(mean_max) or mean_max < 0.1):
            if state.is_world_process_zero:
                print(f"[X] Điểm tin cậy suy biến ({mean_max:.3f} < 0.1): dừng train. Xem expert_{EXPERT_ID}_probe.csv.", flush=True)
            control.should_training_stop = True


def weights_sha256(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode()); h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    with open(COMPACT["train"]) as f:
        items = json.load(f)
    with open(DATA_STATS) as f:
        stats = json.load(f)
    train_items = expand_rfs(items, SEED)
    with open(COMPACT["val"]) as f:
        val_monitor = [r for r in json.load(f) if r["monitor"]]  # 1.000 ảnh val cố định, CHỈ để vẽ đường cong loss
    processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
    model = build_model()

    backbone = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": LR_BACKBONE}, {"params": rest, "lr": LR}],
                                  lr=LR, weight_decay=1e-4)
    steps = 40 if BENCH else MAX_STEPS
    assert steps > 0, "Thiếu MAX_STEPS: hãy chạy cell đo tốc độ trước."
    args = TrainingArguments(
        output_dir=OUT_DIR,
        seed=SEED, data_seed=SEED,            # BẮT BUỘC: nếu thiếu, Trainer đặt lại seed = 42 cho mọi chuyên gia
        max_steps=steps,
        per_device_train_batch_size=PER_GPU_BATCH, gradient_accumulation_steps=ACCUM,
        learning_rate=LR, lr_scheduler_type="cosine", warmup_steps=min(1000, max(1, steps // 20)),
        max_grad_norm=0.1, fp16=torch.cuda.is_available(),
        # Val CHỈ ĐỂ XEM: đo loss trên 1.000 ảnh val mỗi 2.000 bước để thấy overfit. Không chọn checkpoint theo
        # val (load_best_model_at_end=False) và không có early stopping: mô hình lưu ra luôn là bản cuối cùng.
        eval_strategy="no" if BENCH else "steps", eval_steps=int(os.environ.get("EVAL_STEPS", 2000)), prediction_loss_only=True,
        per_device_eval_batch_size=PER_GPU_BATCH, load_best_model_at_end=False,
        save_strategy="no" if BENCH else "steps", save_steps=2000, save_total_limit=2,
        logging_steps=50, report_to="none", disable_tqdm=False,
        dataloader_num_workers=NUM_WORKERS, dataloader_persistent_workers=NUM_WORKERS > 0,
        ddp_find_unused_parameters=False, remove_unused_columns=False,
    )
    speed = Speed()
    trainer = Trainer(model=model, args=args, train_dataset=BDDDet(train_items, processor, augment=True),
                      eval_dataset=None if BENCH else BDDDet(val_monitor, processor, augment=False),
                      data_collator=collate_fn, optimizers=(optimizer, None),
                      callbacks=[speed] if BENCH else [TimeGuard(), NanGuard(), ScoreProbe(items, processor)])
    eff_batch = PER_GPU_BATCH * ACCUM * WORLD
    if trainer.is_world_process_zero():
        print("=" * 70)
        print(f"[*] CHUYÊN GIA {EXPERT_ID} | seed={SEED} | GPU={WORLD} | batch hiệu dụng={eff_batch} | bước={steps}"
              f" | ≈{steps * eff_batch / len(train_items):.2f} epoch | {'ĐO TỐC ĐỘ' if BENCH else 'TRAIN'}")
        print("=" * 70)

    trainer.train(resume_from_checkpoint=None if BENCH else RESUME_FROM)

    if trainer.is_world_process_zero():
        if BENCH:
            t = speed.t; a, b = 10, max(t)
            sps = (b - a) / (t[b] - t[a])
            with open(os.path.join(WORK, f"bench_{EXPERT_ID}.json"), "w") as f:
                json.dump({"steps_per_sec": sps, "effective_batch": eff_batch, "world": WORLD,
                           "images_per_epoch": len(train_items)}, f)
            print(f"[V] Tốc độ: {sps:.3f} bước/giây ({sps * eff_batch:.1f} ảnh/giây)")
        else:
            final_dir = os.path.join(WORK, f"expert_{EXPERT_ID}_final")
            trainer.save_model(final_dir); processor.save_pretrained(final_dir)
            sd = trainer.model.module.state_dict() if hasattr(trainer.model, "module") else trainer.model.state_dict()
            pth = os.path.join(WORK, f"expert_{EXPERT_ID}.pth")
            torch.save(sd, pth)
            manifest = dict(expert_id=EXPERT_ID, seed=SEED, data_seed=SEED, max_steps=steps,
                            steps_done=trainer.state.global_step,
                            epochs_done=trainer.state.global_step * eff_batch / len(train_items),
                            lr=LR, lr_backbone=LR_BACKBONE, effective_batch=eff_batch, world=WORLD,
                            classes=BDD_CLASSES, rfs_t=stats["rfs_t"], images_per_epoch=len(train_items),
                            split_rule=stats["split_rule"], val_usage="chỉ để xem: không chọn checkpoint, không chỉnh siêu tham số",
                            rtdetr_loss_fix="tách query khử nhiễu khỏi loss của lớp decoder cuối (lỗi transformers)",
                            stopped_early=trainer.state.global_step < steps,
                            n_images=stats["n_images"], train_filtered=stats["train_filtered"],
                            kept_boxes_train=stats["kept_boxes"]["train"], weights_sha256=weights_sha256(sd),
                            last_log=trainer.state.log_history[-1],
                            transformers=transformers.__version__, torch=torch.__version__,
                            finished_utc=datetime.datetime.utcnow().isoformat())
            with open(os.path.join(WORK, f"expert_{EXPERT_ID}_manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)
            # Đường cong loss: train (mỗi 50 bước) và val (mỗi 2.000 bước), chỉ để xem
            with open(os.path.join(WORK, f"expert_{EXPERT_ID}_loss_curve.csv"), "w") as f:
                f.write("step,train_loss,val_loss\n")
                for h in trainer.state.log_history:
                    if "loss" in h or "eval_loss" in h:
                        f.write(f"{h.get('step')},{h.get('loss', '')},{h.get('eval_loss', '')}\n")
            print(f"[V] Đã lưu {pth}, {final_dir}/ và expert_{EXPERT_ID}_manifest.json")
            print(f"    {manifest['epochs_done']:.2f} epoch | sha256 trọng số: {manifest['weights_sha256'][:16]}…")
