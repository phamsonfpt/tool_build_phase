"""Load the project's RT-DETR expert checkpoints (saved with a newer transformers naming scheme)."""
import os
import re
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

ROOT = r"D:\tool_build_phase\ai_router"
HF_ID = "PekingU/rtdetr_r50vd"


def remap_state_dict(sd):
    out = {}
    for k, v in sd.items():
        k2 = k.replace("model.encoder.aifi.", "model.encoder.encoder.")
        k2 = k2.replace(".self_attn.o_proj.", ".self_attn.out_proj.")
        k2 = re.sub(r"\.mlp\.fc([12])\.", r".fc\1.", k2)
        out[k2] = v
    # older transformers keeps top-level copies of the decoder heads
    for k, v in list(out.items()):
        if k.startswith("model.decoder.class_embed.") or k.startswith("model.decoder.bbox_embed."):
            out[k[len("model.decoder."):]] = v
    return out


def load_expert(name, device):
    model = RTDetrForObjectDetection.from_pretrained(HF_ID)
    if name != "pretrained":
        sd = torch.load(os.path.join(ROOT, "experiments", name + ".pth"), map_location="cpu", weights_only=True)
        res = model.load_state_dict(remap_state_dict(sd), strict=True)
        assert not res.missing_keys and not res.unexpected_keys, res
    return model.to(device).eval()


def load_processor():
    return RTDetrImageProcessor.from_pretrained(HF_ID)
