"""Inference engine: preprocessing, prediction, Grad-CAM. No UI code here."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from model_defs import BUILDERS, disable_inplace

HERE = Path(__file__).resolve().parent
CFG = json.load(open(HERE / "inference_config.json"))
CLASSES = CFG["classes"]
MEAN = np.array(CFG["mean"], dtype=np.float32)
STD = np.array(CFG["std"], dtype=np.float32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── preprocessing: IDENTICAL to training Block 2 (resize -> LAB CLAHE -> blur L) ──
def preprocess_image(pil: Image.Image) -> np.ndarray:
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (224, 224))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    l = cv2.GaussianBlur(l, (3, 3), 0)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)            # uint8 (224,224,3)


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


# ── model loading ────────────────────────────────────────────
def available_models():
    return [m for m in CFG["weights"] if (HERE / CFG["weights_dir"] / CFG["weights"][m]).exists()]


def load_model(name: str):
    model = disable_inplace(BUILDERS[name]())
    state = torch.load(HERE / CFG["weights_dir"] / CFG["weights"][name], map_location="cpu")
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


@torch.no_grad()
def predict(model, rgb):
    return F.softmax(model(to_tensor(rgb)), 1)[0].cpu().numpy()


# ── Grad-CAM (forward hook + autograd.grad; works for CNNs and ViT/DeiT) ──
def _vit_reshape(m):
    npfx, (gh, gw) = getattr(m, "num_prefix_tokens", 1), m.patch_embed.grid_size
    def f(t):
        t = t[:, npfx:, :]
        return t.reshape(t.size(0), gh, gw, t.size(2)).permute(0, 3, 1, 2)
    return f


def _target(name, m):
    if name == "ResNet-18":                          return m.layer4[-1], None
    if name in ("SqueezeNet1.1", "EfficientNet-B0"): return m.features[-1], None
    if name in ("ViT-Tiny", "DeiT-Tiny"):            return m.blocks[-1].norm1, _vit_reshape(m)
    if name == "AGCTNet":                            return m.cag, None
    raise KeyError(name)


def gradcam(model, name, rgb, class_idx=None):
    """Returns (cam in [0,1] at layer resolution, class_idx)."""
    layer, reshape = _target(name, model)
    store = {}
    h = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    try:
        x = to_tensor(rgb).requires_grad_(True)
        with torch.enable_grad():
            logits = model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(1)[0])
            g = torch.autograd.grad(logits[0, class_idx], store["a"])[0]
        a = store["a"]
        if reshape:
            a, g = reshape(a), reshape(g)
        raw = (g.mean((2, 3), keepdim=True) * a).sum(1)[0].detach().cpu().numpy()
        cam = np.maximum(raw, 0)
        if cam.max() < 1e-8:          # all-negative map: fall back to the signed map
            cam = raw
    finally:
        h.remove()
    cam = cam - cam.min()
    return cam / (cam.max() + 1e-8), class_idx


def overlay(rgb, cam, alpha=0.45):
    up = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    up = np.clip(up, 0, 1)
    heat = cv2.cvtColor(cv2.applyColorMap((up * 255).astype(np.uint8), cv2.COLORMAP_JET),
                        cv2.COLOR_BGR2RGB)
    return (np.clip((1 - alpha) * rgb + alpha * heat, 0, 255)).astype(np.uint8)