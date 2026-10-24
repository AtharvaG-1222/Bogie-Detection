import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

# =====================================================================
# CONFIGURATION (Exact as provided)
# =====================================================================
CHECKPOINT_PATH = "parameters/params.pth"
TEST_IMAGE_PATH = "dataset/test/img-24-_jpg.rf.19fa9d7d318893b9491ffdc575cb3284.jpg"
OUTPUT_IMAGE = "Bogie Feature Representation.jpg"
IMG_SIZE = 224
# =====================================================================

CURRENT_DIR = Path(__file__).resolve().parent
for p in [CURRENT_DIR, CURRENT_DIR.parent]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.models.vit.vit import VisionTransformer


def build_vit_model(checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint nahi mila: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "backbone_state_dict" in ckpt:
            sd = ckpt["backbone_state_dict"]
        elif "student_state_dict" in ckpt:
            sd = ckpt["student_state_dict"]
        elif "student_backbone" in ckpt:
            sd = ckpt["student_backbone"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    clean_sd = {}
    for k, v in sd.items():
        if "head" in k:
            continue
        clean_sd[k.replace("backbone.", "").replace("module.", "")] = v

    embed_dim = 768
    patch_size = 16
    for k, v in clean_sd.items():
        if "patch_embed.proj.weight" in k:
            embed_dim = v.shape[0]
            patch_size = v.shape[-1]
            break

    num_heads = max(1, embed_dim // 64)
    vit = VisionTransformer(
        img_size=IMG_SIZE,
        patch_size=patch_size,
        in_channels=3,
        embed_dim=embed_dim,
        depth=12,
        num_heads=num_heads
    )
    vit.load_state_dict(clean_sd, strict=False)
    vit.to(device).eval()
    return vit, patch_size


def letterbox_image(image: np.ndarray, target_size: int = 224):
    """Aspect ratio maintain karte hue square mein pad karta hai taaki parts squish na hon."""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return canvas, (pad_top, pad_top + new_h, pad_left, pad_left + new_w)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on: {device}")

    ckpt_path = Path(CHECKPOINT_PATH)
    if not ckpt_path.is_absolute():
        ckpt_path = CURRENT_DIR / ckpt_path

    vit_model, patch_size = build_vit_model(str(ckpt_path), device)

    target_img_path = Path(TEST_IMAGE_PATH)
    if not target_img_path.is_absolute():
        target_img_path = CURRENT_DIR / target_img_path

    if target_img_path.is_dir():
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        candidates = sorted([
            p for p in target_img_path.rglob("*")
            if p.suffix.lower() in valid_exts and not p.name.startswith("result_")
        ])
        if not candidates:
            raise FileNotFoundError("dataset/test folder mein koi valid image nahi mili.")
        image_file = str(candidates[0])
    else:
        image_file = str(target_img_path)

    print(f"[*] Processing: {image_file}")
    img = cv2.imread(image_file)
    if img is None:
        raise ValueError(f"Image load nahi ho saki: {image_file}")

    # =================================================================
    # EXPANDED FULL BOGIE ROI (Covers Complete Wheels + Full Frame)
    # =================================================================
    h, w = img.shape[:2]
    y_min, y_max = int(h * 0.30), int(h * 0.80)
    x_min, x_max = int(w * 0.02), int(w * 0.98)
    bogie_crop = img[y_min:y_max, x_min:x_max]

    orig_h, orig_w = bogie_crop.shape[:2]

    # Letterbox processing
    crop_rgb = cv2.cvtColor(bogie_crop, cv2.COLOR_BGR2RGB)
    padded_rgb, (pt, pb, pl, pr) = letterbox_image(crop_rgb, target_size=IMG_SIZE)

    pil_img = Image.fromarray(padded_rgb)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_tensor = normalize(TF.to_tensor(pil_img)).unsqueeze(0).to(device)

    # Feature Extraction
    with torch.no_grad():
        b, c, th, tw = input_tensor.shape
        x = vit_model.patch_embed(input_tensor)
        cls_token = vit_model.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_token, x), dim=1) + vit_model.pos_embed
        x = vit_model.pos_drop(x)

        for blk in vit_model.blocks:
            x = blk(x)

        patch_features = vit_model.norm(x)[:, 1:, :]  # [1, 196, 768]
        feat_np = patch_features.squeeze(0).cpu().numpy()

    # PCA Projection to RGB
    l2_norms = np.linalg.norm(feat_np, axis=-1, keepdims=True) + 1e-6
    feat_norm = feat_np / l2_norms

    # Subtract median across tokens
    spatial_median = np.median(feat_norm, axis=0, keepdims=True)
    centered = feat_norm - spatial_median

    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    pca_3d = np.dot(centered, vt[:3, :].T)  # [196, 3]

    for ch in range(3):
        low, high = np.percentile(pca_3d[:, ch], (5, 95))
        if high - low > 1e-6:
            pca_3d[:, ch] = np.clip((pca_3d[:, ch] - low) / (high - low), 0.0, 1.0)
        else:
            pca_3d[:, ch] = 0.5

    pca_rgb = (pca_3d * 255.0).astype(np.uint8)

    # 14x14 Reshape
    grid_dim = IMG_SIZE // patch_size
    feature_map_14x14 = pca_rgb.reshape(grid_dim, grid_dim, 3)

    # Upscale to 224x224
    feature_map_224 = cv2.resize(feature_map_14x14, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

    # Unpad back to valid crop region
    feature_map_unpadded = feature_map_224[pt:pb, pl:pr]
    feature_map_bgr = cv2.cvtColor(feature_map_unpadded, cv2.COLOR_RGB2BGR)
    feature_map_full = cv2.resize(feature_map_bgr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # Semi-transparent Overlay (50% blend)
    alpha = 0.50
    overlay = cv2.addWeighted(bogie_crop, 1 - alpha, feature_map_full, alpha, 0)

    # Clean Typography & Borders
    cv2.rectangle(bogie_crop, (0, 0), (orig_w, 45), (20, 20, 20), -1)
    cv2.putText(bogie_crop, "INPUT: BOGIE MECHANICAL ROI", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(overlay, (0, 0), (orig_w, 45), (20, 20, 20), -1)
    cv2.putText(overlay, "ViT FEATURE REPRESENTATION", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

    combined = np.vstack([bogie_crop, overlay])
    cv2.imwrite(OUTPUT_IMAGE, combined)
    print(f"\n[+] Presentation deliverable saved: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()