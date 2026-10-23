import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF

# =====================================================================
# CONFIGURATION
# =====================================================================
CHECKPOINT_PATH = "parameters/params.pth"
TEST_IMAGE_PATH = "dataset/test"  # Direct image path ya test folder
OUTPUT_IMAGE = "vit_learned_representation.jpg"
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
        img_size=224,
        patch_size=patch_size,
        in_channels=3,
        embed_dim=embed_dim,
        depth=12,
        num_heads=num_heads
    )
    vit.load_state_dict(clean_sd, strict=False)
    vit.to(device).eval()
    return vit, patch_size


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on: {device}")

    ckpt_path = Path(CHECKPOINT_PATH)
    if not ckpt_path.is_absolute():
        ckpt_path = CURRENT_DIR / ckpt_path

    vit_model, patch_size = build_vit_model(str(ckpt_path), device)

    # Image dhoondhna
    target_img_path = Path(TEST_IMAGE_PATH)
    if not target_img_path.is_absolute():
        target_img_path = CURRENT_DIR / target_img_path

    if target_img_path.is_dir():
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        candidates = sorted([p for p in target_img_path.rglob("*") if p.suffix.lower() in valid_exts and not p.name.startswith("result_")])
        if not candidates:
            raise FileNotFoundError("dataset/test folder mein koi valid image nahi mili.")
        image_file = str(candidates[0])
    else:
        image_file = str(target_img_path)

    print(f"[*] Visualizing ViT Feature Response for: {image_file}")
    img = cv2.imread(image_file)
    if img is None:
        raise ValueError(f"Image load nahi ho saki: {image_file}")

    h, w = img.shape[:2]
    # Bogie ROI isolate karna
    y_min, y_max = int(h * 0.42), int(h * 0.70)
    x_min, x_max = int(w * 0.05), int(w * 0.95)
    bogie_crop = img[y_min:y_max, x_min:x_max]

    # Preprocessing
    crop_rgb = cv2.cvtColor(bogie_crop, cv2.COLOR_BGR2RGB)
    resized_crop = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    pil_img = Image.fromarray(resized_crop)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_tensor = normalize(TF.to_tensor(pil_img)).unsqueeze(0).to(device)

    # ViT Forward Pass (Feature Energy Map from Block 9 & 12)
    with torch.no_grad():
        b, c, th, tw = input_tensor.shape
        x = vit_model.patch_embed(input_tensor)
        cls_token = vit_model.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_token, x), dim=1) + vit_model.pos_embed
        x = vit_model.pos_drop(x)

        tok_9 = None
        for i, blk in enumerate(vit_model.blocks):
            x = blk(x)
            if i == 8:
                tok_9 = x[:, 1:, :]

        tok_12 = vit_model.norm(x[:, 1:, :])
        fused = torch.cat([tok_9, tok_12 * 0.25], dim=-1)

        # Patch representation energy across spatial grid
        grid_dim = 224 // patch_size
        feature_energy = fused.norm(dim=-1).squeeze(0).reshape(grid_dim, grid_dim).cpu().numpy()

    # Normalization & Color Heatmap
    feature_energy = (feature_energy - feature_energy.min()) / (feature_energy.max() - feature_energy.min() + 1e-6)
    feature_energy = cv2.GaussianBlur(feature_energy, (3, 3), 0)
    
    heatmap = cv2.resize((feature_energy * 255).astype(np.uint8), (bogie_crop.shape[1], bogie_crop.shape[0]))
    colored_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Side-by-Side Presentation Layout
    overlay = cv2.addWeighted(bogie_crop, 0.55, colored_map, 0.45, 0)
    
    # Text Labels add karna
    cv2.putText(bogie_crop, "INPUT BOGIE ROI", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(overlay, "ViT LEARNED FEATURE ATTENTION", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    combined = np.vstack([bogie_crop, overlay])
    cv2.imwrite(OUTPUT_IMAGE, combined)
    print(f"\n[+] Visualization successful! Image saved as: {OUTPUT_IMAGE}")
    print("[+] Is image ko presentation slide par lagao ViT feature activation dikhane ke liye.")


if __name__ == "__main__":
    main()