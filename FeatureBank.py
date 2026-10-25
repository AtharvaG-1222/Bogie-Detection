import os
import sys
from pathlib import Path
from typing import Tuple, List
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF

# ======================================================================
# CONFIGURATION
# ======================================================================
CHECKPOINT_PATH = "parameters/params.pth"
DATASET_TRAIN_DIR = "dataset/train"
FEATURE_BANK_SAVE_PATH = "parameters/bogie_feature_bank.pt"
NUM_TRAIN_IMAGES = 200
CORESET_BUDGET = 40000
# ======================================================================

CURRENT_DIR = Path(__file__).resolve().parent
for p in [CURRENT_DIR, CURRENT_DIR.parent]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.models.vit.vit import VisionTransformer


def get_optimal_device() -> torch.device:
    """Windows/Linux (CUDA), Mac (MPS), ya CPU ko automatically detect karta hai."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    elif device.type == "mps":
        return "Apple Silicon GPU (MPS)"
    return "CPU"


class BogieIntermediateFeatureExtractor(nn.Module):
    def __init__(self, vit_model: VisionTransformer):
        super().__init__()
        self.vit = vit_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)

        tok_9 = None
        tok_12 = None
        num_blocks = len(self.vit.blocks)
        idx_b1 = min(8, num_blocks - 4)
        idx_b2 = num_blocks - 1

        for i, block in enumerate(self.vit.blocks):
            x = block(x)
            if i == idx_b1:
                tok_9 = x[:, 1:, :]
            elif i == idx_b2:
                tok_12 = x[:, 1:, :]

        tok_12 = self.vit.norm(tok_12)
        fused = torch.cat([tok_9, tok_12 * 0.25], dim=-1)
        return F.normalize(fused, p=2, dim=-1)


class RailwayBogieBankBuilder:
    def __init__(
        self,
        checkpoint_path: str,
        coreset_budget: int = 40000,
        device: torch.device = None
    ):
        self.device = device if device is not None else get_optimal_device()
        self.coreset_budget = coreset_budget

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        print(f"[*] Loading checkpoint from: {checkpoint_path}")
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
            cleaned = k.replace("backbone.", "").replace("module.", "")
            clean_sd[cleaned] = v

        embed_dim = 768
        self.patch_size = 16
        for k, v in clean_sd.items():
            if "patch_embed.proj.weight" in k:
                embed_dim = v.shape[0]
                self.patch_size = v.shape[-1]
                break

        num_heads = max(1, embed_dim // 64)
        vit = VisionTransformer(
            img_size=224,
            patch_size=self.patch_size,
            in_channels=3,
            embed_dim=embed_dim,
            depth=12,
            num_heads=num_heads
        )

        vit.load_state_dict(clean_sd, strict=False)
        vit.to(self.device).eval()

        self.feature_extractor = BogieIntermediateFeatureExtractor(vit).to(self.device).eval()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.memory_bank = None
        self.calibrated_threshold = 0.2200

        print(f"[+] Active Execution Device: {self.device} ({get_device_name(self.device)})")
        print(f"[+] ViT Initialized: embed_dim={embed_dim}, patch_size={self.patch_size}")

    def _get_bogie_bounds(self, img_h: int, img_w: int) -> Tuple[int, int, int, int]:
        y_min = int(img_h * 0.40)
        y_max = int(img_h * 0.68)
        x_min = int(img_w * 0.05)
        x_max = int(img_w * 0.95)
        return y_min, y_max, x_min, x_max

    @torch.no_grad()
    def build_and_save_memory_bank(self, normal_image_paths: List[str], save_path: str):
        extracted_tokens = []
        print(f"[*] Extracting normal reference tokens from {len(normal_image_paths)} images...")

        for idx, p in enumerate(normal_image_paths, start=1):
            img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            y1, y2, x1, x2 = self._get_bogie_bounds(h, w)
            bogie_crop = img[y1:y2, x1:x2]

            crop_rgb = cv2.cvtColor(bogie_crop, cv2.COLOR_BGR2RGB)
            ch, cw = crop_rgb.shape[:2]

            x_steps = list(range(0, max(1, cw - 224 + 1), 112))
            if not x_steps or x_steps[-1] != max(0, cw - 224):
                x_steps.append(max(0, cw - 224))

            batch_crops = []
            for xs in x_steps:
                sub = crop_rgb[:, xs:min(cw, xs + 224)]
                sub = cv2.resize(sub, (224, 224), interpolation=cv2.INTER_LINEAR)
                sub_tensor = TF.to_tensor(Image.fromarray(sub))
                batch_crops.append(self.normalize(sub_tensor))

            if not batch_crops:
                continue

            batch_tensor = torch.stack(batch_crops).to(self.device)
            patches = self.feature_extractor(batch_tensor)
            patches = patches[:, ::2, :].reshape(-1, patches.shape[-1])
            extracted_tokens.append(patches.cpu())

            if idx % 25 == 0 or idx == len(normal_image_paths):
                print(f"    -> Extracted [{idx:03d}/{len(normal_image_paths):03d}] images")

        if not extracted_tokens:
            raise RuntimeError("Failed to build memory bank from train images.")

        full_bank = torch.cat(extracted_tokens, dim=0).to(self.device)
        total_tokens = full_bank.shape[0]

        if total_tokens > self.coreset_budget:
            step = max(1, total_tokens // self.coreset_budget)
            self.memory_bank = full_bank[::step][:self.coreset_budget]
        else:
            self.memory_bank = full_bank

        sample_bank = self.memory_bank[:min(1200, self.memory_bank.shape[0])]
        sims = torch.mm(sample_bank, self.memory_bank.T)
        top_sims, _ = torch.topk(sims, k=6, dim=-1)
        base_distances = (1.0 - top_sims[:, 1:5].mean(dim=-1)).clamp(min=0.0).cpu().numpy()

        b_mean = float(np.mean(base_distances))
        b_std = float(np.std(base_distances))

        calculated = b_mean + 3.0 * b_std
        self.calibrated_threshold = round(max(0.2200, calculated), 4)

        print(f"\n[+] Memory Bank Ready: {self.memory_bank.shape[0]} normal tokens indexed.")
        print(f"[+] Baseline Normal Distance: {b_mean:.4f} (Std: {b_std:.4f})")
        print(f"[+] Calibrated Defect Threshold: {self.calibrated_threshold:.4f}")

        save_file = Path(save_path)
        if not save_file.is_absolute():
            save_file = CURRENT_DIR / save_file
        save_file.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "memory_bank": self.memory_bank.cpu(),
            "calibrated_threshold": self.calibrated_threshold,
            "embed_dim": self.memory_bank.shape[-1],
            "num_tokens": self.memory_bank.shape[0]
        }
        torch.save(payload, str(save_file))
        print(f"[+] Saved offline feature bank to: {save_file}")


if __name__ == "__main__":
    device = get_optimal_device()

    ckpt_path = Path(CHECKPOINT_PATH)
    if not ckpt_path.is_absolute():
        ckpt_path = CURRENT_DIR / ckpt_path

    train_path = Path(DATASET_TRAIN_DIR)
    if not train_path.is_absolute():
        train_path = CURRENT_DIR / train_path

    print("=" * 65)
    print(f"[*] Checkpoint Path : {ckpt_path}")
    print(f"[*] Train Directory : {train_path}")
    print(f"[*] Target Bank File: {FEATURE_BANK_SAVE_PATH}")
    print(f"[*] Enforcing Device: {device} ({get_device_name(device)})")
    print("=" * 65 + "\n")

    builder = RailwayBogieBankBuilder(
        checkpoint_path=str(ckpt_path),
        coreset_budget=CORESET_BUDGET,
        device=device
    )

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    train_images = sorted([
        str(p) for p in train_path.rglob("*")
        if p.suffix.lower() in valid_exts and not p.name.startswith("result_")
    ])

    if not train_images:
        raise FileNotFoundError(f"Training dataset directory is empty: {train_path}")

    selected_images = train_images[:NUM_TRAIN_IMAGES]
    print(f"[*] Found {len(train_images)} images. Using exact {len(selected_images)} for Feature Bank.")

    builder.build_and_save_memory_bank(selected_images, FEATURE_BANK_SAVE_PATH)