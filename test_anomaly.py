import os
import sys
import glob
from pathlib import Path
from typing import Tuple, List, Dict
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
DATASET_TRAIN_DIR = "dataset/train"
DATASET_TEST_DIR = "dataset/test"
OUTPUT_DIR = "anomaly_results"
CORESET_BUDGET = 40000
# =====================================================================

CURRENT_DIR = Path(__file__).resolve().parent
for p in [CURRENT_DIR, CURRENT_DIR.parent]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.models.vit.vit import VisionTransformer


class BogieIntermediateFeatureExtractor(nn.Module):
    """
    Extracts multi-scale representations directly from Blocks 9 and 12.
    - Block 9 captures local textures, surface cracks, and edge degradation.
    - Block 12 context is scaled (0.25x) to eliminate lighting bloom and shadow bias.
    """
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


class RailwayBogieInspectionEngine:
    def __init__(
        self,
        checkpoint_path: str,
        coreset_budget: int = 40000,
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
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

        print(f"[+] Device: {self.device}")
        print(f"[+] ViT: embed_dim={embed_dim}, patch_size={self.patch_size}")

    def _get_bogie_bounds(self, img_h: int, img_w: int) -> Tuple[int, int, int, int]:
        """
        Tight bogie undercarriage bounds:
        - Drops upper 40% (carriage doors, windows, passengers)
        - Drops bottom 32% (rail tracks, ballast stones, ground clutter)
        """
        y_min = int(img_h * 0.40)
        y_max = int(img_h * 0.68)
        x_min = int(img_w * 0.05)
        x_max = int(img_w * 0.95)
        return y_min, y_max, x_min, x_max

    def _build_gaussian_taper(self, h: int, w: int) -> np.ndarray:
        ky = cv2.getGaussianKernel(h, sigma=max(1.0, h / 4.0))
        kx = cv2.getGaussianKernel(w, sigma=max(1.0, w / 4.0))
        taper = ky @ kx.T
        return taper / max(1e-5, float(taper.max()))

    @torch.no_grad()
    def build_memory_bank(self, normal_image_paths: List[str]):
        extracted_tokens = []
        print(f"[*] Extracting normal reference tokens from {len(normal_image_paths)} images...")

        for p in normal_image_paths:
            img = cv2.imread(p)
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

        if not extracted_tokens:
            raise RuntimeError("Failed to build memory bank from train images.")

        full_bank = torch.cat(extracted_tokens, dim=0).to(self.device)
        total_tokens = full_bank.shape[0]

        if total_tokens > self.coreset_budget:
            step = max(1, total_tokens // self.coreset_budget)
            self.memory_bank = full_bank[::step][:self.coreset_budget]
        else:
            self.memory_bank = full_bank

        # Calibrate distance baseline by cross-matching (skipping self-token index 0)
        sample_bank = self.memory_bank[:min(1200, self.memory_bank.shape[0])]
        sims = torch.mm(sample_bank, self.memory_bank.T)
        top_sims, _ = torch.topk(sims, k=6, dim=-1)
        base_distances = (1.0 - top_sims[:, 1:5].mean(dim=-1)).clamp(min=0.0).cpu().numpy()

        b_mean = float(np.mean(base_distances))
        b_std = float(np.std(base_distances))

        # 0.2200 baseline floor prevents healthy bogie components from crossing the threshold
        calculated = b_mean + 3.0 * b_std
        self.calibrated_threshold = round(max(0.2200, calculated), 4)

        print(f"[+] Memory Bank Ready: {self.memory_bank.shape[0]} normal tokens indexed.")
        print(f"[+] Baseline Normal Distance: {b_mean:.4f} (Std: {b_std:.4f})")
        print(f"[+] Calibrated Defect Threshold: {self.calibrated_threshold:.4f}\n")

    def inspect_bogie_image(
        self,
        img_path: str,
        stride: int = 64,
        window_size: int = 224
    ) -> Tuple[np.ndarray, Dict]:
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            return None, None

        orig_h, orig_w = orig_img.shape[:2]
        y_min, y_max, x_min, x_max = self._get_bogie_bounds(orig_h, orig_w)
        bogie_h = y_max - y_min
        bogie_w = x_max - x_min

        accumulated_map = np.zeros((orig_h, orig_w), dtype=np.float32)
        weight_map = np.zeros((orig_h, orig_w), dtype=np.float32)

        x_anchors = list(range(x_min, max(x_min + 1, x_max - window_size + 1), stride))
        if not x_anchors or x_anchors[-1] < x_max - window_size:
            x_anchors.append(max(x_min, x_max - window_size))

        tile_tensors = []
        crop_coords = []
        pil_img = Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))

        for xs in x_anchors:
            actual_x_end = min(x_max, xs + window_size)
            cur_w = actual_x_end - xs
            cur_h = min(bogie_h, window_size)
            ys = y_min + max(0, (bogie_h - cur_h) // 2)
            actual_y_end = ys + cur_h

            tile = pil_img.crop((xs, ys, actual_x_end, actual_y_end)).resize((224, 224))
            tile_tensors.append(self.normalize(TF.to_tensor(tile)))
            crop_coords.append((xs, ys, actual_x_end, actual_y_end, cur_w, cur_h))

        batch_tensor = torch.stack(tile_tensors).to(self.device)

        with torch.no_grad():
            all_patches = self.feature_extractor(batch_tensor)
            sim_matrix = torch.matmul(all_patches, self.memory_bank.T)
            topk_sim, _ = torch.topk(sim_matrix, k=3, dim=-1, largest=True)
            batch_distances = (1.0 - torch.mean(topk_sim, dim=-1)).clamp(min=0.0).cpu().numpy()

        grid_dim = 224 // self.patch_size

        for idx, (xs, ys, xe, ye, cw, ch) in enumerate(crop_coords):
            patch_grid = batch_distances[idx].reshape(grid_dim, grid_dim)
            dense_tile = cv2.resize(patch_grid, (cw, ch), interpolation=cv2.INTER_LINEAR)
            taper = self._build_gaussian_taper(ch, cw)
            accumulated_map[ys:ye, xs:xe] += dense_tile * taper
            weight_map[ys:ye, xs:xe] += taper

        weight_map[weight_map == 0] = 1.0
        smoothed_heatmap = cv2.GaussianBlur(accumulated_map / weight_map, (5, 5), 0)

        roi_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        roi_mask[y_min:y_max, x_min:x_max] = 1
        smoothed_heatmap = smoothed_heatmap * roi_mask

        # Defect segmentation: pixel must meet or exceed calibrated threshold
        defect_mask = ((smoothed_heatmap >= self.calibrated_threshold) & (roi_mask > 0)).astype(np.uint8) * 255
        defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []
        defect_pixel_count = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Suppress minor noise and paint chips smaller than 120 pixels
            if area < 120:
                continue

            cnt_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 1, -1)
            peak_val = float(smoothed_heatmap[cnt_mask > 0].max())

            # Real structural defects must have strong localized divergence
            if peak_val >= (self.calibrated_threshold * 1.10):
                valid_contours.append(cnt)
                defect_pixel_count += area

        total_active_pixels = max(1, int(np.sum(roi_mask)))
        defect_pct = float((defect_pixel_count / total_active_pixels) * 100)
        is_defective = len(valid_contours) > 0

        overlay = orig_img.copy()

        # ONLY draw overlays if actual defect contours exist
        if is_defective:
            tint_layer = np.zeros_like(orig_img)
            cv2.drawContours(tint_layer, valid_contours, -1, (0, 0, 220), -1)
            overlay = np.where(tint_layer > 0, cv2.addWeighted(orig_img, 0.40, tint_layer, 0.60, 0), overlay)
            cv2.drawContours(overlay, valid_contours, -1, (0, 255, 255), 2, cv2.LINE_AA)

        badge_w = 360
        badge_h = 72
        badge_x = 20
        badge_y = 20

        sub_img = overlay[badge_y:badge_y + badge_h, badge_x:badge_x + badge_w]
        dark_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        cv2.addWeighted(sub_img, 0.20, dark_rect, 0.80, 0, sub_img)

        status_color = (0, 0, 255) if is_defective else (0, 220, 0)
        cv2.rectangle(overlay, (badge_x, badge_y), (badge_x + 6, badge_y + badge_h), status_color, -1)
        cv2.rectangle(overlay, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (80, 80, 80), 1)

        status_str = "STATUS: BOGIE DEFECT" if is_defective else "STATUS: BOGIE NORMAL"
        cv2.putText(overlay, status_str, (badge_x + 18, badge_y + 28), cv2.FONT_HERSHEY_DUPLEX, 0.60, status_color, 1, cv2.LINE_AA)

        metrics_str = f"DEFECT AREA: {defect_pct:.2f}% | THRESH: {self.calibrated_threshold:.4f}"
        cv2.putText(overlay, metrics_str, (badge_x + 18, badge_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

        report = {
            "verdict": "DEFECT DETECTED" if is_defective else "NORMAL",
            "defect_pct": defect_pct,
            "regions": len(valid_contours),
            "peak_score": float(smoothed_heatmap.max())
        }

        return overlay, report


if __name__ == "__main__":
    ckpt_path = Path(CHECKPOINT_PATH)
    if not ckpt_path.is_absolute():
        ckpt_path = CURRENT_DIR / ckpt_path

    train_path = Path(DATASET_TRAIN_DIR)
    if not train_path.is_absolute():
        train_path = CURRENT_DIR / train_path

    test_path = Path(DATASET_TEST_DIR)
    if not test_path.is_absolute():
        test_path = CURRENT_DIR / test_path

    out_path = Path(OUTPUT_DIR)
    if not out_path.is_absolute():
        out_path = CURRENT_DIR / out_path
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"[*] Target Checkpoint : {ckpt_path}")
    print(f"[*] Dataset Train     : {train_path}")
    print(f"[*] Dataset Test      : {test_path}")
    print(f"[*] Output Folder     : {out_path}")
    print("=" * 65 + "\n")

    engine = RailwayBogieInspectionEngine(
        checkpoint_path=str(ckpt_path),
        coreset_budget=CORESET_BUDGET
    )

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    train_images = sorted([
        str(p) for p in train_path.rglob("*")
        if p.suffix.lower() in valid_exts and not p.name.startswith("result_")
    ])

    if not train_images:
        raise FileNotFoundError(f"Training dataset directory is empty: {train_path}")

    engine.build_memory_bank(train_images[:60])

    test_images = sorted([
        str(p) for p in test_path.rglob("*")
        if p.suffix.lower() in valid_exts and not p.name.startswith("result_")
    ])

    print(f"[*] Inspecting {len(test_images)} test images...")
    for idx, p in enumerate(test_images, start=1):
        name = Path(p).name
        annotated_img, report = engine.inspect_bogie_image(p)
        if annotated_img is not None:
            out_file = out_path / f"result_{name}"
            cv2.imwrite(str(out_file), annotated_img)
            tag = "[DEFECT]" if report["verdict"] == "DEFECT DETECTED" else "[NORMAL]"
            print(f"[{idx:02d}/{len(test_images):02d}] {name:<30} -> {tag} | Peak: {report['peak_score']:.4f} | Area: {report['defect_pct']:.2f}% | Regions: {report['regions']}")

    print(f"\n[+] Inspection complete! Output folder: {out_path}")

    # for github