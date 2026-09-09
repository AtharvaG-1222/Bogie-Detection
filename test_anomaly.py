import os
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


# ==============================================================================
# 1. VISION TRANSFORMER BACKBONE
# ==============================================================================
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim=768, num_heads=12, drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]


# ==============================================================================
# 2. CHECKPOINT LOADER WITH COMPATIBILITY REMAPPING
# ==============================================================================
def load_trained_backbone(weights_path: str, device: torch.device):
    model = VisionTransformer().to(device)
    state_dict = torch.load(weights_path, map_location=device)

    if "student_state_dict" in state_dict:
        state_dict = state_dict["student_state_dict"]

    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith("head."):
            continue
        cleaned = k.replace("backbone.", "")
        if ".mlp.fc1." in cleaned:
            cleaned = cleaned.replace(".mlp.fc1.", ".mlp.0.")
        elif ".mlp.fc2." in cleaned:
            cleaned = cleaned.replace(".mlp.fc2.", ".mlp.3.")
        clean_state[cleaned] = v

    model.load_state_dict(clean_state, strict=True)
    model.eval()
    print(f"[*] Backbone weights loaded from: {weights_path}")
    return model


# ==============================================================================
# 3. SPATIAL MEMORY BANK (3x3 SEARCH COMPATIBLE)
# ==============================================================================
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def build_spatial_memory_bank(model, train_folder: str, device: torch.device, num_images: int = 100):
    model.eval()
    transform = get_transform()

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    # Exclude result files to avoid corrupting bank
    image_paths = [
        p for p in Path(train_folder).rglob("*")
        if p.suffix.lower() in valid_exts and not p.name.startswith("result_")
    ][:num_images]

    if not image_paths:
        raise FileNotFoundError(f"No reference training images in: {train_folder}")

    print(f"[*] Building spatial memory bank from {len(image_paths)} normal training images...")
    all_patches = []

    for path in image_paths:
        orig = cv2.imread(str(path))
        if orig is None:
            continue
        rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)

        _, patches = model(tensor)
        patches = F.normalize(patches.squeeze(0), dim=-1, p=2)
        all_patches.append(patches.unsqueeze(1))  # [196, 1, 768]

    # Stored as [196, N_images, 768]
    spatial_bank = torch.cat(all_patches, dim=1).to(device)
    print(f"[+] Spatial bank created: {spatial_bank.shape[1]} reference bogies mapped across 196 patches.")
    return spatial_bank


# ==============================================================================
# 4. TOLERANT 3x3 NEIGHBORHOOD INFERENCE
# ==============================================================================
@torch.no_grad()
def evaluate_image(model, spatial_bank, image_path: str, device: torch.device, threshold: float = 0.40):
    model.eval()
    orig_bgr = cv2.imread(image_path)
    if orig_bgr is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    orig_h, orig_w = orig_bgr.shape[:2]

    # If image is black & white, equalize histogram to prevent channel collapse
    gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
    is_grayscale = np.mean(np.abs(orig_bgr[:, :, 0] - orig_bgr[:, :, 1])) < 5

    if is_grayscale:
        eq_gray = cv2.equalizeHist(gray)
        rgb_input = cv2.cvtColor(eq_gray, cv2.COLOR_GRAY2RGB)
    else:
        rgb_input = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    transform = get_transform()
    tensor = transform(Image.fromarray(rgb_input)).unsqueeze(0).to(device)

    # 1. Extract test features
    _, test_patches = model(tensor)  # [1, 196, 768]
    test_patches = F.normalize(test_patches.squeeze(0), dim=-1, p=2)  # [196, 768]

    # 2. Local 3x3 Neighborhood Cosine Distance
    # Reshape to grid representation: 14x14 patches
    N = spatial_bank.shape[1]
    test_grid = test_patches.view(14, 14, 768)
    bank_grid = spatial_bank.view(14, 14, N, 768)

    patch_anomaly = np.zeros((14, 14), dtype=np.float32)

    for r in range(14):
        for c in range(14):
            # 3x3 search window bounded by spatial limits
            r_min, r_max = max(0, r - 1), min(14, r + 2)
            c_min, c_max = max(0, c - 1), min(14, c + 2)

            # Local candidate patch pool from normal reference images: [Window_size * N, 768]
            local_bank = bank_grid[r_min:r_max, c_min:c_max].reshape(-1, 768)

            # Cosine similarity against test patch (r, c)
            sims = torch.mv(local_bank, test_grid[r, c])
            max_sim = torch.max(sims).item()
            patch_anomaly[r, c] = max(0.0, 1.0 - max_sim)

    # Calculate overall image score: mean of the top 5% most anomalous patches
    flat_scores = patch_anomaly.flatten()
    k = max(1, int(0.05 * len(flat_scores)))
    anomaly_score = float(np.mean(np.sort(flat_scores)[-k:]))

    # 3. Non-relative heatmap rendering
    heatmap_resized = cv2.resize(patch_anomaly, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    clamped_map = np.clip(heatmap_resized / (threshold + 1e-6), 0.0, 1.0)
    colored_map = cv2.applyColorMap((clamped_map * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Only highlight regions where anomaly distance exceeds 60% of the threshold
    active_mask = (clamped_map > 0.60).astype(np.float32)[:, :, np.newaxis]
    blended = (orig_bgr * (1.0 - (active_mask * 0.45)) + colored_map * (active_mask * 0.45)).astype(np.uint8)

    # 4. Status overlay
    is_defect = anomaly_score >= threshold
    status = f"DEFECT | Score: {anomaly_score:.4f}" if is_defect else f"NORMAL | Score: {anomaly_score:.4f}"
    color = (0, 0, 255) if is_defect else (0, 255, 0)

    cv2.putText(blended, status, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.putText(blended, f"Threshold: {threshold:.4f}", (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    return blended, anomaly_score, is_defect


# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Inference Device: {device}")

    project_root = Path(__file__).resolve().parent
    weights_path = project_root / "parameters" / "vit_bogie_dino.pth"
    train_dir = project_root / "dataset" / "train"
    test_dir = project_root / "dataset" / "test"
    output_dir = project_root / "anomaly_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    model = load_trained_backbone(str(weights_path), device)
    spatial_bank = build_spatial_memory_bank(model, str(train_dir), device, num_images=60)

    # Baseline verification on training samples
    print("\n[*] Validating on training samples to verify baseline distance...")
    train_samples = [
        p for p in train_dir.rglob("*")
        if p.suffix.lower() in valid_exts and not p.name.startswith("result_")
    ][:5]

    for p in train_samples:
        _, score, _ = evaluate_image(model, spatial_bank, str(p), device, threshold=0.40)
        print(f"  Training sample [{p.name}]: Distance = {score:.4f}")

    # Set threshold (0.40 is standard; adjust based on training sample distance)
    DETECTION_THRESHOLD = 0.40

    test_images = [
        p for p in test_dir.rglob("*")
        if p.suffix.lower() in valid_exts and not p.name.startswith("result_")
    ]

    print(f"\n[*] Evaluating {len(test_images)} test images...")
    for img_path in test_images:
        res, score, defect = evaluate_image(model, spatial_bank, str(img_path), device, threshold=DETECTION_THRESHOLD)
        out_file = output_dir / f"result_{img_path.name}"
        cv2.imwrite(str(out_file), res)
        print(f"-> {img_path.name:<30} | {'DEFECT' if defect else 'NORMAL'} | Score: {score:.4f}")

    print(f"\n[+] Inspection finished! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()