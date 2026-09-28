import os
import sys
from pathlib import Path

# =====================================================================
# DETERMINISTIC ABSOLUTE PATH RESOLUTION
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.models.vit.vit import VisionTransformer


def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_trained_backbone(weights_path: str, device: torch.device):
    model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12)
    state_dict = torch.load(weights_path, map_location=device)

    if "student_state_dict" in state_dict:
        state_dict = state_dict["student_state_dict"]

    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith("head."):
            continue
        cleaned = k.replace("backbone.", "")
        clean_state[cleaned] = v

    model.load_state_dict(clean_state, strict=False)
    model.to(device).eval()
    print(f"[*] Backbone loaded successfully from: {weights_path}")
    return model


@torch.no_grad()
def build_memory_bank(model, train_folder: str, device: torch.device, num_images: int = 80, ballast_row_cutoff: int = 12):
    """
    Builds a robust patch memory bank.
    ballast_row_cutoff: Discards bottom patch rows corresponding to trackbed/ballast (0-14 rows).
    """
    model.eval()
    transform = get_inference_transform()
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    image_paths = [
        p.resolve() for p in Path(train_folder).resolve().rglob("*")
        if p.suffix.lower() in valid_exts and not p.name.startswith("result_")
    ][:num_images]

    if not image_paths:
        raise FileNotFoundError(f"No reference images found in: {train_folder}")

    print(f"[*] Extracting normal tokens from {len(image_paths)} bogies (Excluding ballast rows >= {ballast_row_cutoff})...")
    bank_tokens = []

    for path in image_paths:
        orig = cv2.imread(str(path))
        if orig is None:
            continue
        rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)

        _, patches = model(tensor)  # [1, 196, 768]
        patches = F.normalize(patches.squeeze(0), dim=-1, p=2)

        # Reshape to 14x14 grid to selectively keep only bogie components
        grid_patches = patches.view(14, 14, 768)
        bogie_patches = grid_patches[:ballast_row_cutoff, :, :].reshape(-1, 768)

        bank_tokens.append(bogie_patches.cpu())

    # Full Bank: [Total_Bogie_Tokens, 768]
    memory_bank = torch.cat(bank_tokens, dim=0).to(device)
    print(f"[+] Active bogie memory bank: {memory_bank.shape[0]} normal tokens indexed.")
    return memory_bank


@torch.no_grad()
def evaluate_image(
    model, 
    memory_bank, 
    image_path: str, 
    device: torch.device, 
    threshold: float = 0.42, 
    ballast_row_cutoff: int = 12
):
    model.eval()
    resolved_path = str(Path(image_path).resolve())
    orig_bgr = cv2.imread(resolved_path)
    if orig_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {resolved_path}")
    orig_h, orig_w = orig_bgr.shape[:2]

    transform = get_inference_transform()
    rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)

    # 1. Extract and normalize patch tokens
    _, patches = model(tensor)
    patches = F.normalize(patches.squeeze(0), dim=-1, p=2)  # [196, 768]

    # 2. Vectorized nearest-neighbor matching across normal bank
    # Similarity: [196, Bank_Size]
    sim_matrix = torch.mm(patches, memory_bank.T)
    max_sim, _ = torch.max(sim_matrix, dim=-1)
    patch_distances = (1.0 - max_sim).cpu().numpy().reshape(14, 14)

    # 3. Suppress ballast rows from anomaly voting
    if ballast_row_cutoff < 14:
        patch_distances[ballast_row_cutoff:, :] = 0.0

    # 4. Image-level score from highest divergent bogie patches
    bogie_scores = patch_distances[:ballast_row_cutoff, :].flatten()
    k_top = max(1, int(0.03 * len(bogie_scores)))  # Top 3% anomalous bogie patches
    anomaly_score = float(np.mean(np.sort(bogie_scores)[-k_top:]))

    # 5. Smooth heatmap generation
    heatmap_resized = cv2.resize(patch_distances, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    
    # Dynamic scaling based on the baseline threshold
    scaled_map = np.clip((heatmap_resized - 0.20) / (threshold - 0.20 + 1e-6), 0.0, 1.0)
    colored_map = cv2.applyColorMap((scaled_map * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Blend exclusively where anomaly distance is significant
    active_mask = (scaled_map > 0.40).astype(np.float32)[:, :, np.newaxis]
    blended = (orig_bgr * (1.0 - (active_mask * 0.50)) + colored_map * (active_mask * 0.50)).astype(np.uint8)

    # 6. Status text
    is_defect = anomaly_score >= threshold
    status = f"DEFECT | Score: {anomaly_score:.4f}" if is_defect else f"NORMAL | Score: {anomaly_score:.4f}"
    color = (0, 0, 255) if is_defect else (0, 255, 0)

    cv2.putText(blended, status, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    cv2.putText(blended, f"Threshold: {threshold:.4f}", (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return blended, anomaly_score, is_defect


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Inference Device: {device}")

    project_root = Path(__file__).resolve().parent
    weights_path = (project_root / "parameters" / "params.pth").resolve()
    train_dir = (project_root / "dataset" / "train").resolve()
    test_dir = (project_root / "dataset" / "test").resolve()
    output_dir = (project_root / "anomaly_results").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        fallback_ckpt = (project_root / "parameters" / "checkpoint.pth").resolve()
        if fallback_ckpt.exists():
            weights_path = fallback_ckpt
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")

    model = load_trained_backbone(str(weights_path), device)

    # Build bank (rows 0-11 for bogie, discarding bottom rows 12-13 for trackbed)
    BALLAST_CUTOFF = 12
    spatial_bank = build_memory_bank(model, str(train_dir), device, num_images=60, ballast_row_cutoff=BALLAST_CUTOFF)

    # Calibrate threshold against normal train samples
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    train_samples = [
        p.resolve() for p in train_dir.rglob("*")
        if p.suffix.lower() in valid_exts and not p.name.startswith("result_")
    ][:5]

    print("\n[*] Calibrating baseline distance on normal training samples:")
    scores = []
    for p in train_samples:
        _, s, _ = evaluate_image(model, spatial_bank, str(p), device, threshold=0.40, ballast_row_cutoff=BALLAST_CUTOFF)
        scores.append(s)
        print(f"  Sample [{p.name}]: Normal Score = {s:.4f}")

    baseline_mean = float(np.mean(scores)) if scores else 0.25
    DETECTION_THRESHOLD = round(baseline_mean + 0.12, 4)
    print(f"[+] Calibrated Detection Threshold: {DETECTION_THRESHOLD} (Baseline + 0.12)\n")

    # Evaluate test folder
    test_images = [
        p.resolve() for p in test_dir.rglob("*")
        if p.suffix.lower() in valid_exts and not p.name.startswith("result_")
    ]

    print(f"[*] Evaluating {len(test_images)} test images...")
    for img_path in test_images:
        res, score, defect = evaluate_image(
            model, 
            spatial_bank, 
            str(img_path), 
            device, 
            threshold=DETECTION_THRESHOLD, 
            ballast_row_cutoff=BALLAST_CUTOFF
        )
        out_file = (output_dir / f"result_{img_path.name}").resolve()
        cv2.imwrite(str(out_file), res)
        print(f"-> {img_path.name:<30} | {'DEFECT' if defect else 'NORMAL'} | Score: {score:.4f}")

    print(f"\n[+] Finished. Heatmaps saved to: {output_dir}")


if __name__ == "__main__":
    main()