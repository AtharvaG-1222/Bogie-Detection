# ==============================================================================
# TEST ANOMALY SCRIPT (CLEAN HEATMAP WITH NOISE SUPPRESSION & ABSOLUTE THRESHOLD)
# ==============================================================================
import os
import math
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

# Import the existing ViT model from your project structure
from src.models.vit.vit import VisionTransformer


def get_default_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Inference Device: {device}")
    return device


def load_model(weights_path: str, device: torch.device):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights file not found at: '{weights_path}'. "
            f"Make sure training has completed and saved weights in the 'parameters' folder."
        )

    # Initialize ViT Backbone (ViT-Base: patch 16, embed 768, depth 12, heads 12)
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12
    ).to(device)

    # Load weights
    state_dict = torch.load(weights_path, map_location=device)
    # Handle if state dict was saved from full student or student.backbone
    if "backbone" in state_dict:
        state_dict = state_dict["backbone"]

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"[+] Loaded trained ViT backbone successfully from: {weights_path}")
    return model


def build_normal_memory_bank(model, normal_images_dir: str, device: torch.device, num_samples: int = 50):
    """
    Extracts baseline patch feature distributions from known normal bogie images.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [
        p for p in Path(normal_images_dir).rglob("*")
        if p.is_file() and p.suffix.lower() in valid_exts
    ][:num_samples]

    if not image_paths:
        print("[!] Warning: No images found to build normal bank. Using patch self-deviation mode.")
        return None

    print(f"[*] Building normal feature baseline from {len(image_paths)} normal reference images...")
    features_list = []

    with torch.no_grad():
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            _, patch_tokens = model(tensor, return_patch_tokens=True)  # [1, N, C]
            patch_tokens = F.normalize(patch_tokens, dim=-1, p=2)
            features_list.append(patch_tokens.squeeze(0).cpu())

    # [K * N, C] -> Mean vector across normal features
    all_features = torch.cat(features_list, dim=0)
    mean_normal_feature = torch.mean(all_features, dim=0, keepdim=True).to(device)
    mean_normal_feature = F.normalize(mean_normal_feature, dim=-1, p=2)
    return mean_normal_feature


def run_inference_on_image(
    model,
    image_path: str,
    output_dir: str,
    device: torch.device,
    mean_normal_feature=None,
    noise_floor: float = 0.08,        # Anomaly scores below this stay uncolored (original image)
    anomaly_threshold: float = 0.25   # Anomaly scores reaching this will be bright yellow/red
):
    orig_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(orig_img).unsqueeze(0).to(device)

    with torch.no_grad():
        _, patch_tokens = model(input_tensor, return_patch_tokens=True)  # [1, 196, 768]
        patch_tokens = F.normalize(patch_tokens, dim=-1, p=2)

    # Number of patches along height/width (14x14 = 196 for 224x224 with patch size 16)
    grid_size = int(math.sqrt(patch_tokens.shape[1]))

    # Compute patch-level distance
    if mean_normal_feature is not None:
        # Distance to baseline normal representation
        cosine_sim = torch.sum(patch_tokens * mean_normal_feature, dim=-1)  # [1, 196]
        raw_dist = 1.0 - cosine_sim
    else:
        # Self-deviation mode against image average patch
        center = torch.mean(patch_tokens, dim=1, keepdim=True)
        center = F.normalize(center, dim=-1, p=2)
        raw_dist = 1.0 - torch.sum(patch_tokens * center, dim=-1)

    raw_dist = raw_dist.squeeze(0).cpu().numpy()
    dist_grid = raw_dist.reshape((grid_size, grid_size))

    # Resize anomaly grid to the original input resolution
    dist_map = cv2.resize(dist_grid, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # Aggregate global image defect score (top 5% highest patch distances)
    top_k_elements = max(1, int(raw_dist.size * 0.05))
    defect_score = float(np.mean(np.sort(raw_dist)[-top_k_elements:]))

    # ==============================================================================
    # CLEAN HEATMAP RENDERING (NO WHOLE-IMAGE DISCOLORATION)
    # ==============================================================================
    # 1. Zero out baseline noise so normal metal/tracks do not get colored
    clean_dist_map = np.where(dist_map < noise_floor, 0.0, dist_map - noise_floor)

    # 2. Scale against absolute threshold rather than min-max
    norm_map = np.clip(clean_dist_map / (anomaly_threshold - noise_floor), 0.0, 1.0)

    # 3. Create Heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * norm_map), cv2.COLORMAP_JET)

    # 4. Prepare original OpenCV image (BGR)
    cv_img = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)

    # 5. Mask blending: only apply color where norm_map > 0
    alpha_mask = np.clip(norm_map * 0.7, 0.0, 0.7)[:, :, np.newaxis]
    blended = (cv_img * (1.0 - alpha_mask) + heatmap * alpha_mask).astype(np.uint8)

    # 6. Status Text overlay
    is_anomaly = defect_score > anomaly_threshold
    status_text = f"{'DEFECT DETECTED' if is_anomaly else 'NORMAL'} | Score: {defect_score:.4f}"
    text_color = (0, 0, 255) if is_anomaly else (0, 255, 0)  # Red if defect, Green if normal

    cv2.putText(
        blended,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        text_color,
        2,
        cv2.LINE_AA
    )

    # Save visualization output
    os.makedirs(output_dir, exist_ok=True)
    out_filename = os.path.join(output_dir, f"result_{Path(image_path).name}")
    cv2.imwrite(out_filename, blended)
    print(f"[{'!' if is_anomaly else '*'}] Processed: {Path(image_path).name} | Score: {defect_score:.4f} -> Saved: {out_filename}")


def main():
    device = get_default_device()

    # Paths mapped to parameters folder
    weights_path = os.path.join("parameters", "vit_bogie_dino.pth")
    test_data_dir = os.path.join("dataset", "test")
    train_reference_dir = os.path.join("dataset", "train")
    output_dir = os.path.join("anomaly_results")

    # If test folder doesn't exist, check single test image
    if not os.path.exists(test_data_dir):
        test_data_dir = train_reference_dir

    # 1. Load Trained Model
    model = load_model(weights_path, device)

    # 2. Extract Normal Memory Distribution Baseline
    mean_normal_feature = build_normal_memory_bank(model, train_reference_dir, device, num_samples=30)

    # 3. Locate test images
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    test_images = [
        str(p) for p in Path(test_data_dir).rglob("*")
        if p.is_file() and p.suffix.lower() in valid_exts
    ]

    print(f"[*] Found {len(test_images)} images for anomaly evaluation.")
    if not test_images:
        print("[!] No images found to test. Place an image in 'dataset/test/' and rerun.")
        return

    # 4. Run inference
    for img_path in test_images:
        run_inference_on_image(
            model=model,
            image_path=img_path,
            output_dir=output_dir,
            device=device,
            mean_normal_feature=mean_normal_feature,
            noise_floor=0.08,
            anomaly_threshold=0.25
        )

    print(f"\n[+] Anomaly detection finished. Inspection images saved to '{output_dir}'.")


if __name__ == "__main__":
    main()