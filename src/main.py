import os
import math
from pathlib import Path

import torch
from torch.optim import AdamW

from src.models.vit.vit import VisionTransformer
from src.dino.head import DINOHead, FullDINOModel
from src.dino.loss import DINOLoss
from src.preprocess import DINOMultiCropTransform
from src.img_loader import get_paths, get_data_loader


@torch.no_grad()
def update_teacher_ema(student_model, teacher_model, momentum: float):
    for param_s, param_t in zip(student_model.parameters(), teacher_model.parameters()):
        param_t.data.mul_(momentum).add_((1.0 - momentum) * param_s.data)


def save_checkpoint(epoch, student, teacher, optimizer, loss, filepath):
    checkpoint = {
        "epoch": epoch,
        "student_state_dict": student.state_dict(),
        "teacher_state_dict": teacher.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, filepath)
    print(f"[*] Checkpoint saved at epoch {epoch:02d} -> {filepath}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")

    train_data_dir, _ = get_paths(target_subfolder="train")
    
    # Save artifacts into dedicated parameters folder
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameters")
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_file = os.path.join(save_dir, "checkpoint.pth")
    final_backbone_file = os.path.join(save_dir, "new_para.pth")

    BATCH_SIZE = 16
    TOTAL_EPOCHS = 40
    BASE_LR = 2e-4
    MIN_LR = 1e-6
    WARMUP_EPOCHS = 5
    MOMENTUM_TEACHER = 0.996
    OUT_DIM=65536    #65536 for more images

    """
    For big training data
    # In main.py
    BATCH_SIZE = 32           # Increase if GPU VRAM allows (use PyTorch AMP if needed)
    OUT_DIM = 65536           # Scale back to 65k prototypes for rich feature clustering
    TOTAL_EPOCHS = 100        # Increase epochs for deep self-supervised convergence
    WARMUP_EPOCHS = 10        # Extend warmup to match longer training duration
    BASE_LR = 5e-4            # Standard DINO base learning rate
    MIN_LR = 1e-6
    MOMENTUM_TEACHER = 0.996
    """

    transform = DINOMultiCropTransform()
    train_loader = get_data_loader(
        data_dir=train_data_dir,
        transform=transform,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    total_images = len(train_loader.dataset)
    print(f"[*] Total training images loaded: {total_images}")
    if total_images == 0:
        print("[!] No training images found. Terminating.")
        return

    # Student network
    student_vit = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12)
    student_head = DINOHead(in_dim=768, out_dim=OUT_DIM)
    student = FullDINOModel(student_vit, student_head).to(device)

    # Teacher network (Instantiate cleanly without deepcopy)
    teacher_vit = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12)
    teacher_head = DINOHead(in_dim=768, out_dim=OUT_DIM)
    teacher = FullDINOModel(teacher_vit, teacher_head).to(device)
    teacher.load_state_dict(student.state_dict())

    for p in teacher.parameters():
        p.requires_grad = False

    # Loss with dynamic temperature schedule to prevent uniform collapse
    dino_loss = DINOLoss(out_dim=OUT_DIM, ncrops=8, nepochs=TOTAL_EPOCHS).to(device)
    optimizer = AdamW(student.parameters(), lr=BASE_LR, weight_decay=0.04)

    start_epoch = 1
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device)
        student.load_state_dict(checkpoint["student_state_dict"])
        teacher.load_state_dict(checkpoint["teacher_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"[+] Resumed successfully from epoch {start_epoch}")

    print("\n--- Starting Local DINO Self-Supervised Training ---")
    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        student.train()
        total_loss = 0.0

        if epoch <= WARMUP_EPOCHS:
            lr = BASE_LR * (epoch / WARMUP_EPOCHS)
        else:
            cosine_progress = (epoch - WARMUP_EPOCHS) / max(1, (TOTAL_EPOCHS - WARMUP_EPOCHS))
            lr = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1.0 + math.cos(math.pi * cosine_progress))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        for step, crops in enumerate(train_loader):
            crops = [crop.to(device, non_blocking=True) for crop in crops]

            with torch.no_grad():
                teacher_global_1 = teacher(crops[0])
                teacher_global_2 = teacher(crops[1])
                teacher_output = torch.cat([teacher_global_1, teacher_global_2], dim=0)

            student_outputs = [student(crop) for crop in crops]
            student_output = torch.cat(student_outputs, dim=0)

            # Pass the epoch index to compute the scheduled teacher temperature
            loss = dino_loss(student_output, teacher_output, epoch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0)
            optimizer.step()

            update_teacher_ema(student, teacher, MOMENTUM_TEACHER)
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch [{epoch:02d}/{TOTAL_EPOCHS:02d}] | Loss: {avg_loss:.4f} | LR: {lr:.6f}")

        if epoch % 5 == 0 or epoch == TOTAL_EPOCHS:
            save_checkpoint(epoch, student, teacher, optimizer, avg_loss, checkpoint_file)

    torch.save(student.backbone.state_dict(), final_backbone_file)
    print(f"\n[+] Training complete! Saved backbone to: {final_backbone_file}")


if __name__ == "__main__":
    main()