# main.py
import os
import math
import torch
import torch.nn as nn
from torch.optim import AdamW

# Custom project modules
from src.models.vit.vit import VisionTransformer
from src.dino.head import DINOHead
from src.dino.loss import DINOLoss
from src.preprocess import DINOMultiCropTransform
from src.img_loader import get_paths, get_data_loader


class FullDINOModel(nn.Module):
    """
    Wraps the Vision Transformer backbone and the DINO projection head together.
    Handles resolution-safe multi-crop forward passes.
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # Backbone returns (cls_out, patch_out)
        cls_token, _ = self.backbone(x)
        # Head projects CLS token into distribution space: [B, 65536]
        return self.head(cls_token)


@torch.no_grad()
def update_teacher_ema(student_model, teacher_model, momentum):
    """
    Exponential Moving Average update for Teacher weights:
    theta_t = m * theta_t + (1 - m) * theta_s
    """
    for param_s, param_t in zip(student_model.parameters(), teacher_model.parameters()):
        param_t.data.mul_(momentum).add_((1.0 - momentum) * param_s.data)


def save_checkpoint(epoch, student, teacher, optimizer, loss, filepath):
    """
    Saves complete training state dictionary for resuming later.
    """
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
    # -------------------------------------------------------------
    # 1. Device and Environment Path Detection
    # -------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")

    train_data_dir, _ = get_paths(target_subfolder="train")
    
    # Save directly into 'parameters' folder in project root
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameters")
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_file = os.path.join(save_dir, "vit_bogie_checkpoint.pth")
    final_backbone_file = os.path.join(save_dir, "vit_bogie_dino.pth")

    # -------------------------------------------------------------
    # 2. Hyperparameters
    # -------------------------------------------------------------
    BATCH_SIZE = 4            # Reduce to 2 if running on low VRAM (e.g. 4GB)
    TOTAL_EPOCHS = 40
    BASE_LR = 5e-4
    MIN_LR = 1e-6
    WARMUP_EPOCHS = 5
    MOMENTUM_TEACHER = 0.996  # EMA momentum factor

    # -------------------------------------------------------------
    # 3. Data Pipeline Setup
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # 4. Student and Teacher Initialization
    # -------------------------------------------------------------
    student_vit = VisionTransformer(
        img_size=224, 
        patch_size=16, 
        embed_dim=768, 
        depth=12, 
        num_heads=12
    )
    student_head = DINOHead(in_dim=768, out_dim=65536)
    student = FullDINOModel(student_vit, student_head).to(device)

    # Clean manual instantiation without copy.deepcopy
    teacher_vit = VisionTransformer(
        img_size=224, 
        patch_size=16, 
        embed_dim=768, 
        depth=12, 
        num_heads=12
    )
    teacher_head = DINOHead(in_dim=768, out_dim=65536)
    teacher = FullDINOModel(teacher_vit, teacher_head).to(device)
    teacher.load_state_dict(student.state_dict())

    for p in teacher.parameters():
        p.requires_grad = False

    # -------------------------------------------------------------
    # 5. Loss Function and Optimizer
    # -------------------------------------------------------------
    dino_loss = DINOLoss(
        out_dim=65536,
        ncrops=8,
        teacher_temp=0.04,
        student_temp=0.1,
        center_momentum=0.9
    ).to(device)

    optimizer = AdamW(student.parameters(), lr=BASE_LR, weight_decay=0.04)

    # -------------------------------------------------------------
    # 6. Checkpoint Auto-Resume Logic
    # -------------------------------------------------------------
    start_epoch = 1
    if os.path.exists(checkpoint_file):
        print(f"[*] Found existing checkpoint at {checkpoint_file}. Resuming training...")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        student.load_state_dict(checkpoint["student_state_dict"])
        teacher.load_state_dict(checkpoint["teacher_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"[+] Resumed successfully. Starting from epoch {start_epoch}")
    else:
        print("[*] No checkpoint found. Starting training from epoch 1.")

    # -------------------------------------------------------------
    # 7. Training Loop
    # -------------------------------------------------------------
    print("\n--- Starting DINO Self-Supervised Training ---")

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        student.train()
        total_loss = 0.0

        # Cosine Learning Rate Schedule with Warmup
        if epoch <= WARMUP_EPOCHS:
            lr = BASE_LR * (epoch / WARMUP_EPOCHS)
        else:
            cosine_progress = (epoch - WARMUP_EPOCHS) / max(1, (TOTAL_EPOCHS - WARMUP_EPOCHS))
            lr = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1.0 + math.cos(math.pi * cosine_progress))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        for step, crops in enumerate(train_loader):
            crops = [crop.to(device, non_blocking=True) for crop in crops]

            # 1. Teacher forward pass (Global views only: crops[0], crops[1])
            with torch.no_grad():
                teacher_global_1 = teacher(crops[0])
                teacher_global_2 = teacher(crops[1])
                teacher_output = torch.cat([teacher_global_1, teacher_global_2], dim=0)

            # 2. Student forward pass (Pass global and local views cleanly)
            student_outputs = []
            for crop in crops:
                student_outputs.append(student(crop))
            student_output = torch.cat(student_outputs, dim=0)

            # 3. Multi-crop cross-entropy loss
            loss = dino_loss(student_output, teacher_output)

            # 4. Backward pass & optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0)
            optimizer.step()

            # 5. Teacher EMA update
            update_teacher_ema(student, teacher, MOMENTUM_TEACHER)

            total_loss += loss.item()

        num_batches = max(1, len(train_loader))
        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch:02d}/{TOTAL_EPOCHS:02d}] | Loss: {avg_loss:.4f} | LR: {lr:.6f}")

        # Checkpoint every epoch
        save_checkpoint(epoch, student, teacher, optimizer, avg_loss, checkpoint_file)

    # -------------------------------------------------------------
    # 8. Export Final Clean ViT Backbone for Testing
    # -------------------------------------------------------------
    torch.save(student.backbone.state_dict(), final_backbone_file)
    print(f"\n[+] Training complete!")
    print(f"    - Full checkpoint: {checkpoint_file}")
    print(f"    - Exported ViT backbone: {final_backbone_file}")


if __name__ == "__main__":
    main()  