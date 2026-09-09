# ==============================================================================
# SINGLE-FILE DINO VISION TRANSFORMER TRAINING FOR KAGGLE
# ==============================================================================
import os
import sys
import copy
import math
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torchvision.transforms as transforms
from IPython.display import FileLink

# Deterministic seed & GPU check
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Execution device: {device}")
if not torch.cuda.is_available():
    print("[!] Warning: GPU not detected. Make sure Accelerator is set to GPU T4 x2 or P100.")

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
    def __init__(self, dim, num_heads=12, drop=0.0):
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
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.0
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
            align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x: torch.Tensor, return_patch_tokens: bool = True):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.interpolate_pos_encoding(x, w, h)
        x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        cls_out = x[:, 0]
        patch_out = x[:, 1:]

        if return_patch_tokens:
            return cls_out, patch_out
        return cls_out


# ==============================================================================
# 2. DINO PROJECTION HEAD & MODEL WRAPPER (MANUAL L2 NORMALIZATION)
# ==============================================================================
class DINOHead(nn.Module):
    def __init__(self, in_dim=768, out_dim=65536, hidden_dim=2048, bottleneck_dim=256, nlayers=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Standard linear projection without fragile wrapper
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        nn.init.trunc_normal_(self.last_layer.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        # Normalize prototype weights to unit sphere (exact DINO weight normalization)
        norm_weight = F.normalize(self.last_layer.weight, dim=-1, p=2)
        return F.linear(x, norm_weight)


# ==============================================================================
# 3. MULTI-CROP AUGMENTATION & DATASET
# ==============================================================================
class DINOMultiCropTransform:
    def __init__(self, global_crops_scale=(0.4, 1.0), local_crops_scale=(0.05, 0.4), local_crops_number=6):
        self.local_crops_number = local_crops_number

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.global_transform_1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            normalize,
        ])

        self.global_transform_2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            normalize,
        ])

        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            normalize,
        ])

    def __call__(self, image: Image.Image):
        crops = [self.global_transform_1(image), self.global_transform_2(image)]
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops


class RailwayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.image_paths = [
            p for p in self.root_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in valid_exts
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            return self.transform(image)
        return image


# ==============================================================================
# 4. DINO LOSS & EMA UPDATE
# ==============================================================================
class DINOLoss(nn.Module):
    def __init__(self, out_dim=65536, ncrops=8, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor) -> torch.Tensor:
        student_out = (student_output / self.student_temp).chunk(self.ncrops)
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1).detach().chunk(2)

        total_loss = 0.0
        n_loss_terms = 0

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss = total_loss / n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        batch_center = torch.mean(teacher_output.detach(), dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=(1.0 - self.center_momentum))


@torch.no_grad()
def update_teacher_ema(student_model, teacher_model, momentum):
    for param_s, param_t in zip(student_model.parameters(), teacher_model.parameters()):
        param_t.data.mul_(momentum).add_((1.0 - momentum) * param_s.data)


# ==============================================================================
# 5. AUTOMATIC KAGGLE PATH DETECTION
# ==============================================================================
def find_kaggle_train_dir():
    kaggle_input = Path("/kaggle/input")
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Search for an explicit 'train' directory with valid images
    for p in kaggle_input.rglob("train"):
        if p.is_dir() and any(f.suffix.lower() in valid_exts for f in p.rglob("*")):
            return str(p)

    # Fallback: Find any directory containing images
    for p in kaggle_input.rglob("*"):
        if p.is_dir() and any(f.suffix.lower() in valid_exts for f in p.glob("*.*")):
            return str(p)

    raise FileNotFoundError("Could not find bogie training images inside /kaggle/input.")


# ==============================================================================
# 6. MAIN KAGGLE TRAINING LOOP
# ==============================================================================
def train_dino():
    save_dir = "/kaggle/working"
    final_weights_path = os.path.join(save_dir, "vit_bogie_dino.pth")
    checkpoint_path = os.path.join(save_dir, "vit_bogie_checkpoint.pth")

    train_data_dir = find_kaggle_train_dir()
    print(f"[+] Training dataset mapped at: {train_data_dir}")

    # Hyperparameters for Kaggle GPU
    BATCH_SIZE = 8
    TOTAL_EPOCHS = 40
    BASE_LR = 5e-4
    MIN_LR = 1e-6
    WARMUP_EPOCHS = 5
    MOMENTUM_TEACHER = 0.996

    # Build dataset and dataloader
    transform = DINOMultiCropTransform()
    dataset = RailwayDataset(train_data_dir, transform=transform)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=len(dataset) >= BATCH_SIZE
    )

    print(f"[*] Total training images detected: {len(dataset)}")
    if len(dataset) == 0:
        print("[!] No training images found. Aborting.")
        return

    # Student Network
    student_vit = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12)
    student_head = DINOHead(in_dim=768, out_dim=65536)
    student = FullDINOModel(student_vit, student_head).to(device)

    # Teacher Network (Instantiate cleanly + copy weights without deepcopy)
    teacher_vit = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12)
    teacher_head = DINOHead(in_dim=768, out_dim=65536)
    teacher = FullDINOModel(teacher_vit, teacher_head).to(device)
    teacher.load_state_dict(student.state_dict())

    for p in teacher.parameters():
        p.requires_grad = False

    dino_loss = DINOLoss().to(device)
    optimizer = AdamW(student.parameters(), lr=BASE_LR, weight_decay=0.04)

    # Resume checkpoint if running on Kaggle session reload
    start_epoch = 1
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        student.load_state_dict(ckpt["student_state_dict"])
        teacher.load_state_dict(ckpt["teacher_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[+] Loaded checkpoint. Resuming from epoch {start_epoch}")

    print("\n--- Starting DINO Self-Supervised Training on GPU ---")
    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        student.train()
        total_loss = 0.0

        # Cosine LR Schedule with Warmup
        if epoch <= WARMUP_EPOCHS:
            lr = BASE_LR * (epoch / WARMUP_EPOCHS)
        else:
            progress = (epoch - WARMUP_EPOCHS) / max(1, (TOTAL_EPOCHS - WARMUP_EPOCHS))
            lr = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1.0 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        for step, crops in enumerate(train_loader):
            crops = [c.to(device, non_blocking=True) for c in crops]

            # 1. Teacher evaluates only the two 224x224 global views
            with torch.no_grad():
                t_global_1 = teacher(crops[0])
                t_global_2 = teacher(crops[1])
                teacher_output = torch.cat([t_global_1, t_global_2], dim=0)

            # 2. Student evaluates all 8 views cleanly
            student_outputs = [student(c) for c in crops]
            student_output = torch.cat(student_outputs, dim=0)

            # 3. Loss & Backpropagation
            loss = dino_loss(student_output, teacher_output)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0)
            optimizer.step()

            # 4. Teacher EMA update
            update_teacher_ema(student, teacher, MOMENTUM_TEACHER)
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch [{epoch:02d}/{TOTAL_EPOCHS:02d}] | Loss: {avg_loss:.4f} | LR: {lr:.6f}")

        # Checkpoint every 5 epochs
        if epoch % 5 == 0 or epoch == TOTAL_EPOCHS:
            torch.save({
                "epoch": epoch,
                "student_state_dict": student.state_dict(),
                "teacher_state_dict": teacher.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss
            }, checkpoint_path)

    # 7. Save and export clean ViT backbone weights
    torch.save(student.backbone.state_dict(), final_weights_path)
    print(f"\n[+] Backbone training complete! Weights saved to: {final_weights_path}")
    print("[*] Click the link below to download your weights file to your PC:")
    return FileLink(r"vit_bogie_dino.pth")


# Execute training
train_dino()