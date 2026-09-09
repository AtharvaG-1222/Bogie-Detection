# src/dino/loss.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """
    Cross-Entropy Loss between Student and Centered/Sharpened Teacher Distributions
    supporting multi-crop views (2 Global + 6 Local = 8 crops).

    Includes dynamic teacher temperature scheduling (warmup from 0.04 to 0.07)
    to prevent representation collapse (loss frozen at ln(65536) ≈ 11.09).
    """
    def __init__(
        self, 
        out_dim=65536, 
        ncrops=8, 
        warmup_teacher_temp=0.04, 
        teacher_temp=0.07, 
        warmup_teacher_temp_epochs=10, 
        nepochs=40, 
        student_temp=0.1, 
        center_momentum=0.9
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        
        # Center buffer for teacher sharpening
        self.register_buffer("center", torch.zeros(1, out_dim))

        # Build dynamic teacher temperature schedule to prevent uniform distribution trap
        # Warmup phase linearly increases sharpness threshold, then stabilizes
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor, epoch: int) -> torch.Tensor:
        """
        student_output: [(ncrops * B), out_dim] -> (2 global + 6 local = 8 views)
        teacher_output: [(2 * B), out_dim]      -> (2 global views only)
        epoch: Current training epoch index (1-based) to select scheduled teacher temperature
        """
        # 1. Student temperature scaling + chunking
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # 2. Retrieve scheduled teacher temperature for current epoch
        cur_epoch_idx = min(epoch - 1, len(self.teacher_temp_schedule) - 1)
        cur_teacher_temp = self.teacher_temp_schedule[cur_epoch_idx]

        # 3. Teacher centering + dynamic sharpening + chunking
        teacher_out = F.softmax((teacher_output - self.center) / cur_teacher_temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0.0
        n_loss_terms = 0

        # 4. Cross-entropy across pairs: Each teacher view matches all student views (except itself)
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # Do not match the exact same view against itself
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss = total_loss / n_loss_terms

        # 5. Update running center with detached teacher outputs
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        """
        Exponential Moving Average (EMA) update for teacher centering vector.
        Prevents any single dimension/prototype from permanently dominating.
        """
        batch_center = torch.mean(teacher_output.detach(), dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=(1.0 - self.center_momentum))