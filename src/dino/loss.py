# src/dino/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """
    Cross-Entropy Loss between Student and Centered/Sharpened Teacher Distributions
    supporting multi-crop views (2 Global + 6 Local = 8 crops).
    """
    def __init__(
        self, 
        out_dim=65536, 
        ncrops=8, 
        teacher_temp=0.04, 
        student_temp=0.1, 
        center_momentum=0.9
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        
        # Center buffer for teacher sharpening
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor) -> torch.Tensor:
        """
        student_output: [(ncrops * B), out_dim] -> (2 global + 6 local = 8 views)
        teacher_output: [(2 * B), out_dim]      -> (2 global views only)
        """
        # 1. Student temperature scaling + chunking
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # 2. Teacher centering + sharpening + chunking
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0.0
        n_loss_terms = 0

        # Cross-entropy across pairs: Each teacher view matches all student views (except itself)
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # Do not match the exact same view against itself
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss = total_loss / n_loss_terms

        # Update running center with detached teacher outputs
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        """
        Exponential Moving Average (EMA) update for teacher centering vector.
        """
        batch_center = torch.mean(teacher_output.detach(), dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=(1.0 - self.center_momentum))