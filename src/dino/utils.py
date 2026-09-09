# src/dino/utils.py
import torch

@torch.no_grad()
def update_teacher_ema(student_model: torch.nn.Module, teacher_model: torch.nn.Module, momentum: float = 0.996):
    """
    Updates teacher parameters using an exponential moving average (EMA) of student parameters:
    theta_t = momentum * theta_t + (1 - momentum) * theta_s
    """
    for param_s, param_t in zip(student_model.parameters(), teacher_model.parameters()):
        param_t.data.mul_(momentum).add_(param_s.data, alpha=(1.0 - momentum))