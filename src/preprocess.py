# src/preprocess.py
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms


class DINOMultiCropTransform:
    """
    Generates multi-crop views for DINO self-distillation:
    - 2 Global views (224x224) covering global railway bogie context
    - 6 Local views (96x96) capturing fine features (bolts, cracks, track joints)
    """
    def __init__(
        self,
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=6
    ):
        self.local_crops_number = local_crops_number

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Global View 1 (with Gaussian Blur)
        self.global_transform_1 = transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            normalize,
        ])

        # Global View 2 (without heavy blur)
        self.global_transform_2 = transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            normalize,
        ])

        # Local Views (96x96 crops)
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                96, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            normalize,
        ])

    def __call__(self, image):
        # Ensure input is always a PIL RGB Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image or np.ndarray, got {type(image)}")

        crops = [self.global_transform_1(image), self.global_transform_2(image)]
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops


def get_inference_transform(target_size: int = 224) -> transforms.Compose:
    """
    Standard deterministic evaluation transform used during testing/inference.
    """
    return transforms.Compose([
        transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])