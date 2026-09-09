# src/img_loader.py
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

# Valid image extensions across formats
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def get_paths(target_subfolder="train"):
    """
    Automatically detects Local PC vs Kaggle environments and locates 
    the target directory ('train' or 'test').
    Returns: (data_dir, save_dir)
    """
    # 1. Kaggle Environment Check
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        save_dir = Path("/kaggle/working")
        
        # Look for target_subfolder recursively
        target_dirs = [p for p in kaggle_input.rglob(target_subfolder) if p.is_dir()]
        for p in target_dirs:
            if any(f.suffix.lower() in VALID_EXTENSIONS for f in p.rglob("*")):
                print(f"[Auto-Detect: Kaggle] Located '{target_subfolder}' folder: {p}")
                return str(p), str(save_dir)
        
        # Fallback to first available directory inside /kaggle/input
        all_dirs = [p for p in kaggle_input.iterdir() if p.is_dir()]
        fallback = all_dirs[0] if all_dirs else kaggle_input
        print(f"[Auto-Detect: Kaggle] Subfolder '{target_subfolder}' not matched. Using: {fallback}")
        return str(fallback), str(save_dir)

    # 2. Local PC Environment Check
    save_dir = Path.cwd()
    search_roots = [
        Path.cwd(),
        Path("dataset"),
        Path("datasets"),
        Path("Bogie dataset"),
        Path("../Bogie dataset"),
        Path("../dataset")
    ]

    for root in search_roots:
        if root.exists():
            for p in root.rglob(target_subfolder):
                if p.is_dir():
                    has_imgs = any(f.suffix.lower() in VALID_EXTENSIONS for f in p.rglob("*"))
                    if has_imgs:
                        print(f"[Auto-Detect: Local PC] Located '{target_subfolder}' folder: {p.resolve()}")
                        return str(p.resolve()), str(save_dir.resolve())

    # Fallback to any directory that contains images directly
    for root in search_roots:
        if root.exists() and root.is_dir():
            has_imgs = any(f.suffix.lower() in VALID_EXTENSIONS for f in root.rglob("*"))
            if has_imgs:
                print(f"[Auto-Detect: Local PC] Using available image directory: {root.resolve()}")
                return str(root.resolve()), str(save_dir.resolve())

    raise FileNotFoundError(
        f"Could not locate folder '{target_subfolder}' containing images in local workspace or Kaggle input."
    )


class RailwayDataset(Dataset):
    """
    Robust Dataset scanner that loads images safely with PIL.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.image_paths = [
            p for p in self.root_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]

        if len(self.image_paths) == 0:
            print(f"[Warning] Zero images detected in: {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            return self.transform(image)
        return image


def get_data_loader(data_dir, transform, batch_size=4, shuffle=True, num_workers=None):
    """
    Builds an optimized DataLoader that prevents Windows multiprocessing deadlocks.
    """
    dataset = RailwayDataset(root_dir=data_dir, transform=transform)

    # Windows safety: default to 0 workers on Windows to prevent runtime deadlocks
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 2

    drop_last = len(dataset) >= batch_size if shuffle else False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last
    )
    return loader