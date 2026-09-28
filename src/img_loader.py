# src/img_loader.py
import os
import sys
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

# =====================================================================
# DETERMINISTIC ABSOLUTE PATH RESOLUTION
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
                return str(p.resolve()), str(save_dir.resolve())
        
        # Fallback to first available directory inside /kaggle/input
        all_dirs = [p for p in kaggle_input.iterdir() if p.is_dir()]
        fallback = all_dirs[0] if all_dirs else kaggle_input
        print(f"[Auto-Detect: Kaggle] Subfolder '{target_subfolder}' not matched. Using: {fallback}")
        return str(fallback.resolve()), str(save_dir.resolve())

    # 2. Local PC Environment Check (Always anchored to PROJECT_ROOT)
    save_dir = PROJECT_ROOT
    search_roots = [
        PROJECT_ROOT,
        PROJECT_ROOT / "dataset",
        PROJECT_ROOT / "datasets",
        PROJECT_ROOT / "Bogie dataset",
        PROJECT_ROOT.parent / "Bogie dataset",
        PROJECT_ROOT.parent / "dataset"
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
        self.root_dir = Path(root_dir).resolve()
        self.transform = transform

        self.image_paths = [
            p.resolve() for p in self.root_dir.rglob("*")
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