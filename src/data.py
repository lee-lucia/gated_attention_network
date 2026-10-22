from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def tensor_to_rgb8(image):
    """CHW float model tensor -> RGB uint8, rounded consistently when saving."""
    rgb = image.detach().cpu().permute(1, 2, 0).numpy()
    return np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def list_images(folder):
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"Directory does not exist: {folder}")
    return sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMG_EXTS])


class PairedImageDataset(Dataset):
    """
    Expected structure:
      root/
        train/input/   degraded images
        train/target/  reference images
        val/input/
        val/target/

    Input/target pairs are matched by filename stem (case-insensitive).
    """
    def __init__(self, input_dir, target_dir, patch_size=256):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.patch_size = patch_size
        input_paths, target_paths = list_images(self.input_dir), list_images(self.target_dir)
        inputs = {p.stem.lower(): p for p in input_paths}
        targets = {p.stem.lower(): p for p in target_paths}
        if len(inputs) != len(input_paths) or len(targets) != len(target_paths):
            raise ValueError("Duplicate image stems in paired dataset")
        if inputs.keys() != targets.keys():
            raise ValueError("Paired input/target filenames do not match; provide only paired images")
        keys = sorted(inputs)
        if not keys:
            raise RuntimeError(
                f"No paired images found.\nInput: {input_dir}\nTarget: {target_dir}\n"
                "Ensure paired files have matching names."
            )
        self.pairs = [(inputs[k], targets[k]) for k in keys]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, tp = self.pairs[idx]
        with Image.open(ip) as image:
            x = image.convert("RGB")
        with Image.open(tp) as image:
            y = image.convert("RGB")
        if y.size != x.size:
            raise ValueError(f"Paired sizes differ: {ip.name} {x.size} vs {tp.name} {y.size}")

        if self.patch_size and self.patch_size > 0:
            ps = self.patch_size
            x = x.resize((ps, ps), Image.Resampling.BILINEAR)
            y = y.resize((ps, ps), Image.Resampling.BILINEAR)

        return TF.to_tensor(x), TF.to_tensor(y), ip.name


def training_datasets(root, image_size=256, train_ratio=0.8):
    """Pre-split folders, or a documented legacy sorted split for UIEB/LSUI."""
    root = Path(root)
    if (root / "train").is_dir():
        return (PairedImageDataset(root / "train/input", root / "train/target", image_size),
                PairedImageDataset(root / "val/input", root / "val/target", 0))
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    for input_name, target_name in (("raw", "reference"), ("input", "GT")):
        if (root / input_name).is_dir() and (root / target_name).is_dir():
            train = PairedImageDataset(root / input_name, root / target_name, image_size)
            val = PairedImageDataset(root / input_name, root / target_name, 0)
            cut = int(len(train) * train_ratio)
            if cut == 0 or cut == len(train):
                raise ValueError("Split requires at least one training and one validation pair")
            train.pairs, val.pairs = train.pairs[:cut], val.pairs[cut:]
            return train, val
    raise ValueError("Expected train/val input/target, UIEB raw/reference, or LSUI input/GT")


class ImageFolderDataset(Dataset):
    """Unpaired folder dataset for inference on RUIE or other test images."""
    def __init__(self, input_dir):
        self.paths = list_images(input_dir)
        if not self.paths:
            raise RuntimeError(f"No images found in: {input_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return TF.to_tensor(img), p.name
