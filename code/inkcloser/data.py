"""Split manifest and datasets (CODE BLOCKS 1 and 4 of the notebook)."""
import json
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

DEFAULT_SPLIT_ROOT = "/content/split"


def load_split(split_root=DEFAULT_SPLIT_ROOT):
    """Restore paths, file lists and IMAGE_SIZE from split.json."""
    with open(os.path.join(split_root, "split.json")) as fh:
        manifest = json.load(fh)
    pre_root = os.path.join(split_root, "precorrupted")
    return {
        "IMAGE_SIZE": manifest["image_size"],
        "train_files": manifest["train"],
        "test_files": manifest["test"],
        "sample_files": manifest["sample"],
        "test_clean_dir": os.path.join(split_root, "clean_test"),
        "test_corr_dir": os.path.join(split_root, "corrupted_test"),
        "pre_root": pre_root,
        "pre_clean": os.path.join(pre_root, "clean"),
        "pre_corr": os.path.join(pre_root, "corrupted"),
        # clean_training/ held symlinks and was excluded from the zip. The same
        # images, already resized to IMAGE_SIZE, are in precorrupted/clean.
        "train_dir": os.path.join(pre_root, "clean"),
    }


class PrecorruptedPairDataset(Dataset):
    """Training set: pairs read from disk (random variant per load) + paired augmentation.

    Each epoch shows every training image once with a random stored variant, then
    applies the same paired augmentation to both images.
    """

    def __init__(self, clean_dir, corrupted_dir, files, image_size=896, augment=True):
        self.clean_dir = clean_dir
        self.corrupted_dir = corrupted_dir
        self.image_size = image_size
        self.augment = augment

        variants = {}
        for f in os.listdir(corrupted_dir):
            if f.endswith(".png") and not f.endswith(".tmp.png") and "_v" in f:
                variants.setdefault(f.rsplit("_v", 1)[0], []).append(f)

        stems = sorted(os.path.splitext(f)[0] for f in files)
        self.items = [(s, sorted(variants[s])) for s in stems
                      if s in variants
                      and os.path.exists(os.path.join(clean_dir, s + ".png"))]
        self.files = [s + ".png" for s, _ in self.items]

        n_var = [len(v) for _, v in self.items]
        print(f"[precorrupted] {len(self.items)} images, "
              f"{sum(n_var)} pairs ({min(n_var, default=0)}-{max(n_var, default=0)} variants/image)")
        if len(self.items) < len(stems):
            print(f"  WARNING: {len(stems) - len(self.items)} training images have no "
                  f"pre-corrupted pairs (generation incomplete?) and are skipped")

        self.to_tensor = transforms.ToTensor()
        # fill=1.0 = white on [0, 1] tensors
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(
                degrees=10,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=1.0),
            transforms.RandomAffine(
                degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05),
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=1.0),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        stem, vs = self.items[idx]
        v = vs[int(torch.randint(len(vs), (1,)))]     # torch RNG is seeded per worker
        clean = Image.open(os.path.join(self.clean_dir, stem + ".png")).convert("L")
        corr = Image.open(os.path.join(self.corrupted_dir, v)).convert("L")
        assert clean.size == corr.size == (self.image_size, self.image_size), \
            f"{v}: expected {self.image_size}², got {clean.size} / {corr.size}"

        corrupted, clean = self.to_tensor(corr), self.to_tensor(clean)
        if self.augment:
            # one call on a 2-channel stack -> identical geometry for both images
            pair = self.augmentation(torch.cat([corrupted, clean], dim=0))
            corrupted, clean = pair[0:1], pair[1:2]
        return corrupted, clean


class FrozenPairDataset(Dataset):
    """Test set: pairs read straight from disk. No augmentation."""

    def __init__(self, clean_dir, corrupted_dir, files=None, image_size=896):
        self.clean_dir = clean_dir
        self.corrupted_dir = corrupted_dir
        self.files = sorted(files) if files is not None else sorted(
            f for f in os.listdir(clean_dir) if f.lower().endswith(".png"))
        missing = [f for f in self.files
                   if not os.path.exists(os.path.join(corrupted_dir, f))]
        assert not missing, f"Missing corrupted counterparts: {missing[:5]}"
        self.to_tensor = transforms.ToTensor()
        self.image_size = image_size
        print(f"[frozen] {len(self.files)} pairs in {clean_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        clean = Image.open(os.path.join(self.clean_dir, f)).convert("L")
        corr = Image.open(os.path.join(self.corrupted_dir, f)).convert("L")
        assert clean.size == corr.size == (self.image_size, self.image_size), \
            f"{f}: expected {self.image_size}², got {clean.size} / {corr.size}"
        return self.to_tensor(corr), self.to_tensor(clean), f


def build_datasets(split, max_test=None):
    """Train / test / preview datasets exactly as in CODE BLOCK 4.

    max_test: evaluate only the first N test files (sorted), for quick check runs.
    """
    size = split["IMAGE_SIZE"]
    train = PrecorruptedPairDataset(split["pre_clean"], split["pre_corr"],
                                    split["train_files"], image_size=size)
    test_files = None
    if max_test:
        test_files = sorted(f for f in os.listdir(split["test_clean_dir"])
                            if f.lower().endswith(".png"))[:max_test]
    test = FrozenPairDataset(split["test_clean_dir"], split["test_corr_dir"],
                             files=test_files, image_size=size)
    sample = FrozenPairDataset(split["test_clean_dir"], split["test_corr_dir"],
                               files=split["sample_files"], image_size=size)
    return train, test, sample


def build_loaders(train_dataset, test_dataset, batch_size=4, num_workers=None, generator=None):
    if num_workers is None:
        num_workers = min(4, os.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0, drop_last=False,
                              generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False,
                             num_workers=min(2, num_workers), pin_memory=True)
    return train_loader, test_loader


def preview_tensors(sample_dataset):
    """The 4 fixed preview images, loaded once as tensors."""
    ex_corrupted = torch.stack([sample_dataset[i][0] for i in range(len(sample_dataset))])
    ex_clean = torch.stack([sample_dataset[i][1] for i in range(len(sample_dataset))])
    return ex_corrupted, ex_clean, list(sample_dataset.files)
