"""Dataset organization, splitting, and preprocessing."""

from __future__ import annotations

import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMG_SIZE = (224, 224)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


def organize_dataset(source_dir: Path, target_dir: Path) -> None:
    target_folders = ["Amphiprion_ocellaris_images"]
    non_target_folders = [
        "Amphiprion_clarkii_images",
        "Neoglyphidodon_oxyodon_images",
        "Neopetrolisthes_maculatus_images",
        "Heteractis_aurora_images",
    ]

    for label, folders in [("target", target_folders), ("non_target", non_target_folders)]:
        dest = target_dir / label
        dest.mkdir(parents=True, exist_ok=True)
        for folder in folders:
            for img_file in (source_dir / folder).glob("*.jpg"):
                shutil.move(str(img_file), dest / f"{folder}_{img_file.name}")
            for img_file in (source_dir / folder).glob("*.png"):
                shutil.move(str(img_file), dest / f"{folder}_{img_file.name}")

    for label in ["target", "non_target"]:
        count = len(list((target_dir / label).iterdir()))
        print(f"{label}: {count} images")


def split_dataset(source_dir: Path, output_dir: Path) -> None:
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9
    random.seed(RANDOM_SEED)

    for cls_path in source_dir.iterdir():
        if not cls_path.is_dir():
            continue
        files = [f for f in cls_path.iterdir() if f.suffix in {".jpg", ".png"}]
        random.shuffle(files)

        n = len(files)
        n_train = math.floor(n * TRAIN_RATIO)
        n_val = math.floor(n * VAL_RATIO)

        splits = {
            "train": files[:n_train],
            "val": files[n_train : n_train + n_val],
            "test": files[n_train + n_val :],
        }

        for split_name, split_files in splits.items():
            dest = output_dir / split_name / cls_path.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy(f, dest / f.name)

        print(f"{cls_path.name}: {n_train} train / {n_val} val / {n - n_train - n_val} test")


def load_images(folder: Path, label: int) -> tuple[list[np.ndarray], list[int]]:
    images, labels = [], []
    for f in tqdm(folder.iterdir(), desc=str(folder)):
        if f.suffix not in {".jpg", ".png"}:
            continue
        try:
            img = cv2.imread(str(f))
            img = cv2.resize(img, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img.astype("float32") / 255.0)
            labels.append(label)
        except Exception:
            continue
    return images, labels


def load_split(split_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    all_images: list[np.ndarray] = []
    all_labels: list[int] = []

    for label_idx, cls_path in enumerate(sorted(split_dir.iterdir())):
        if cls_path.is_dir():
            imgs, lbls = load_images(cls_path, label_idx)
            all_images.extend(imgs)
            all_labels.extend(lbls)

    return np.array(all_images), np.array(all_labels)
