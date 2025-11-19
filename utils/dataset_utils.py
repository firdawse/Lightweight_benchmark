"""Utility functions for loading and processing datasets."""


from pathlib import Path

import datasets
import numpy as np
from omegaconf import DictConfig
from datasets import load_from_disk

from typing import List, Dict, Tuple, Set


def load_leafy_spurge(
    cfg_dataset: DictConfig,
) -> Tuple[List[str], List[str], np.ndarray, List[str]]:
    """Load the mpg-ranch/leafy_spurge dataset (https://huggingface.co/datasets/mpg-ranch/leafy_spurge).

    Args:
        cfg_dataset: configuration file

    Returns:
        images: list of image (in PIL format) (train + test)
        labels: list of binary labels (train + test)
        idx2label: a dict of index to label
    """
    # We only take the crop set of 39x39 pixel images
    # load the dataset from huggingface
    if Path(cfg_dataset.paths.dataset_path ).exists():
        trains_ds = datasets.load_from_disk(cfg_dataset.paths.dataset_path + "context_train")
        test_ds = datasets.load_from_disk(cfg_dataset.paths.dataset_path + "context_test")
        print('dataset downloaded')
    else:
        print('from hugging face')
        from huggingface_hub import login
        
        trains_ds = datasets.load_dataset(
            "mpg-ranch/leafy_spurge",
            # "crop",
            "context",
            split="train",
        )  # 800
        test_ds = datasets.load_dataset(
            "mpg-ranch/leafy_spurge",
            # "crop",
            "context",
            split="test",
        )  # 100

    idx2label = {0: "not leafy spurge", 1: "leafy spurge"}
    print( trains_ds["image"][0],
)
    return (
        trains_ds["image"] + test_ds["image"],
        trains_ds["label"] + test_ds["label"],
        idx2label,
    )
def load_imagenet(
    cfg_dataset: DictConfig,
) -> Tuple[List[str], np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Load the ImageNet dataset from Hugging Face Arrow format and
    corresponding precomputed embeddings.

    Args:
        cfg_dataset: configuration file (expects cfg_dataset.paths.dataset_path)
    
    Returns:
        img_path: dummy list of image identifiers (or placeholder paths)
        mturks_idx: array of image labels (same as orig_idx, kept for compatibility)
        orig_idx: ground truth class indices (int)
        clsidx_to_labels: a dict of class idx to str.
    """
    dataset_path = Path(cfg_dataset.paths.dataset_path)
    embeddings_path = dataset_path / "embeddings"

    # === 1. Load dataset ===
    print("Loading ImageNet dataset from Hugging Face Arrow format...")
    dataset = load_from_disk(str(dataset_path))
    print(f"Loaded {len(dataset)} samples.")

    # The dataset contains image-label pairs
    labels = np.array(dataset["label"])
    #img_path = [f"sample_{i}" for i in range(len(dataset))]  # no real paths in Arrow

    # For compatibility with existing downstream code expecting
    # (mturks_idx, orig_idx), we’ll reuse labels as both
    orig_idx = labels
    # === 3. Build class-to-label mapping ===
    if "label" in dataset.features and hasattr(dataset.features["label"], "names"):
        clsidx_to_labels = {
            i: name for i, name in enumerate(dataset.features["label"].names)
        }
    else:
        clsidx_to_labels = {int(i): f"class_{i}" for i in np.unique(labels)}

    print("ImageNet total count:", len(labels))
    print("Number of unique classes:", len(clsidx_to_labels))

    return  orig_idx, clsidx_to_labels

def get_train_test_split_index(
    train_test_ration: float, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the index of the training and validation set.

    Args:
        train_test_ration: ratio of training set
        n: number of samples
    Returns:
        index of the training and validation set
    """
    arange = np.arange(n)
    np.random.shuffle(arange)
    train_idx = arange[: int(n * train_test_ration)]
    val_idx = arange[int(n * train_test_ration) :]
    return train_idx, val_idx

def train_test_split(
    data: np.ndarray, train_idx: List[int], val_idx: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Split the data into training and validation set.

    Args:
        data: data
        train_idx: index of the training set
        val_idx: index of the validation set
    Return:
        training and validation set
    """
    if data is not np.ndarray:
        data = np.array(data)
    return data[train_idx], data[val_idx]
