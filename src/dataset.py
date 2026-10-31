import os
import re
import json
import numpy as np

try:
    from sklearn.datasets import fetch_olivetti_faces
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def extract_numeric_seed(roll_no: str) -> int:
    """Extracts all numeric characters of complete roll number, in order, as integer seed."""
    digits = "".join(re.findall(r"\d", roll_no))
    if not digits:
        raise ValueError(f"No digits found in roll number: {roll_no}")
    return int(digits)


def load_olivetti_dataset(cache_path: str = "data/olivetti_faces.npz"):
    """Loads Olivetti Faces (40 identities, 10 images/identity, 64x64 float32 in [0, 1])."""
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        return d["data"].astype(np.float32), d["images"].astype(np.float32), d["target"].astype(np.int32)
    if HAS_SKLEARN:
        bunch = fetch_olivetti_faces()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, data=bunch.data, images=bunch.images, target=bunch.target)
        return bunch.data.astype(np.float32), bunch.images.astype(np.float32), bunch.target.astype(np.int32)
    raise RuntimeError("Neither cached olivetti_faces.npz nor scikit-learn is available.")


def create_or_load_split(roll_no: str = "P26CS0001", save_dir: str = "data"):
    """
    Within each of the 40 identities, randomly splits the 10 images into
    6 train, 2 validation, and 2 test images using numpy.random.default_rng(seed).
    """
    seed = extract_numeric_seed(roll_no)
    npz_path = os.path.join(save_dir, f"split_{roll_no}.npz")
    json_path = os.path.join(save_dir, f"split_{roll_no}.json")

    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cid in range(40):
        idx_c = np.arange(cid * 10, (cid + 1) * 10)
        perm = rng.permutation(idx_c)
        train_idx.extend(perm[:6])
        val_idx.extend(perm[6:8])
        test_idx.extend(perm[8:10])

    train_idx = np.array(train_idx, dtype=np.int32)
    val_idx = np.array(val_idx, dtype=np.int32)
    test_idx = np.array(test_idx, dtype=np.int32)

    os.makedirs(save_dir, exist_ok=True)
    np.savez(npz_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, seed=np.array(seed))
    with open(json_path, "w") as f:
        json.dump({
            "roll_number": roll_no,
            "seed": seed,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "test_idx": test_idx.tolist(),
        }, f, indent=2)
    return train_idx, val_idx, test_idx, seed


def create_masked_test_set(images_test: np.ndarray) -> np.ndarray:
    """
    Creates the occluded test set by setting the centred 16x16 block,
    rows and columns 24 to 39 (inclusive, slice 24:40), to zero.
    """
    masked = images_test.copy()
    masked[:, 24:40, 24:40] = 0.0
    return masked.reshape(len(images_test), -1)
