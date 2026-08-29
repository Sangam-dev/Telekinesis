import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from ml.model import GESTURE_CLASSES

CSV_PATH = "data/raw/gesture.csv"


class GestureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_splits(csv_path=CSV_PATH, test_size=0.15, val_size=0.15, random_state=42):
    df = pd.read_csv(csv_path)
    label_to_idx = {name: i for i, name in enumerate(GESTURE_CLASSES)}
    df = df[df["label"].isin(label_to_idx.keys())]  # guard against stray labels

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].map(label_to_idx).values.astype(np.int64)

    # first split off test set, then split remaining into train/val
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size,
        stratify=y_train_val, random_state=random_state
    )

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }
