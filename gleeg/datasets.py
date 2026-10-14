"""data.npz -> PyTorch Dataset / DataLoader.

build_dataset.py 가 만든 .npz 를 읽는다. 원본 5.4GB 를 다시 읽지 않으므로
Colab 에서도 가볍게 돌아간다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .preprocess.montage import channel_indices


@dataclass
class Bundle:
    """npz 한 묶음. 학습 루프와 평가가 공유한다."""

    X: np.ndarray                 # (N, C, T) float16
    y_disease: np.ndarray         # (N,) int8
    y_level: np.ndarray           # (N,) int8  (-100 = 없음)
    severity_score: np.ndarray    # (N,) float32
    subject_id: np.ndarray        # (N,) str
    dataset_id: np.ndarray        # (N,) str
    quality: np.ndarray           # (N,) float32
    channels: list[str]
    sfreq: float
    meta: dict

    @property
    def n_epochs(self) -> int:
        return len(self.X)

    @property
    def n_subjects(self) -> int:
        return len(np.unique(self.subject_id))

    def describe(self) -> str:
        classes = self.meta["disease_classes"]
        u, c = np.unique(self.y_disease, return_counts=True)
        dist = " / ".join(f"{classes[i]} {n}" for i, n in zip(u, c))
        return (
            f"{self.n_epochs} epochs · {self.n_subjects} subjects · "
            f"{len(self.channels)}ch @ {self.sfreq:g}Hz · {dist}"
        )

    def select_channels(self, wanted: list[str]) -> "Bundle":
        """채널 서브셋만 남긴 새 Bundle. 저채널 실험이 이것 한 줄로 끝난다."""
        idx = channel_indices(self.channels, wanted)
        new = Bundle(**{**self.__dict__})
        new.X = self.X[:, idx, :]
        new.channels = list(wanted)
        return new


def load_bundle(path: Path | str) -> Bundle:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. 먼저 만드세요:\n"
            "  python -m scripts.build_dataset --exp base --source derivatives"
        )
    z = np.load(path, allow_pickle=False)
    return Bundle(
        X=z["X"],
        y_disease=z["y_disease"],
        y_level=z["y_level"],
        severity_score=z["severity_score"],
        subject_id=z["subject_id"],
        dataset_id=z["dataset_id"],
        quality=z["quality"],
        channels=[str(c) for c in z["channels"]],
        sfreq=float(z["sfreq"]),
        meta=json.loads(str(z["meta"])),
    )


class EEGEpochDataset(Dataset):
    """float16 로 저장된 epoch 을 꺼낼 때 float32 로 올린다."""

    def __init__(self, bundle: Bundle) -> None:
        self.b = bundle

    def __len__(self) -> int:
        return self.b.n_epochs

    def __getitem__(self, i: int) -> dict:
        return {
            "x": torch.from_numpy(self.b.X[i].astype(np.float32)),
            "y_disease": torch.tensor(int(self.b.y_disease[i]), dtype=torch.long),
            "y_level": torch.tensor(int(self.b.y_level[i]), dtype=torch.long),
            "quality": torch.tensor([float(self.b.quality[i])], dtype=torch.float32),
            "idx": i,
        }


def make_loaders(
    bundle: Bundle,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int = 64,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """fold 하나의 train/test 로더.

    인덱스는 evaluate.make_splits() 가 준 것을 그대로 넣는다. 그래야
    피험자 단위 분할이 유지된다.
    """
    ds = EEGEpochDataset(bundle)
    g = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),   # GPU 없으면 경고만 나므로 끈다
    }
    return (
        DataLoader(Subset(ds, train_idx), shuffle=True, generator=g, drop_last=True, **common),
        DataLoader(Subset(ds, test_idx), shuffle=False, **common),
    )
