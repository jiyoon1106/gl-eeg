"""어댑터 기반 클래스.

새 데이터셋을 추가할 때 **이 파일만 상속해서 파일 하나 쓰면** 끝이다.
전처리 · 모델 · 평가 코드는 손대지 않는다.

    gleeg/adapters/
      ├── base.py          <- 이 파일
      ├── ds004504.py      <- 구현 예시
      └── ds004584.py      <- 새 데이터셋은 여기에 추가
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SubjectRecord:
    """피험자 한 명의 메타데이터 + 파일 위치.

    신호는 담지 않는다 (10분 x 19채널은 무겁다). load_raw() 로 그때 읽는다.
    """

    subject_id: str                          # "ds004504/sub-001" 형태
    dataset_id: str
    path: Path                               # 신호 파일
    disease: str                             # schema.DISEASE_CODES 중 하나
    severity_score: Optional[float] = None   # 원점수 — 구간화하지 않는다
    severity_scale: Optional[str] = None     # "MMSE" | "UPDRS" | ...
    age: Optional[int] = None
    sex: Optional[str] = None
    extra: dict = field(default_factory=dict)


class BaseAdapter(ABC):
    """원본 데이터셋 -> 표준 형태로 바꾸는 창구."""

    dataset_id: str = "unknown"

    @abstractmethod
    def list_subjects(self) -> list[SubjectRecord]:
        """피험자 목록. 신호는 읽지 않는다."""

    @abstractmethod
    def load_raw(self, rec: SubjectRecord):
        """해당 피험자의 신호를 MNE Raw 로 읽는다."""

    @abstractmethod
    def channel_names(self) -> list[str]:
        """이 데이터셋이 가진 채널 목록.

        montage.common_channels() 가 여러 어댑터의 이 값을 받아 교집합을
        계산한다. 채널 수를 코드에 박지 않는 이유가 이것이다.
        """

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """로그용 한 줄 요약."""
        subs = self.list_subjects()
        by_disease: dict[str, int] = {}
        for s in subs:
            by_disease[s.disease] = by_disease.get(s.disease, 0) + 1
        parts = " / ".join(f"{k} {v}" for k, v in sorted(by_disease.items()))
        return (
            f"{self.dataset_id}: {len(subs)}명 ({parts}), "
            f"{len(self.channel_names())}채널"
        )
