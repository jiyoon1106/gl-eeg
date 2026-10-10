"""인터페이스 ① — 표준 샘플 정의.

모든 데이터셋은 어댑터를 거쳐 이 형태가 된다. 이 파일을 바꾸면 팀 전체 코드가
영향을 받으므로, 바꿀 때는 반드시 공유할 것.

설계 원칙
---------
1. 채널 개수를 고정하지 않는다.
   `channels` 가 이 샘플이 실제로 가진 채널 목록을 들고 다닌다.
   데이터셋 조합의 공통 채널은 montage.common_channels() 가 실행 시점에 계산한다.

2. 중증도는 "단계"가 아니라 "원점수 + 척도명"으로 저장한다.
   severity_score=22, severity_scale="MMSE"
   단계 변환은 labels.to_level() 이 config 의 구간을 받아 수행한다.
   정수 라벨로 하드코딩하면 구간을 바꿀 때 전처리를 전부 다시 돌려야 한다.

3. subject_id 와 dataset_id 를 끝까지 들고 다닌다.
   - subject_id : 피험자 단위 교차검증의 근거 (없으면 성능이 허위로 뜬다)
   - dataset_id : 데이터셋을 합칠 때 도메인 시프트 검증용
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# 표준 질환 코드. 열린 집합이므로 새 질환은 그냥 추가한다.
#   CN  = 정상 대조군        MCI = 경도인지장애
#   AD  = 알츠하이머          FTD = 전두측두치매
#   PD  = 파킨슨             ADHD = 주의력결핍 과행동장애
DISEASE_CODES = ("CN", "MCI", "AD", "FTD", "PD", "ADHD")

# 지원하는 중증도 척도
SEVERITY_SCALES = ("MMSE", "MoCA", "CDR", "UPDRS", "MADRS")


@dataclass
class EEGSample:
    """EEG 한 조각(epoch 또는 연속 구간) + 그에 딸린 메타데이터."""

    # ---- 신호 ----
    x: np.ndarray                       # (n_channels, n_times), float32
    channels: list[str]                 # 이 샘플이 실제로 가진 채널 (개수 고정 아님)
    sfreq: float                        # 샘플링 주파수 (Hz)

    # ---- 식별 ----
    subject_id: str                     # 예: "ds004504/sub-001"
    dataset_id: str                     # 예: "ds004504"

    # ---- 라벨 ----
    disease: str                        # DISEASE_CODES 중 하나
    severity_score: Optional[float] = None   # 원점수 그대로 (구간화하지 않음)
    severity_scale: Optional[str] = None     # SEVERITY_SCALES 중 하나

    # ---- 인구통계 (선택) ----
    age: Optional[int] = None
    sex: Optional[str] = None           # "M" | "F"

    # ---- 확장 여지 ----
    # 새 데이터셋이 추가 정보를 주면 여기에 담는다.
    # Optional 필드 추가는 기존 코드를 깨지 않는다.
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------
    def validate(self) -> None:
        if self.x.ndim != 2:
            raise ValueError(f"x 는 (n_channels, n_times) 2차원이어야 합니다: {self.x.shape}")
        if self.x.shape[0] != len(self.channels):
            raise ValueError(
                f"채널 수 불일치: x={self.x.shape[0]}, channels={len(self.channels)}"
            )
        if self.disease not in DISEASE_CODES:
            raise ValueError(f"알 수 없는 질환 코드 {self.disease!r} (허용: {DISEASE_CODES})")
        if self.severity_score is not None and self.severity_scale is None:
            raise ValueError("severity_score 가 있으면 severity_scale 도 있어야 합니다")
        if self.severity_scale is not None and self.severity_scale not in SEVERITY_SCALES:
            raise ValueError(
                f"알 수 없는 척도 {self.severity_scale!r} (허용: {SEVERITY_SCALES})"
            )
        if self.x.dtype != np.float32:
            self.x = self.x.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    @property
    def n_channels(self) -> int:
        return self.x.shape[0]

    @property
    def n_times(self) -> int:
        return self.x.shape[1]

    @property
    def duration(self) -> float:
        """초 단위 길이."""
        return self.n_times / self.sfreq

    def __repr__(self) -> str:  # 디버깅용 — x 전체를 찍지 않는다
        sev = (
            f"{self.severity_scale}={self.severity_score:g}"
            if self.severity_score is not None
            else "severity=None"
        )
        return (
            f"EEGSample({self.subject_id}, {self.disease}, {sev}, "
            f"{self.n_channels}ch x {self.n_times} @ {self.sfreq:g}Hz)"
        )
