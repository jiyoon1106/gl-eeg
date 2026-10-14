"""ds004504 어댑터 — 알츠하이머 / 전두측두치매 / 정상 대조군 EEG.

실측 사양 (2026-10-10 확인)
---------------------------
    피험자  88명  (A 36 / C 29 / F 23)
    채널    19개, 10-20 표준 이름 그대로
            Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 Fz Cz Pz
    샘플링  500 Hz
    참조    A1 A2 (양쪽 귀)
    필터    0.4~50 Hz 적용됨
    길이    599.8초 (약 10분)
    형식    EEGLAB .set  (MNE read_raw_eeglab 로 읽힘)

원본과 전처리본(derivatives) 둘 다 제공된다. derivatives 는 저자들이
밴드패스 + ASR + ICA 로 아티팩트를 제거한 버전이다.

    source="raw"          원본 — 전처리를 우리가 통제. 재현성·설명 가능
    source="derivatives"  전처리본 — 바로 학습 가능. 빠른 베이스라인용

둘 다 지원하므로 비교 실험이 가능하다. derivatives 로 먼저 결과를 뽑고,
원본 전처리는 그다음에 하는 순서를 권한다.

⚠️ 라벨 품질 (scripts/peek_labels.py 참조)
   대조군 29명 전원 MMSE=30, std=0 → 실측값이 아니라 만점 기록.
   MMSE 28·29 구간 0명, 임상 중증(<10) 3명.
   그래서 중증도는 3단계로 축소했다. config/exp/base.yaml 의 severity_bins.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..labels import DS004504_GROUP_MAP
from .base import BaseAdapter, SubjectRecord

TASK = "eyesclosed"

# channels.tsv 기준 원본 순서. 10-20 표준 순서와 다르므로
# montage.sort_canonical() 이 재배열한다.
CHANNELS_AS_STORED = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]


class DS004504Adapter(BaseAdapter):
    dataset_id = "ds004504"

    def __init__(self, root: Path | str, source: str = "derivatives") -> None:
        self.root = Path(root)
        if source not in ("raw", "derivatives"):
            raise ValueError(f'source 는 "raw" 또는 "derivatives": {source!r}')
        self.source = source

        tsv = self.root / "participants.tsv"
        if not tsv.exists():
            raise FileNotFoundError(
                f"participants.tsv 가 없습니다: {tsv}\n"
                "먼저 받으세요: openneuro-py download --dataset=ds004504 "
                "--target-dir=data/ds004504"
            )
        self._participants = pd.read_csv(tsv, sep="\t")

    # ------------------------------------------------------------------
    def _signal_path(self, participant_id: str) -> Path:
        fname = f"{participant_id}_task-{TASK}_eeg.set"
        if self.source == "derivatives":
            return self.root / "derivatives" / participant_id / "eeg" / fname
        return self.root / participant_id / "eeg" / fname

    def list_subjects(self) -> list[SubjectRecord]:
        recs: list[SubjectRecord] = []
        for row in self._participants.itertuples(index=False):
            pid = row.participant_id
            group = row.Group
            if group not in DS004504_GROUP_MAP:
                raise ValueError(f"{pid}: 알 수 없는 Group {group!r}")

            path = self._signal_path(pid)
            if not path.exists():
                import warnings

                warnings.warn(f"{pid}: 신호 파일 없음, 건너뜀 ({path})", stacklevel=2)
                continue

            recs.append(
                SubjectRecord(
                    subject_id=f"{self.dataset_id}/{pid}",
                    dataset_id=self.dataset_id,
                    path=path,
                    disease=DS004504_GROUP_MAP[group],
                    severity_score=float(row.MMSE),
                    severity_scale="MMSE",
                    age=int(row.Age),
                    sex=str(row.Gender),
                    extra={"source": self.source, "group_raw": group},
                )
            )
        return recs

    def load_raw(self, rec: SubjectRecord):
        import mne

        return mne.io.read_raw_eeglab(rec.path, preload=False, verbose="error")

    def channel_names(self) -> list[str]:
        """실측으로 확인된 19채널. 파일을 열지 않아 빠르다.

        교집합 계산 단계에서 호출되므로 I/O 없이 반환하는 게 중요하다.
        """
        return list(CHANNELS_AS_STORED)

    # ------------------------------------------------------------------
    def verify(self) -> dict:
        """대표 1명을 실제로 열어 사양이 기대와 맞는지 확인한다.

        build_dataset 을 돌리기 전에 한 번 호출하면 88명을 다 처리한 뒤에
        문제를 발견하는 일을 막는다.
        """
        recs = self.list_subjects()
        if not recs:
            raise RuntimeError("피험자가 하나도 없습니다")

        raw = self.load_raw(recs[0])
        info = {
            "n_subjects": len(recs),
            "source": self.source,
            "sfreq": float(raw.info["sfreq"]),
            "n_channels": len(raw.ch_names),
            "channels": list(raw.ch_names),
            "duration_s": round(float(raw.times[-1]), 1),
        }

        expected = set(CHANNELS_AS_STORED)
        actual = set(raw.ch_names)
        if expected != actual:
            info["channel_mismatch"] = {
                "missing": sorted(expected - actual),
                "unexpected": sorted(actual - expected),
            }
        return info
