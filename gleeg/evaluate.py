"""인터페이스 ③ — 공통 평가.

전원 이 모듈로만 성능을 보고한다. 평가 경로가 갈라지면 팀 결과를 비교할 수 없다.

가장 중요한 규칙: 피험자 단위 분할
----------------------------------
4초 epoch 으로 자르면 88명이 수천 샘플이 된다. 이때 같은 사람의 epoch 이 train
과 test 양쪽에 들어가면 정확도가 허위로 급등한다. 이 분야 논문의 최빈 오류다.
make_splits() 가 subject_id 를 groups 로 받아 이를 코드로 강제한다.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold


# ----------------------------------------------------------------------
# 분할
# ----------------------------------------------------------------------
def make_splits(
    subject_ids: Sequence[str],
    stratify: Sequence[int],
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """피험자 단위 층화 K-fold.

    같은 subject_id 의 샘플은 절대 train/test 로 쪼개지지 않는다.

    Parameters
    ----------
    subject_ids : 샘플별 피험자 ID (길이 = 샘플 수)
    stratify : 샘플별 층화 라벨 (보통 질환 인덱스)
    """
    subject_ids = np.asarray(subject_ids)
    stratify = np.asarray(stratify)
    if len(subject_ids) != len(stratify):
        raise ValueError(
            f"길이 불일치: subject_ids={len(subject_ids)}, stratify={len(stratify)}"
        )

    n_subj = len(np.unique(subject_ids))
    if n_subj < n_splits:
        raise ValueError(f"피험자 {n_subj}명으로 {n_splits}-fold 는 불가능합니다")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_dummy = np.zeros((len(subject_ids), 1))
    return list(sgkf.split(X_dummy, stratify, groups=subject_ids))


def assert_no_leakage(
    subject_ids: Sequence[str], train_idx: np.ndarray, test_idx: np.ndarray
) -> None:
    """피험자 누수 검사. 학습 루프에서 fold 마다 불러 두면 사고를 막는다."""
    subject_ids = np.asarray(subject_ids)
    overlap = set(subject_ids[train_idx]) & set(subject_ids[test_idx])
    if overlap:
        raise AssertionError(
            f"피험자 누수! train 과 test 에 같은 피험자가 있습니다: {sorted(overlap)[:5]}"
        )


# ----------------------------------------------------------------------
# 지표
# ----------------------------------------------------------------------
def compute_metrics(
    y_disease_true: Sequence[int],
    y_disease_pred: Sequence[int],
    disease_classes: Sequence[str],
    y_level_true: Optional[Sequence[int]] = None,
    y_level_pred: Optional[Sequence[int]] = None,
    n_levels: Optional[int] = None,
    normal_class: str = "CN",
) -> dict:
    """질환 분류 + 중증도 지표를 한 번에.

    중증도는 정확도가 아니라 **MAE(평균 레벨 오차)와 QWK**로 본다. 순서형
    출력에서는 "몇 단계 빗나갔는지"가 "맞췄는지"보다 의미 있는 정보다.
    """
    yt = np.asarray(y_disease_true)
    yp = np.asarray(y_disease_pred)

    m: dict = {
        "n_samples": int(len(yt)),
        "disease_acc": float(accuracy_score(yt, yp)),
        "disease_f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "disease_classes": list(disease_classes),
        "confusion_matrix": confusion_matrix(
            yt, yp, labels=range(len(disease_classes))
        ).tolist(),
    }

    # 클래스별 재현율 — 어느 질환을 놓치는지가 보여야 한다
    per_class = recall_score(
        yt, yp, labels=range(len(disease_classes)), average=None, zero_division=0
    )
    m["disease_recall"] = {c: float(r) for c, r in zip(disease_classes, per_class)}

    # 선별(screening) 민감도 — 환자를 놓치지 않는가
    # 치매 선별에서 놓침(FN)은 조기 개입 실패, 오탐(FP)은 불필요한 정밀검사다.
    # 비용이 다르므로 정확도보다 민감도를 먼저 본다.
    if normal_class in disease_classes:
        normal_idx = list(disease_classes).index(normal_class)
        patient_true = (yt != normal_idx).astype(int)
        patient_pred = (yp != normal_idx).astype(int)
        m["screening_sensitivity"] = float(
            recall_score(patient_true, patient_pred, zero_division=0)
        )
        m["screening_specificity"] = float(
            recall_score(1 - patient_true, 1 - patient_pred, zero_division=0)
        )

    # 중증도
    if y_level_true is not None and y_level_pred is not None:
        lt = np.asarray(y_level_true)
        lp = np.asarray(y_level_pred)
        valid = lt >= 0                      # IGNORE(-100) 제외
        lt, lp = lt[valid], lp[valid]
        if len(lt):
            labels = list(range(n_levels)) if n_levels else None
            m["severity_mae"] = float(np.abs(lt - lp).mean())
            m["severity_acc"] = float(accuracy_score(lt, lp))
            m["severity_exact_or_adjacent"] = float((np.abs(lt - lp) <= 1).mean())
            try:
                m["severity_qwk"] = float(
                    cohen_kappa_score(lt, lp, weights="quadratic", labels=labels)
                )
            except ValueError:
                m["severity_qwk"] = float("nan")   # 한 클래스만 존재하는 fold
            m["severity_n"] = int(len(lt))

    return m


def aggregate_folds(fold_metrics: Sequence[dict]) -> dict:
    """fold 별 지표 → 평균 ± 표준편차.

    단일 수치로 보고하지 않는다. 88명 규모에서는 fold 간 분산이 크고, 그
    분산 자체가 결과의 신뢰도를 말해준다.
    """
    out: dict = {"n_folds": len(fold_metrics)}
    keys = [
        k
        for k in fold_metrics[0]
        if isinstance(fold_metrics[0][k], (int, float)) and k != "n_samples"
    ]
    for k in keys:
        vals = np.array([f[k] for f in fold_metrics if k in f], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals):
            out[k] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "folds": [round(float(v), 4) for v in vals],
            }
    out["total_samples"] = int(sum(f.get("n_samples", 0) for f in fold_metrics))
    return out


def format_report(agg: dict) -> str:
    """사람이 읽을 요약 — 로그·발표용."""
    lines = [f"=== {agg['n_folds']}-fold (피험자 단위) · 총 {agg['total_samples']} 샘플 ==="]
    order = [
        ("disease_acc", "질환 정확도"),
        ("disease_f1_macro", "질환 F1(macro)"),
        ("screening_sensitivity", "선별 민감도"),
        ("screening_specificity", "선별 특이도"),
        ("severity_mae", "중증도 MAE(레벨)"),
        ("severity_qwk", "중증도 QWK"),
        ("severity_exact_or_adjacent", "중증도 ±1단계 이내"),
    ]
    for key, label in order:
        if key in agg:
            s = agg[key]
            lines.append(f"  {label:<22} {s['mean']:.4f} ± {s['std']:.4f}")
    return "\n".join(lines)


def save_results(agg: dict, cfg: dict, out_dir: Path | str) -> Path:
    """결과와 설정을 함께 저장한다.

    단계 수가 바뀌면 MAE/QWK 를 직접 비교할 수 없으므로, 어떤 설정으로 돌렸는지
    를 결과와 같은 폴더에 남겨야 한다.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return out
