"""학습 루프 스모크 테스트 — 데이터 없이 돈다.

실제 ds004504 없이 합성 신호로 다음을 확인한다.

1. train_fold() 가 끝까지 돌고 FoldResult 를 돌려준다
2. early stopping 용 val 이 **train 안에서** 나온다 (test 와 피험자 겹침 0)
3. epoch 단위 / 피험자 단위 지표가 둘 다 나온다
4. 베이스라인(밴드파워 + 부스팅 트리) 교차검증이 돈다
5. 저채널 서브셋으로 바꿔도 모델이 채널 수를 따라간다

    python tests/test_train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gleeg import evaluate                                    # noqa: E402
from gleeg.datasets import Bundle                              # noqa: E402

SFREQ = 250.0
CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]
CLASSES = ["CN", "AD", "FTD"]
BINS = {"scale": "MMSE", "edges": [20, 28],
        "labels": ["moderate_or_worse", "mild", "normal"]}


def make_fake_bundle(n_subjects: int = 24, n_epochs: int = 8, n_times: int = 500) -> Bundle:
    """질환별 대역 파워가 다른 합성 EEG.

    CN 은 알파(10Hz)가 강하고 AD 는 세타(6Hz)가 강하다 — 실제 치매 EEG 의
    "서파화"를 모사한 것. 학습이 전혀 안 되면 이 데이터조차 못 맞춘다.
    """
    rng = np.random.default_rng(0)
    t = np.arange(n_times) / SFREQ

    X, yd, yl, score, sid = [], [], [], [], []
    for s in range(n_subjects):
        cls = s % 3                                  # 0 CN / 1 AD / 2 FTD
        mmse = [30.0, 18.0, 24.0][cls] + rng.normal(0, 1.5)
        alpha, theta = [(1.0, 0.3), (0.3, 1.0), (0.5, 0.7)][cls]
        for _ in range(n_epochs):
            sig = (
                alpha * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6.28))
                + theta * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 6.28))
            )
            x = sig[None, :] * rng.uniform(0.7, 1.3, (len(CHANNELS), 1))
            x = x + rng.normal(0, 0.3, x.shape)
            X.append(x.astype(np.float16))
            yd.append(cls)
            yl.append(int(np.digitize(mmse, BINS["edges"])))
            score.append(mmse)
            sid.append(f"sub-{s:03d}")

    n = len(X)
    return Bundle(
        X=np.stack(X),
        y_disease=np.array(yd, dtype=np.int8),
        y_level=np.array(yl, dtype=np.int8),
        severity_score=np.array(score, dtype=np.float32),
        subject_id=np.array(sid),
        dataset_id=np.array(["fake"] * n),
        quality=np.ones(n, dtype=np.float32),
        channels=list(CHANNELS),
        sfreq=SFREQ,
        meta={"disease_classes": CLASSES, "severity_bins": BINS},
    )


def small_cfg(backbone: str = "ShallowFBCSPNet", max_epochs: int = 3) -> dict:
    return {
        "seed": 42,
        "model": {"backbone": backbone, "embed_dim": 32,
                  "backbone_kwargs": {"filter_time_length": 25, "n_filters_time": 20,
                                      "n_filters_spat": 20, "drop_prob": 0.3}
                  if backbone == "ShallowFBCSPNet" else {}},
        "loss": {"severity_weight": 1.0},
        "train": {"batch_size": 16, "lr": 0.002, "weight_decay": 0.0005,
                  "max_epochs": max_epochs, "early_stopping_patience": 5, "device": "cpu"},
        "eval": {"n_splits": 3, "group_by": "subject_id", "stratify_by": "disease"},
    }


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> int:
    import torch

    from gleeg.train.loop import (
        cross_validate,
        cross_validate_baseline,
        inner_val_split,
        train_fold,
    )

    b = make_fake_bundle()
    print(f"모사 데이터: {b.describe()}")

    # ------------------------------------------------------------------
    section("① val 은 train 에서 나온다 (test 를 보지 않는다)")
    splits = evaluate.make_splits(b.subject_id, b.y_disease, n_splits=3, seed=42)
    tr_idx, te_idx = splits[0]
    tr, va = inner_val_split(b.subject_id, b.y_disease, tr_idx, 42)

    s = lambda i: set(b.subject_id[i])                              # noqa: E731
    assert s(tr) | s(va) == s(tr_idx), "val 이 train 밖에서 왔다"
    assert not (s(tr) & s(va)), "train/val 피험자 겹침"
    assert not ((s(tr) | s(va)) & s(te_idx)), "test 피험자가 학습에 들어갔다"
    print(f"  [OK] train {len(s(tr))}명 / val {len(s(va))}명 / test {len(s(te_idx))}명")
    print("  [OK] 세 집합 모두 피험자 겹침 0 — early stopping 이 test 를 보지 않는다")

    # ------------------------------------------------------------------
    section("② train_fold — 끝까지 돈다")
    r = train_fold(b, small_cfg(), tr_idx, te_idx, 0, torch.device("cpu"))
    assert r.history and "val_f1" in r.history[0]
    for key in ("disease_acc", "disease_f1_macro", "screening_sensitivity"):
        assert key in r.metrics_epoch and key in r.metrics_subject, key
    assert "severity_mae" in r.metrics_subject, "중증도 지표가 없다"
    print(f"  [OK] {len(r.history)} epoch 학습 · best ep {r.best_epoch}")
    print(f"  [OK] epoch 단위 n={r.metrics_epoch['n_samples']} / "
          f"피험자 단위 n={r.metrics_subject['n_samples']}")
    assert r.metrics_subject["n_samples"] == len(s(te_idx)), "피험자 단위 집계 수가 안 맞는다"
    print("  [OK] 피험자 단위 샘플 수 = test 피험자 수 (확률 평균이 제대로 모였다)")

    # ------------------------------------------------------------------
    section("③ cross_validate — 3-fold 전체")
    res = cross_validate(b, small_cfg(max_epochs=2), torch.device("cpu"))
    assert set(res) >= {"epoch_level", "subject_level", "best_epochs"}
    print(evaluate.format_report(res["subject_level"]))
    print("  [OK] 두 단위 지표 모두 집계됨")

    # ------------------------------------------------------------------
    section("④ 베이스라인 — 밴드파워 + 부스팅 트리")
    bl = cross_validate_baseline(b, small_cfg())
    print(evaluate.format_report(bl["subject_level"]))
    acc = bl["subject_level"]["disease_acc"]["mean"]
    assert acc > 0.5, f"합성 데이터조차 못 맞춘다 (acc={acc:.3f}) — 특징 추출 점검 필요"
    print(f"  [OK] 합성 신호의 대역 파워 차이를 잡아낸다 (acc {acc:.3f})")

    # ------------------------------------------------------------------
    section("⑤ 저채널 — 채널 수가 코드에 박혀 있지 않다")
    b4 = b.select_channels(["Fp1", "Fp2", "F3", "F4"])
    assert b4.X.shape[1] == 4 and len(b4.channels) == 4
    r4 = train_fold(b4, small_cfg(max_epochs=2), tr_idx, te_idx, 0, torch.device("cpu"),
                    verbose=False)
    print(f"  [OK] 4채널로도 학습 완료 — 피험자 단위 acc "
          f"{r4.metrics_subject['disease_acc']:.3f}")

    b2 = b.select_channels(["Fp1", "Fp2"])
    r2 = train_fold(b2, small_cfg(max_epochs=2), tr_idx, te_idx, 0, torch.device("cpu"),
                    verbose=False)
    print(f"  [OK] 2채널(이마 전극)도 학습 완료 — acc "
          f"{r2.metrics_subject['disease_acc']:.3f}")

    # ------------------------------------------------------------------
    section("⑥ EEGNet 로 백본 교체")
    rn = train_fold(b, small_cfg("EEGNet", max_epochs=2), tr_idx, te_idx, 0,
                    torch.device("cpu"), verbose=False)
    print(f"  [OK] EEGNet 학습 완료 — acc {rn.metrics_subject['disease_acc']:.3f}")

    # ------------------------------------------------------------------
    section("⑦ 백본 교체 — 인자 이름이 달라도 생성된다")
    from gleeg.train.loop import build_model

    for name in ("ShallowFBCSPNet", "EEGNet", "SincShallowNet", "EEGNeX",
                 "FBCNet", "Deep4Net"):
        cfg = small_cfg()
        cfg["model"]["backbone"] = name
        cfg["model"]["backbone_kwargs"] = {        # ShallowFBCSPNet 전용 인자
            "filter_time_length": 25, "n_filters_time": 40,
            "n_filters_spat": 40, "drop_prob": 0.5,
        }
        row = []
        for C in (19, 4, 2):
            m = build_model(cfg, C, 500, 3, 3, SFREQ)
            row.append(f"{C}ch {sum(p.numel() for p in m.parameters()):,}")
        print(f"  [OK] {name:<16} " + " | ".join(row))
    print("  주력 백본 인자를 그대로 줘도 받지 않는 쪽은 걸러낸다")
    print("  (sfreq 는 SincShallowNet·FBCNet 처럼 Hz 로 필터를 만드는 백본에만 전달)")

    print("\n" + "=" * 60)
    print("전체 통과 — 학습 루프가 규격대로 동작합니다.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
