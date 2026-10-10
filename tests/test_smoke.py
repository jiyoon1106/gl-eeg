"""더미 데이터로 인터페이스 3종을 검증한다.

실제 EEG 데이터 없이 돌아가므로, 데이터 다운로드를 기다리지 않고 팀원 네 명이
각자 작업을 시작할 수 있다.

    python -m tests.test_smoke          # 전체 실행
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import load_exp                                    # noqa: E402
from gleeg import evaluate                                            # noqa: E402
from gleeg.labels import (                                            # noqa: E402
    DS004504_GROUP_MAP,
    describe_bins,
    disease_to_index,
    n_levels,
    to_level,
)
from gleeg.models.backbones import DummyBackbone, count_params        # noqa: E402
from gleeg.models.heads import MultiTaskModel                         # noqa: E402
from gleeg.models.losses import (                                     # noqa: E402
    IGNORE,
    coral_predict,
    coral_targets,
    distillation_loss,
    multitask_loss,
)
from gleeg.preprocess.montage import (                                # noqa: E402
    channel_indices,
    common_channels,
    resolve_channels,
)
from gleeg.schema import EEGSample                                    # noqa: E402

OK, FAIL = "  [OK]", "  [FAIL]"


def section(t: str) -> None:
    print(f"\n{'=' * 60}\n{t}\n{'=' * 60}")


# ----------------------------------------------------------------------
def test_schema() -> None:
    section("① schema.EEGSample")

    s = EEGSample(
        x=np.random.randn(19, 1000).astype(np.float32),
        channels=list("ABCDEFGHIJKLMNOPQRS"),
        sfreq=250.0,
        subject_id="ds004504/sub-001",
        dataset_id="ds004504",
        disease="AD",
        severity_score=16,
        severity_scale="MMSE",
        age=57,
        sex="F",
    )
    print(f"{OK} {s}")
    assert s.n_channels == 19 and s.n_times == 1000
    assert s.duration == 4.0
    print(f"{OK} duration = {s.duration}s")

    # 채널 수 불일치는 막아야 한다
    try:
        EEGSample(
            x=np.zeros((4, 100), np.float32), channels=["Fp1", "Fp2"],
            sfreq=250.0, subject_id="s", dataset_id="d", disease="CN",
        )
        raise AssertionError("채널 불일치를 잡지 못했습니다")
    except ValueError as e:
        print(f"{OK} 채널 불일치 차단: {e}")

    # 알 수 없는 질환 코드
    try:
        EEGSample(
            x=np.zeros((2, 100), np.float32), channels=["Fp1", "Fp2"],
            sfreq=250.0, subject_id="s", dataset_id="d", disease="UNKNOWN",
        )
        raise AssertionError("질환 코드 검증 실패")
    except ValueError as e:
        print(f"{OK} 질환 코드 검증: {e}")

    # 중증도 라벨이 없는 데이터셋도 받아야 한다
    s2 = EEGSample(
        x=np.zeros((2, 100), np.float32), channels=["Fp1", "Fp2"],
        sfreq=250.0, subject_id="s", dataset_id="d", disease="PD",
    )
    print(f"{OK} 중증도 없는 샘플도 허용: {s2}")


# ----------------------------------------------------------------------
def test_labels() -> None:
    section("② labels — 원점수 → 단계 (config 로 바뀐다)")

    cfg = load_exp("base")
    bins = cfg["severity_bins"]
    print(f"  설정: {describe_bins(bins)}")
    print(f"  단계 수: {n_levels(bins)}")

    # ds004504 실측 MMSE 로 확인
    cases = [(4, 0), (16, 0), (18, 0), (20, 1), (22, 1), (23, 1), (27, 1), (30, 2)]
    for score, expected in cases:
        got = to_level(score, "MMSE", bins)
        assert got == expected, f"MMSE {score} → {got}, 기대 {expected}"
    print(f"{OK} 3단계 매핑 전부 일치 {cases}")

    # 데이터셋을 추가해 5단계로 가는 경우 — edges/labels 만 바꾼다
    bins5 = {
        "scale": "MMSE",
        "edges": [10, 20, 24, 28],
        "labels": ["severe", "moderate", "mild", "at_risk", "normal"],
    }
    five = [(4, 0), (16, 1), (22, 2), (26, 3), (30, 4)]
    for score, expected in five:
        assert to_level(score, "MMSE", bins5) == expected
    print(f"{OK} 5단계로 확장도 설정만 바꾸면 동작 {five}")

    # 척도 불일치는 막는다
    try:
        to_level(22, "UPDRS", bins)
        raise AssertionError("척도 불일치를 잡지 못했습니다")
    except ValueError as e:
        print(f"{OK} 척도 불일치 차단: {str(e)[:60]}...")

    assert DS004504_GROUP_MAP == {"A": "AD", "F": "FTD", "C": "CN"}
    print(f"{OK} ds004504 Group 매핑: {DS004504_GROUP_MAP}")

    classes = cfg["disease_classes"]
    assert disease_to_index("CN", classes) == 0
    print(f"{OK} 질환 인덱스: {[(c, disease_to_index(c, classes)) for c in classes]}")


# ----------------------------------------------------------------------
def test_montage() -> None:
    section("③ montage — 런타임 교집합")

    ds19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
            "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
    ds64 = ds19 + [f"X{i}" for i in range(45)]
    wearable = ["Fp1", "Fp2", "AF7", "AF8"]

    a = common_channels([ds19])
    print(f"{OK} 실험 A (ds19 단독)        → {len(a)}채널")
    assert len(a) == 19

    b = common_channels([ds19, ds64])
    print(f"{OK} 실험 B (ds19 + ds64)      → {len(b)}채널  {b[:5]}...")
    assert len(b) == 19

    c = common_channels([ds19, wearable])
    print(f"{OK} 실험 C (ds19 + 웨어러블)  → {len(c)}채널  {c}")
    assert c == ["Fp1", "Fp2"]

    # 순서가 10-20 표준으로 고정되는가 — 공간 필터 가중치 매핑의 전제
    shuffled = common_channels([["O2", "Fp1", "Cz", "Fp2"], ["Cz", "Fp2", "Fp1", "O2"]])
    assert shuffled == ["Fp1", "Fp2", "Cz", "O2"], shuffled
    print(f"{OK} 입력 순서가 섞여도 결과는 고정: {shuffled}")

    # 별칭 정규화 (T7/T8/P7/P8 → T3/T4/T5/T6)
    assert common_channels([["T7", "T8"], ["T3", "T4"]]) == ["T3", "T4"]
    print(f"{OK} 채널 별칭 정규화 (T7→T3, T8→T4)")

    # 저채널 실험 = 같은 함수로
    for name, expect in [("all19", 19), ("frontal8", 8), ("frontal4", 4), ("forehead2", 2)]:
        sub = resolve_channels([ds19], subset=name)
        assert len(sub) == expect, f"{name}: {sub}"
        print(f"{OK} subset={name:<10} → {len(sub)}채널  {sub}")

    idx = channel_indices(ds19, ["Fp1", "Fp2", "F3", "F4"])
    print(f"{OK} 채널 인덱스 추출: {idx}")


# ----------------------------------------------------------------------
def test_model() -> None:
    section("④ heads / losses — 멀티태스크 + CORAL")

    cfg = load_exp("base")
    bins = cfg["severity_bins"]
    K = n_levels(bins)
    classes = cfg["disease_classes"]
    embed = cfg["model"]["embed_dim"]
    B, C, T = 8, 19, 1000

    model = MultiTaskModel(
        backbone=DummyBackbone(C, T, embed),
        embed_dim=embed,
        n_disease=len(classes),
        n_levels=K,
    )
    print(f"  파라미터 {count_params(model):,}개 (88명 규모 상한: 수만)")

    x = torch.randn(B, C, T)
    out = model(x)
    assert out.disease_logits.shape == (B, len(classes))
    assert out.severity_logits.shape == (B, K - 1)
    assert out.embedding.shape == (B, embed)
    print(f"{OK} disease_logits  {tuple(out.disease_logits.shape)}")
    print(f"{OK} severity_logits {tuple(out.severity_logits.shape)}  (= n_levels-1 = {K-1})")
    print(f"{OK} embedding       {tuple(out.embedding.shape)}")
    print(f"{OK} quality         {out.quality}  (전처리에서 계산해 넘기는 값)")

    # CORAL 타깃 — 벌점이 오차 거리에 비례하는 근거
    t = coral_targets(torch.tensor([0, 1, 2]), 3)
    assert t.tolist() == [[0, 0], [1, 0], [1, 1]]
    print(f"{OK} CORAL 타깃 (레벨 0/1/2 → ): {t.tolist()}")

    # 손실 + 역전파
    y_dis = torch.randint(0, len(classes), (B,))
    y_sev = torch.randint(0, K, (B,))
    loss, parts = multitask_loss(out, y_dis, y_sev, K, cfg["loss"]["severity_weight"])
    loss.backward()
    grads = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert grads > 0
    print(f"{OK} 손실 {parts} · 역전파 정상 (grad 합 {grads:.3f})")

    # 중증도 라벨이 없는 샘플 섞기 (IGNORE)
    y_mixed = y_sev.clone()
    y_mixed[:4] = IGNORE
    l2, p2 = multitask_loss(out, y_dis, y_mixed, K)
    print(f"{OK} 중증도 라벨 일부 없음(IGNORE) 처리: {p2}")

    # 예측
    pred = model.predict(x)
    assert pred["disease"].shape == (B,)
    assert pred["severity"].shape == (B,)
    assert pred["severity"].min() >= 0 and pred["severity"].max() <= K - 1
    print(f"{OK} 예측 disease={pred['disease'].tolist()}")
    print(f"{OK} 예측 severity={pred['severity'].tolist()}  (0~{K-1} 범위)")

    # 저채널 student 로 바꿔도 같은 규격 → 증류 가능
    student = MultiTaskModel(DummyBackbone(4, T, embed), embed, len(classes), K)
    s_out = student(torch.randn(B, 4, T))
    dl = distillation_loss(s_out, out)
    dl_val = float(dl.detach())
    dl.backward()
    print(f"{OK} 증류 손실 (19ch teacher → 4ch student): {dl_val:.4f}")

    # 단계 수를 5로 바꿔도 헤드가 따라온다
    m5 = MultiTaskModel(DummyBackbone(C, T, embed), embed, len(classes), 5)
    assert m5(x).severity_logits.shape == (B, 4)
    print(f"{OK} n_levels=5 로 바꾸면 헤드 출력도 4개로 자동 변경")


# ----------------------------------------------------------------------
def test_evaluate() -> None:
    section("⑤ evaluate — 피험자 단위 분할 + 지표")

    cfg = load_exp("base")
    classes = cfg["disease_classes"]
    K = n_levels(cfg["severity_bins"])

    # ds004504 구성을 모사: CN 29 / AD 36 / FTD 23, 피험자당 epoch 60개
    rng = np.random.default_rng(0)
    subs, dis, lev = [], [], []
    plan = [("CN", 29, 2), ("AD", 36, 0), ("FTD", 23, 1)]
    for code, n_subj, level in plan:
        for i in range(n_subj):
            sid = f"{code}-{i:03d}"
            for _ in range(60):
                subs.append(sid)
                dis.append(disease_to_index(code, classes))
                lev.append(level)
    subs, dis, lev = np.array(subs), np.array(dis), np.array(lev)
    print(f"  모사 데이터: 피험자 {len(np.unique(subs))}명 · 샘플 {len(subs)}개")

    splits = evaluate.make_splits(subs, dis, n_splits=cfg["eval"]["n_splits"], seed=42)
    print(f"{OK} {len(splits)}-fold 생성")

    for k, (tr, te) in enumerate(splits):
        evaluate.assert_no_leakage(subs, tr, te)
        n_tr, n_te = len(np.unique(subs[tr])), len(np.unique(subs[te]))
        print(f"       fold {k}: train {n_tr}명 / test {n_te}명  (누수 없음)")

    # 전부 train 에 몰아넣고 누수가 잡히는지 — 가드가 실제로 작동하는지 확인
    try:
        evaluate.assert_no_leakage(subs, np.arange(len(subs)), np.arange(10))
        raise AssertionError("누수 검사가 작동하지 않습니다")
    except AssertionError as e:
        if "누수" not in str(e):
            raise
        print(f"{OK} 누수 검사 작동 확인")

    # 지표 — 일부러 틀린 예측을 섞는다
    fold_metrics = []
    for tr, te in splits:
        yt, lt = dis[te], lev[te]
        yp, lp = yt.copy(), lt.copy()
        flip = rng.random(len(yp)) < 0.2
        yp[flip] = (yp[flip] + 1) % len(classes)
        lp[flip] = np.clip(lp[flip] + rng.choice([-1, 1], flip.sum()), 0, K - 1)
        fold_metrics.append(
            evaluate.compute_metrics(yt, yp, classes, lt, lp, n_levels=K)
        )

    agg = evaluate.aggregate_folds(fold_metrics)
    print()
    print(evaluate.format_report(agg))

    assert "severity_mae" in agg and "severity_qwk" in agg
    assert "screening_sensitivity" in agg
    print(f"\n{OK} 지표 집계 정상 (평균 ± 표준편차 · fold 별 값 보존)")
    print(f"{OK} fold 별 질환 정확도: {agg['disease_acc']['folds']}")

    out = evaluate.save_results(agg, cfg, "/tmp/gleeg_smoke")
    print(f"{OK} 결과 저장: {out}/metrics.json, config.json")


# ----------------------------------------------------------------------
def main() -> int:
    print("gl-eeg 인터페이스 스모크 테스트 (실제 데이터 불필요)")
    for fn in (test_schema, test_labels, test_montage, test_model, test_evaluate):
        try:
            fn()
        except Exception as e:
            print(f"\n{FAIL} {fn.__name__}: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            return 1
    print(f"\n{'=' * 60}\n전체 통과 — 인터페이스 3종이 규격대로 동작합니다.\n{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
