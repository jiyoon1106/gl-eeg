"""실제 .set 파일을 만들어 전체 파이프라인을 검증한다.

ds004504 과 같은 구조·사양(19채널 10-20, 500Hz, 600초)의 가짜 데이터셋을
임시 폴더에 만들고, 어댑터 → 전처리 → epoch → npz → Dataset 까지 돌린다.

    python -m tests.test_pipeline

eeglabio 가 없으면 .set 쓰기 단계만 건너뛰고 나머지를 검증한다.
    pip install eeglabio
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mne                                                        # noqa: E402

from gleeg.adapters.ds004504 import CHANNELS_AS_STORED, DS004504Adapter  # noqa: E402
from gleeg.datasets import load_bundle                            # noqa: E402
from gleeg.labels import n_levels, to_level                       # noqa: E402
from gleeg.models.losses import IGNORE                            # noqa: E402
from gleeg.preprocess.epoching import epoch_subject, make_epochs, normalize  # noqa: E402
from gleeg.preprocess.filters import get_data_uv, preprocess_raw, signal_quality  # noqa: E402
from gleeg.preprocess.montage import resolve_channels             # noqa: E402

OK = "  [OK]"
SFREQ = 500.0
DURATION = 600.0
BINS = {"scale": "MMSE", "edges": [20, 28],
        "labels": ["moderate_or_worse", "mild", "normal"]}
CLASSES = ["CN", "AD", "FTD"]


def section(t: str) -> None:
    print(f"\n{'=' * 62}\n{t}\n{'=' * 62}")


# ----------------------------------------------------------------------
def synth_raw(disease: str, seed: int) -> mne.io.RawArray:
    """질환별로 대역 파워가 다른 가짜 EEG.

    치매의 실제 소견(서파 증가 · 알파 감소)을 모사한다. 파이프라인이 그
    차이를 보존하는지 확인하는 용도.
    """
    rng = np.random.default_rng(seed)
    n = int(SFREQ * DURATION)
    t = np.arange(n) / SFREQ
    # (델타, 세타, 알파, 베타) 진폭 — CN 은 알파 우세, AD 는 서파 우세
    amp = {"CN": (5, 8, 30, 10), "AD": (20, 25, 8, 6), "FTD": (14, 18, 14, 8)}[disease]
    freqs = (2.0, 6.0, 10.0, 20.0)

    data = np.zeros((len(CHANNELS_AS_STORED), n))
    for ch in range(len(CHANNELS_AS_STORED)):
        sig = rng.normal(0, 3, n)
        for a, f in zip(amp, freqs):
            sig += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        data[ch] = sig

    info = mne.create_info(list(CHANNELS_AS_STORED), SFREQ, ch_types="eeg")
    return mne.io.RawArray(data * 1e-6, info, verbose="error")   # µV -> V


def make_fake_dataset(root: Path, n_per_group: dict[str, int]) -> bool:
    """ds004504 구조의 가짜 데이터셋. .set 쓰기 성공 여부를 반환."""
    try:
        import eeglabio  # noqa: F401
    except ImportError:
        return False

    group_map = {"C": "CN", "A": "AD", "F": "FTD"}
    mmse_pool = {"C": [30], "A": [4, 14, 16, 18, 20, 22, 23], "F": [18, 20, 22, 24, 26, 27]}

    rows = ["participant_id\tGender\tAge\tGroup\tMMSE"]
    pid = 1
    rng = np.random.default_rng(0)
    for g, n in n_per_group.items():
        for k in range(n):
            p = f"sub-{pid:03d}"
            mmse = mmse_pool[g][k % len(mmse_pool[g])]
            rows.append(f"{p}\t{rng.choice(['M','F'])}\t{rng.integers(50,80)}\t{g}\t{mmse}")

            for sub_dir in (root / p / "eeg", root / "derivatives" / p / "eeg"):
                sub_dir.mkdir(parents=True, exist_ok=True)
                raw = synth_raw(group_map[g], seed=pid)
                raw.export(
                    sub_dir / f"{p}_task-eyesclosed_eeg.set",
                    fmt="eeglab", overwrite=True, verbose="error",
                )
            pid += 1

    (root / "participants.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return True


# ----------------------------------------------------------------------
def test_preprocess_epoching() -> None:
    section("① filters / epoching — .set 없이 검증")

    raw = synth_raw("AD", seed=1)
    print(f"  원본: {len(raw.ch_names)}ch @ {raw.info['sfreq']:g}Hz, "
          f"{raw.times[-1]:.1f}s")

    chans = resolve_channels([list(CHANNELS_AS_STORED)])
    assert chans[0] == "Fp1" and len(chans) == 19
    print(f"{OK} 교집합 {len(chans)}채널, 10-20 순서로 재배열")
    print(f"       저장 순서: {CHANNELS_AS_STORED[:6]}...")
    print(f"       사용 순서: {chans[:6]}...")
    assert chans != CHANNELS_AS_STORED, "재배열이 일어나야 한다"

    p = preprocess_raw(raw, channels=chans, sfreq_target=250.0, reference="average")
    assert p.info["sfreq"] == 250.0
    assert p.ch_names == chans
    print(f"{OK} 리샘플 500 -> {p.info['sfreq']:g}Hz, average 참조, 채널 순서 적용")

    x = get_data_uv(p)
    assert x.dtype == np.float32
    print(f"{OK} 데이터 {x.shape} {x.dtype}, 범위 {x.min():.1f} ~ {x.max():.1f} µV")

    q = signal_quality(x)
    print(f"{OK} 신호 품질 {q:.3f}")

    # 정규화 -> epoch 순서
    xn = normalize(x, "robust_zscore")
    print(f"{OK} robust z-score 후 범위 {xn.min():.2f} ~ {xn.max():.2f}")

    ep = make_epochs(xn, 250.0, epoch_seconds=4.0, overlap=0.0, max_epochs=None)
    print(f"{OK} 겹침 없음  -> {ep.shape}  (예상 150)")
    assert ep.shape == (150, 19, 1000), ep.shape

    ep2 = make_epochs(xn, 250.0, epoch_seconds=4.0, overlap=0.5, max_epochs=None)
    print(f"{OK} 50% 겹침   -> {ep2.shape}  (예상 299)")

    ep3 = epoch_subject(x, 250.0, max_epochs=100)
    assert ep3.shape == (100, 19, 1000)
    print(f"{OK} 상한 100 적용 -> {ep3.shape}  (녹화 전체에서 균등 솎아내기)")

    mb = 88 * 100 * 19 * 1000 * 2 / 1e6
    print(f"{OK} 88명 환산 용량 (float16): {mb:.0f} MB")


# ----------------------------------------------------------------------
def test_full_pipeline(tmp: Path) -> None:
    section("② 어댑터 -> npz -> Dataset (실제 .set 파일로)")

    root = tmp / "ds004504"
    n_per = {"C": 3, "A": 4, "F": 3}
    if not make_fake_dataset(root, n_per):
        print("  eeglabio 가 없어 건너뜀 (pip install eeglabio)")
        return
    print(f"  가짜 데이터셋 생성: {sum(n_per.values())}명, 원본 + derivatives")

    for source in ("derivatives", "raw"):
        ad = DS004504Adapter(root, source=source)
        info = ad.verify()
        assert info["sfreq"] == SFREQ and info["n_channels"] == 19
        assert "channel_mismatch" not in info
        print(f"{OK} source={source:<12} {ad.summary()}  "
              f"sfreq={info['sfreq']:g} dur={info['duration_s']}s")

    ad = DS004504Adapter(root, source="derivatives")
    recs = ad.list_subjects()
    r0 = recs[0]
    print(f"{OK} SubjectRecord: {r0.subject_id} {r0.disease} "
          f"{r0.severity_scale}={r0.severity_score:g} age={r0.age}")
    assert r0.subject_id.startswith("ds004504/sub-")
    assert r0.severity_scale == "MMSE"

    # 전체 처리 — build_dataset 의 핵심 로직
    chans = resolve_channels([ad.channel_names()])
    K = n_levels(BINS)
    Xs, yd, yl, sid, qs = [], [], [], [], []
    for rec in recs:
        raw = preprocess_raw(ad.load_raw(rec), channels=chans, sfreq_target=250.0)
        x = get_data_uv(raw)
        ep = epoch_subject(x, 250.0, max_epochs=20)
        lv = to_level(rec.severity_score, rec.severity_scale, BINS)
        Xs.append(ep.astype(np.float16))
        yd.append(np.full(len(ep), CLASSES.index(rec.disease), np.int8))
        yl.append(np.full(len(ep), IGNORE if lv is None else lv, np.int8))
        sid.append(np.full(len(ep), rec.subject_id, dtype=object))
        qs.append(np.full(len(ep), signal_quality(x), np.float32))

    X = np.concatenate(Xs)
    print(f"{OK} 처리 완료 X {X.shape} {X.dtype}")

    out = tmp / "test.npz"
    np.savez_compressed(
        out, X=X,
        y_disease=np.concatenate(yd), y_level=np.concatenate(yl),
        severity_score=np.zeros(len(X), np.float32),
        subject_id=np.concatenate(sid).astype("U48"),
        dataset_id=np.full(len(X), "ds004504", dtype="U24"),
        quality=np.concatenate(qs),
        channels=np.array(chans, dtype="U8"), sfreq=np.float32(250.0),
        meta=json.dumps({"disease_classes": CLASSES, "severity_bins": BINS}),
    )
    print(f"{OK} npz 저장 {out.stat().st_size/1e6:.1f} MB")

    b = load_bundle(out)
    print(f"{OK} load_bundle: {b.describe()}")
    assert b.n_subjects == sum(n_per.values())
    assert b.channels == chans

    # 저채널 실험 — 한 줄
    b4 = b.select_channels(["Fp1", "Fp2", "F3", "F4"])
    assert b4.X.shape[1] == 4
    print(f"{OK} select_channels(frontal4) -> {b4.X.shape}  {b4.channels}")

    # DataLoader + 모델 연결
    import torch

    from gleeg import evaluate
    from gleeg.datasets import make_loaders
    from gleeg.models.backbones import DummyBackbone, count_params
    from gleeg.models.heads import MultiTaskModel
    from gleeg.models.losses import coral_predict, multitask_loss

    splits = evaluate.make_splits(b.subject_id, b.y_disease, n_splits=3, seed=0)
    tr, te = splits[0]
    evaluate.assert_no_leakage(b.subject_id, tr, te)
    print(f"{OK} 3-fold 분할 · 피험자 누수 없음 "
          f"(train {len(np.unique(b.subject_id[tr]))}명 / test {len(np.unique(b.subject_id[te]))}명)")

    dl_tr, dl_te = make_loaders(b, tr, te, batch_size=8, num_workers=0)
    batch = next(iter(dl_tr))
    print(f"{OK} 배치: x {tuple(batch['x'].shape)} {batch['x'].dtype} · "
          f"y_disease {tuple(batch['y_disease'].shape)} · quality {tuple(batch['quality'].shape)}")

    model = MultiTaskModel(
        DummyBackbone(len(b.channels), b.X.shape[2], 64), 64, len(CLASSES), K
    )
    out_m = model(batch["x"], quality=batch["quality"])
    loss, parts = multitask_loss(out_m, batch["y_disease"], batch["y_level"], K)
    loss.backward()
    print(f"{OK} 모델 forward/backward 정상 · 파라미터 {count_params(model):,}개")
    print(f"       손실 {parts}")
    print(f"       quality 전달 확인: {out_m.quality.flatten()[:3].tolist()}")

    # 한 fold 실제 평가
    model.eval()
    yt, yp, lt, lp = [], [], [], []
    with torch.no_grad():
        for bt in dl_te:
            o = model(bt["x"])
            yt += bt["y_disease"].tolist()
            yp += o.disease_logits.argmax(1).tolist()
            lt += bt["y_level"].tolist()
            lp += coral_predict(o.severity_logits).tolist()
    m = evaluate.compute_metrics(yt, yp, CLASSES, lt, lp, n_levels=K)
    print(f"{OK} 평가 동작 — acc {m['disease_acc']:.3f}, "
          f"민감도 {m.get('screening_sensitivity', float('nan')):.3f}, "
          f"중증도 MAE {m.get('severity_mae', float('nan')):.3f}")
    print("       (학습 전 랜덤 가중치라 수치 자체는 의미 없음 — 경로 검증)")


# ----------------------------------------------------------------------
def main() -> int:
    print("gl-eeg 파이프라인 테스트")
    tmp = Path(tempfile.mkdtemp(prefix="gleeg_"))
    try:
        test_preprocess_epoching()
        test_full_pipeline(tmp)
    except Exception as e:
        print(f"\n  [FAIL] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'=' * 62}\n전체 통과 — 원본 -> npz -> 학습 경로가 연결됩니다.\n{'=' * 62}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
