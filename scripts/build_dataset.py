"""원본 데이터셋 -> data.npz (한 번만 실행, 결과를 팀에 공유).

    python -m scripts.build_dataset --exp base --source derivatives
    python -m scripts.build_dataset --exp base --source raw --out data/npz/raw.npz

만들어진 .npz 를 Google Drive 공유 폴더에 올리면, Colab 쓰는 팀원들이 원본
5.4GB 를 받지 않고 바로 학습할 수 있다. 그리고 **전원이 같은 데이터를 본다.**

저장 형식
---------
    X              (n_epochs, n_channels, n_times)  float16
    y_disease      (n_epochs,)  int8   — disease_classes 인덱스
    y_level        (n_epochs,)  int8   — 중증도 단계 (없으면 -100)
    severity_score (n_epochs,)  float32 — 원점수 보존. 구간을 바꿔도 재계산 가능
    subject_id     (n_epochs,)  str
    dataset_id     (n_epochs,)  str
    quality        (n_epochs,)  float32 — 융합 가중치용 신호 품질
    channels       (n_channels,) str   — 순서가 곧 모델 입력 채널 순서
    sfreq          scalar
    meta           JSON 문자열 — 설정 전체 스냅샷
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import DATA_DIR, load_exp                      # noqa: E402
from gleeg.adapters.ds004504 import DS004504Adapter              # noqa: E402
from gleeg.labels import disease_to_index, n_levels, to_level     # noqa: E402
from gleeg.models.losses import IGNORE                           # noqa: E402
from gleeg.preprocess.epoching import epoch_subject              # noqa: E402
from gleeg.preprocess.filters import (                           # noqa: E402
    get_data_uv,
    preprocess_raw,
    signal_quality,
)
from gleeg.preprocess.montage import resolve_channels            # noqa: E402

ADAPTERS = {"ds004504": DS004504Adapter}


def build_adapters(cfg: dict, source: str) -> list:
    out = []
    for name in cfg["datasets"]:
        if name not in ADAPTERS:
            raise ValueError(f"{name!r} 어댑터가 없습니다. 등록된 것: {list(ADAPTERS)}")
        root = DATA_DIR / name
        kwargs = {"source": source} if name == "ds004504" else {}
        out.append(ADAPTERS[name](root, **kwargs))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="base", help="config/exp/<name>.yaml")
    ap.add_argument("--source", default="derivatives", choices=["raw", "derivatives"])
    ap.add_argument("--out", default=None, help="저장 경로 (기본: data/npz/<exp>_<source>.npz)")
    ap.add_argument("--limit", type=int, default=None, help="피험자 수 제한 (디버깅)")
    args = ap.parse_args()

    cfg = load_exp(args.exp)
    sig, bins = cfg["signal"], cfg["severity_bins"]
    classes = cfg["disease_classes"]
    K = n_levels(bins)

    out_path = Path(args.out) if args.out else DATA_DIR / "npz" / f"{args.exp}_{args.source}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- 어댑터 확인 ----------
    adapters = build_adapters(cfg, args.source)
    print("=" * 66)
    for ad in adapters:
        print(f"  {ad.summary()}")
        if hasattr(ad, "verify"):
            info = ad.verify()
            print(f"    sfreq={info['sfreq']} n_ch={info['n_channels']} "
                  f"duration={info['duration_s']}s source={info['source']}")
            if "channel_mismatch" in info:
                print(f"    (!) 채널 불일치: {info['channel_mismatch']}")
                return 1

    # ---------- 런타임 교집합 ----------
    chans = resolve_channels(
        [ad.channel_names() for ad in adapters], subset=cfg["channels"]["subset"]
    )
    print(f"\n  사용 채널 {len(chans)}개 (10-20 순서 고정)")
    print(f"    {chans}")
    print(f"  중증도 {K}단계 · 질환 {len(classes)}분류 {classes}")
    print("=" * 66)

    # ---------- 피험자별 처리 ----------
    Xs, yd, yl, sc, sid, did, qs = [], [], [], [], [], [], []
    skipped: list[tuple[str, str]] = []
    t0 = time.time()

    recs = [r for ad in adapters for r in ad.list_subjects()]
    if args.limit:
        recs = recs[: args.limit]
    adapter_of = {ad.dataset_id: ad for ad in adapters}

    for i, rec in enumerate(recs, 1):
        try:
            raw = adapter_of[rec.dataset_id].load_raw(rec)
            raw = preprocess_raw(
                raw,
                channels=chans,
                sfreq_target=sig["sfreq"],
                reference=sig["reference"],
                bandpass=sig.get("bandpass"),
            )
            x_uv = get_data_uv(raw)
            q = signal_quality(x_uv)

            ep = epoch_subject(
                x_uv,
                sfreq=sig["sfreq"],
                epoch_seconds=sig["epoch_seconds"],
                overlap=sig["epoch_overlap"],
                max_epochs=sig.get("max_epochs_per_subject"),
                normalize_mode=sig["normalize"],
            )
            n = len(ep)

            level = to_level(rec.severity_score, rec.severity_scale, bins)
            Xs.append(ep.astype(np.float16))
            yd.append(np.full(n, disease_to_index(rec.disease, classes), np.int8))
            yl.append(np.full(n, IGNORE if level is None else level, np.int8))
            sc.append(np.full(n, np.nan if rec.severity_score is None else rec.severity_score, np.float32))
            sid.append(np.full(n, rec.subject_id, dtype=object))
            did.append(np.full(n, rec.dataset_id, dtype=object))
            qs.append(np.full(n, q, np.float32))

            el = time.time() - t0
            print(
                f"  [{i:>3}/{len(recs)}] {rec.subject_id:<22} {rec.disease:<4} "
                f"MMSE={rec.severity_score:>4.0f} lv={level} "
                f"epochs={n:>3} q={q:.3f}  ({el:.0f}s)",
                flush=True,
            )
        except Exception as e:  # 한 명 실패로 전체가 멈추면 안 된다
            skipped.append((rec.subject_id, f"{type(e).__name__}: {e}"))
            print(f"  [{i:>3}/{len(recs)}] {rec.subject_id:<22} (!) 건너뜀 — {e}", flush=True)

    if not Xs:
        print("\n처리된 피험자가 없습니다.")
        return 1

    # ---------- 저장 ----------
    X = np.concatenate(Xs)
    meta = {
        "exp": args.exp,
        "source": args.source,
        "config": cfg,
        "channels": chans,
        "disease_classes": classes,
        "severity_bins": bins,
        "n_subjects": len(recs) - len(skipped),
        "skipped": skipped,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    np.savez_compressed(
        out_path,
        X=X,
        y_disease=np.concatenate(yd),
        y_level=np.concatenate(yl),
        severity_score=np.concatenate(sc),
        subject_id=np.concatenate(sid).astype("U48"),
        dataset_id=np.concatenate(did).astype("U24"),
        quality=np.concatenate(qs),
        channels=np.array(chans, dtype="U8"),
        sfreq=np.float32(sig["sfreq"]),
        meta=json.dumps(meta, ensure_ascii=False, default=str),
    )

    size_mb = out_path.stat().st_size / 1e6
    print("\n" + "=" * 66)
    print(f"  저장: {out_path}  ({size_mb:.1f} MB)")
    print(f"  X {X.shape} {X.dtype}   피험자 {meta['n_subjects']}명")
    uniq, cnt = np.unique(np.concatenate(yd), return_counts=True)
    print(f"  질환 분포: " + " / ".join(f"{classes[u]} {c}" for u, c in zip(uniq, cnt)))
    uniq, cnt = np.unique(np.concatenate(yl), return_counts=True)
    print(f"  중증도 분포: " + " / ".join(
        f"{'없음' if u == IGNORE else bins['labels'][u]} {c}" for u, c in zip(uniq, cnt)))
    if skipped:
        print(f"  (!) 건너뛴 피험자 {len(skipped)}명: {[s[0] for s in skipped]}")
    print(f"  소요 {time.time()-t0:.0f}초")
    print("=" * 66)
    print("\n다음: 이 파일을 Google Drive 공유 폴더에 올리면 팀원들이 바로 쓸 수 있습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
