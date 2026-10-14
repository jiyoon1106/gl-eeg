"""학습 실행.

    python -m scripts.run_train --model ShallowFBCSPNet
    python -m scripts.run_train --model EEGNet --channels frontal4
    python -m scripts.run_train --model baseline

결과는 results/<태그>/ 에 metrics.json + config.json 으로 저장된다.
단계 수나 채널 수가 바뀌면 지표를 직접 비교할 수 없으므로 설정을 함께 남긴다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import DATA_DIR, RESULTS_DIR, load_exp          # noqa: E402
from gleeg import evaluate                                         # noqa: E402
from gleeg.datasets import load_bundle                             # noqa: E402
from gleeg.labels import describe_bins                             # noqa: E402
from gleeg.preprocess.montage import SUBSETS                       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="base")
    ap.add_argument("--npz", default=None, help="기본: data/npz/<exp>_derivatives.npz")
    ap.add_argument("--model", default=None,
                    help="ShallowFBCSPNet | EEGNet | SincShallowNet | baseline")
    ap.add_argument("--channels", default=None, choices=list(SUBSETS),
                    help="저채널 실험 (기본: 전체)")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--tag", default=None, help="결과 폴더 이름")
    args = ap.parse_args()

    cfg = load_exp(args.exp)
    if args.model:
        cfg["model"]["backbone"] = args.model
    if args.max_epochs:
        cfg["train"]["max_epochs"] = args.max_epochs

    npz = Path(args.npz) if args.npz else DATA_DIR / "npz" / f"{args.exp}_derivatives.npz"
    b = load_bundle(npz)

    if args.channels:
        b = b.select_channels(list(SUBSETS[args.channels]))

    model_name = "baseline" if args.model == "baseline" else cfg["model"]["backbone"]
    tag = args.tag or f"{time.strftime('%m%d-%H%M')}_{model_name}_{len(b.channels)}ch"

    print("=" * 70)
    print(f"  실험 {tag}")
    print(f"  데이터 {npz.name} — {b.describe()}")
    print(f"  채널 {len(b.channels)}개: {b.channels}")
    print(f"  중증도: {describe_bins(b.meta['severity_bins'])}")
    print(f"  모델 {model_name}")
    print("=" * 70)

    if args.model == "baseline":
        from gleeg.train.loop import cross_validate_baseline

        res = cross_validate_baseline(b, cfg)
    else:
        from gleeg.train.loop import cross_validate

        res = cross_validate(b, cfg)

    print("\n" + "=" * 70)
    print(f"  [ epoch 단위 ]  — 샘플은 {b.n_epochs}개로 보이지만 실제 표본은 "
          f"{b.n_subjects}명이다")
    print(evaluate.format_report(res["epoch_level"], unit="epoch 단위 집계"))
    print("\n  [ 피험자 단위 ] — 이쪽이 정직한 수치. 보고·발표는 이것으로")
    print(evaluate.format_report(res["subject_level"], unit="피험자 단위 집계"))
    print("=" * 70)

    out = evaluate.save_results(
        {"epoch_level": res["epoch_level"], "subject_level": res["subject_level"],
         "best_epochs": res.get("best_epochs")},
        {"exp": args.exp, "model": model_name, "channels": b.channels,
         "n_channels": len(b.channels), "npz": str(npz), "config": cfg,
         "data_meta": {k: v for k, v in b.meta.items() if k != "config"}},
        RESULTS_DIR / tag,
    )
    print(f"\n  저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
