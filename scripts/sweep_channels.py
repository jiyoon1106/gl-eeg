"""채널 수 vs 성능 곡선 — 모델을 고정하고 채널만 줄인다.

    python -m scripts.sweep_channels --model EEGNet
    python -m scripts.sweep_channels --model ShallowFBCSPNet --subsets all19,frontal4

지엘 제품은 이마 전극만 쓴다. 19채널에서 얻은 성능이 4채널·2채널에서 얼마나
남는지가 이 과제의 차별점이고, 그 숫자가 곧 비즈니스 KPI(센서 비용·착용
시간·검사 처리량)의 근거가 된다.

**모델을 고정하는 것이 핵심이다.** 모델과 채널이 같이 바뀌면 성능 차이가
어느 쪽에서 왔는지 말할 수 없다. 이 스크립트는 그 실수를 구조적으로 막는다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import DATA_DIR, RESULTS_DIR, load_exp     # noqa: E402
from gleeg import evaluate                                    # noqa: E402
from gleeg.datasets import load_bundle                        # noqa: E402
from gleeg.preprocess.montage import SUBSETS                  # noqa: E402

DEFAULT_SUBSETS = ("all19", "frontal8", "frontal4", "forehead2")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="base")
    ap.add_argument("--npz", default=None)
    ap.add_argument("--model", default="EEGNet")
    ap.add_argument("--subsets", default=",".join(DEFAULT_SUBSETS))
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--tag-prefix", default=None)
    args = ap.parse_args()

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    bad = [s for s in subsets if s not in SUBSETS]
    if bad:
        print(f"없는 서브셋: {bad}. 가능한 값: {list(SUBSETS)}")
        return 1

    cfg = load_exp(args.exp)
    cfg["model"]["backbone"] = args.model
    if args.max_epochs:
        cfg["train"]["max_epochs"] = args.max_epochs

    npz = Path(args.npz) if args.npz else DATA_DIR / "npz" / f"{args.exp}_derivatives.npz"
    base = load_bundle(npz)
    prefix = args.tag_prefix or f"{time.strftime('%m%d-%H%M')}_sweep_{args.model}"

    print("=" * 70)
    print(f"  채널 스윕 · 모델 고정 {args.model}")
    print(f"  {base.describe()}")
    print(f"  서브셋 {subsets}")
    print("=" * 70)

    from gleeg.train.loop import cross_validate, pick_device

    device = pick_device(cfg["train"]["device"])
    curve = []
    for name in subsets:
        wanted = [c for c in SUBSETS[name] if c in base.channels]
        missing = [c for c in SUBSETS[name] if c not in base.channels]
        if missing:
            print(f"\n  [주의] {name}: 데이터에 없는 채널 제외 {missing}")
        if not wanted:
            print(f"  [건너뜀] {name}: 사용 가능한 채널이 없습니다")
            continue

        b = base.select_channels(wanted)
        print(f"\n--- {name} ({len(b.channels)}ch) {b.channels}")
        res = cross_validate(b, cfg, device)
        agg = res["subject_level"]
        tag = f"{prefix}_{name}"
        evaluate.save_results(
            {"epoch_level": res["epoch_level"], "subject_level": agg,
             "best_epochs": res.get("best_epochs")},
            {"exp": args.exp, "model": args.model, "channels": b.channels,
             "n_channels": len(b.channels), "subset": name, "npz": str(npz),
             "config": cfg, "data_meta": {k: v for k, v in b.meta.items() if k != "config"}},
            RESULTS_DIR / tag,
        )
        curve.append((name, len(b.channels), agg))

    print("\n" + "=" * 70)
    print(f"  채널 수 vs 성능 ({args.model}, 피험자 단위)")
    print("=" * 70)
    print(f"  {'서브셋':<12}{'채널':>5}{'정확도':>16}{'민감도':>16}{'QWK':>16}")
    ref = None
    for name, n, a in curve:
        acc = a.get("disease_acc", {})
        sen = a.get("screening_sensitivity", {})
        qwk = a.get("severity_qwk", {})
        ref = ref if ref is not None else acc.get("mean")
        delta = "" if ref is None else f"  ({acc.get('mean', 0) - ref:+.3f})"
        print(f"  {name:<12}{n:>5}"
              f"{acc.get('mean', float('nan')):>10.3f}±{acc.get('std', 0):.3f}"
              f"{sen.get('mean', float('nan')):>10.3f}±{sen.get('std', 0):.3f}"
              f"{qwk.get('mean', float('nan')):>10.3f}±{qwk.get('std', 0):.3f}"
              f"{delta}")
    print("\n  괄호는 전체 채널 대비 정확도 변화량. 이 값이 작으면 저채널 기기로")
    print("  옮겨도 성능이 유지된다는 뜻이고, 그게 제품 관점의 핵심 수치다.")
    print(f"\n  python -m scripts.compare  로 다른 실험과 함께 볼 수 있습니다.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
