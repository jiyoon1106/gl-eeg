"""results/ 아래 실험들을 한 표로.

    python -m scripts.compare
    python -m scripts.compare --sort severity_qwk --level epoch
    python -m scripts.compare --csv results/summary.csv

왜 필요한가
-----------
실험이 쌓이면 로그를 거슬러 올라가 숫자를 찾는 일이 생긴다. 그러다 **설정이
다른 실험을 같은 표에 놓고 비교하는 사고**가 난다. 이 스크립트는 단계 수와
채널 수를 같은 줄에 찍어서, 비교 가능한 것끼리만 비교하게 만든다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import RESULTS_DIR          # noqa: E402

# (키, 표 머리글, 높을수록 좋은가)
COLUMNS = [
    ("disease_acc", "정확도", True),
    ("disease_f1_macro", "F1", True),
    ("screening_sensitivity", "민감도", True),
    ("screening_specificity", "특이도", True),
    ("severity_mae", "중증도MAE", False),
    ("severity_qwk", "QWK", True),
]


def load_runs(root: Path) -> list[dict]:
    runs = []
    for d in sorted(root.iterdir()):
        mj, cj = d / "metrics.json", d / "config.json"
        if not mj.exists():
            continue
        m = json.loads(mj.read_text(encoding="utf-8"))
        c = json.loads(cj.read_text(encoding="utf-8")) if cj.exists() else {}
        runs.append({"tag": d.name, "metrics": m, "cfg": c})
    return runs


def describe_setup(cfg: dict) -> tuple[str, str, str]:
    """모델 · 채널 · 중증도 단계 — 비교 가능성을 판단하는 세 축."""
    model = cfg.get("model", "?")
    nch = cfg.get("n_channels")
    ch = f"{nch}ch" if nch else "?"
    bins = (cfg.get("data_meta") or {}).get("severity_bins") or {}
    lv = f"{len(bins.get('labels', []))}단계" if bins else "?"
    return str(model), ch, lv


def fmt(stat: dict | None) -> str:
    if not stat:
        return "—"
    return f"{stat['mean']:.3f}±{stat['std']:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="기본: results/")
    ap.add_argument("--level", default="subject", choices=["subject", "epoch"],
                    help="집계 단위 (기본 subject — 이쪽이 정직한 수치)")
    ap.add_argument("--sort", default="disease_acc")
    ap.add_argument("--csv", default=None, help="CSV 로도 저장")
    args = ap.parse_args()

    root = Path(args.dir) if args.dir else RESULTS_DIR
    if not root.exists():
        print(f"{root} 가 없습니다. 먼저 학습을 돌리세요.")
        return 1

    runs = load_runs(root)
    if not runs:
        print(f"{root} 에 결과가 없습니다.")
        return 1

    key = f"{args.level}_level"
    rows = []
    for r in runs:
        agg = r["metrics"].get(key) or {}
        model, ch, lv = describe_setup(r["cfg"])
        rows.append({
            "tag": r["tag"], "model": model, "ch": ch, "levels": lv,
            "folds": agg.get("n_folds", "?"),
            "best_epochs": r["metrics"].get("best_epochs"),
            **{k: agg.get(k) for k, _, _ in COLUMNS},
        })

    hi = dict((k, h) for k, h, _ in COLUMNS)
    rows.sort(key=lambda r: (r[args.sort] or {}).get("mean", -1e9), reverse=True)

    w_tag = max(len(r["tag"]) for r in rows) + 1
    w_mod = max(len(r["model"]) for r in rows) + 1
    head = (f"{'실험':<{w_tag}} {'모델':<{w_mod}} {'채널':>5} {'단계':>5}  "
            + "  ".join(f"{hi[k]:>11}" for k, _, _ in COLUMNS))

    print(f"\n[ {args.level} 단위 집계 ]  {root}")
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['tag']:<{w_tag}} {r['model']:<{w_mod}} {r['ch']:>5} {r['levels']:>5}  "
              + "  ".join(f"{fmt(r[k]):>11}" for k, _, _ in COLUMNS))
    print("=" * len(head))

    # 비교 가능성 경고 — 설정이 섞여 있으면 표를 그대로 읽으면 안 된다
    levels = {r["levels"] for r in rows}
    if len(levels) > 1:
        print(f"  (!) 중증도 단계 수가 섞여 있습니다 {sorted(levels)} — "
              "MAE/QWK 를 서로 비교하지 마세요.")
    if len({r["ch"] for r in rows}) > 1 and len({r["model"] for r in rows}) > 1:
        print("  (!) 모델과 채널이 동시에 다릅니다. 채널 효과를 보려면 "
              "모델을 고정한 줄끼리만 비교하세요.")
    print("  중증도 MAE 는 낮을수록 좋고, 나머지는 높을수록 좋습니다.\n")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = ["tag", "model", "ch", "levels", "folds"]
        with out.open("w", encoding="utf-8") as f:
            f.write(",".join(cols + [f"{k}_mean,{k}_std" for k, _, _ in COLUMNS]) + "\n")
            for r in rows:
                vals = [str(r[c]) for c in cols]
                for k, _, _ in COLUMNS:
                    s = r[k]
                    vals += [f"{s['mean']:.4f}", f"{s['std']:.4f}"] if s else ["", ""]
                f.write(",".join(vals) + "\n")
        print(f"  저장: {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
