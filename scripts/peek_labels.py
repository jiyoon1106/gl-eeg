"""ds004504 라벨 품질 분석.

participants.tsv 하나만 있으면 돌아간다 (EEG 본체 불필요).

이 스크립트가 찾아낸 것이 과제 정의를 바꿨다. 보고서의 "공개 데이터 라벨 품질
분석" 절이 이 출력에서 나온다. 데이터를 받을 때마다 먼저 돌려볼 것.

    python -m scripts.peek_labels
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.paths import DS004504_DIR, load_exp          # noqa: E402
from gleeg.labels import DS004504_GROUP_MAP, describe_bins, to_level   # noqa: E402

# 임상에서 통용되는 MMSE 구간 — 우리 데이터가 이걸 지지하는지 확인한다
CLINICAL_BANDS = [
    ("정상      (24~30)", 24, 31),
    ("경도      (20~23)", 20, 24),
    ("중등도    (10~19)", 10, 20),
    ("중증      (< 10) ", 0, 10),
]


def bar(n: int, total: int, width: int = 30) -> str:
    filled = 0 if total == 0 else round(width * n / total)
    return "█" * filled + "·" * (width - filled)


def main() -> int:
    tsv = DS004504_DIR / "participants.tsv"
    if not tsv.exists():
        print(f"participants.tsv 가 없습니다: {tsv}")
        print("먼저 받으세요:")
        print("  openneuro-py download --dataset=ds004504 "
              "--include=participants.tsv --target-dir=data/ds004504")
        return 1

    df = pd.read_csv(tsv, sep="\t")
    df["disease"] = df["Group"].map(DS004504_GROUP_MAP)

    print("=" * 66)
    print(f"ds004504 라벨 분석 — 피험자 {len(df)}명")
    print("=" * 66)

    # ---------- 군 구성 ----------
    print("\n[군 구성]")
    for code, grp in df.groupby("disease"):
        print(f"  {code:<4} {len(grp):>3}명   {bar(len(grp), len(df))}")

    # ---------- 군별 MMSE ----------
    print("\n[군별 MMSE]")
    stat = df.groupby("disease")["MMSE"].agg(["count", "mean", "std", "min", "max"])
    print(stat.to_string(float_format=lambda v: f"{v:6.2f}"))

    # ⚠ 대조군 표준편차 확인 — 실측값인지 판정하는 핵심
    print("\n[라벨 품질 점검]")
    flags = []
    for code, grp in df.groupby("disease"):
        sd = grp["MMSE"].std()
        if sd == 0:
            flags.append(
                f"  (!) {code} 군 {len(grp)}명 전원 MMSE={grp['MMSE'].iloc[0]:g}, std=0\n"
                f"      → 실측값이 아니라 만점 기록으로 보인다. "
                f"MMSE 회귀는 '{grp['MMSE'].iloc[0]:g}이냐 아니냐' 이진 분류로 퇴화한다."
            )
    # 값이 비어 있는 구간
    present = set(df["MMSE"])
    lo, hi = int(df["MMSE"].min()), int(df["MMSE"].max())
    gaps = [v for v in range(lo, hi + 1) if v not in present]
    if gaps:
        runs, cur = [], [gaps[0]]
        for v in gaps[1:]:
            (cur.append(v) if v == cur[-1] + 1 else (runs.append(cur), cur := [v]))
        runs.append(cur)
        long_runs = [r for r in runs if len(r) >= 2]
        if long_runs:
            flags.append(
                "  (!) MMSE 값이 비어 있는 구간: "
                + ", ".join(f"{r[0]}~{r[-1]}" for r in long_runs)
                + "\n      → 그 경계로 단계를 나눌 수 없다."
            )
    print("\n".join(flags) if flags else "  특이사항 없음")

    # ---------- 임상 구간으로 나눠보면 ----------
    print("\n[임상 통용 4구간으로 나눌 경우]")
    too_small = []
    for label, lo_, hi_ in CLINICAL_BANDS:
        n = int(((df["MMSE"] >= lo_) & (df["MMSE"] < hi_)).sum())
        mark = "  <-- 너무 적다" if 0 < n < 10 else ("  <-- 없음" if n == 0 else "")
        print(f"  {label}  {n:>3}명   {bar(n, len(df))}{mark}")
        if 0 < n < 10:
            too_small.append((label.strip(), n))
    if too_small:
        print()
        for label, n in too_small:
            print(
                f"  (!) '{label}' {n}명 → 피험자 단위 5-fold 시 fold 당 "
                f"{n/5:.1f}명. 학습·평가 불가."
            )

    # ---------- 채택한 구간 ----------
    cfg = load_exp("base")
    bins = cfg["severity_bins"]
    print(f"\n[채택한 구간 — config/exp/base.yaml]")
    print(f"  {describe_bins(bins)}")
    df["level"] = [to_level(s, "MMSE", bins) for s in df["MMSE"]]
    for lv, name in enumerate(bins["labels"]):
        n = int((df["level"] == lv).sum())
        print(f"  {lv}  {name:<20} {n:>3}명   {bar(n, len(df))}")

    # ---------- 질환 x 중증도 교차표 ----------
    print("\n[질환 × 중증도 교차표]")
    ct = pd.crosstab(df["disease"], df["level"])
    ct.columns = [bins["labels"][c] for c in ct.columns]
    print(ct.to_string())
    print(
        "\n  → 두 축이 거의 같은 정보를 담고 있으면 멀티태스크 헤드의 이득이 작다.\n"
        "     중증도는 보조 과제로 두고 질환 분류를 주 과제로 삼는 근거."
    )

    # ---------- MMSE 히스토그램 ----------
    print("\n[MMSE 값별 인원]")
    vc = df["MMSE"].value_counts().sort_index()
    for score, n in vc.items():
        print(f"  {score:>2}  {n:>2}명  {bar(int(n), int(vc.max()), 24)}")

    # ---------- 나이 ----------
    print("\n[군별 나이] 연령 효과와 질환 효과를 분리해야 하므로 확인")
    print(df.groupby("disease")["Age"].agg(["mean", "std", "min", "max"])
          .to_string(float_format=lambda v: f"{v:6.2f}"))

    print("\n" + "=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
