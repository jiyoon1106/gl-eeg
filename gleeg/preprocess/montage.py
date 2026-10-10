"""채널 선택 — 런타임 교집합.

채널 개수를 코드에 박지 않는다. 어떤 데이터셋 조합을 쓰느냐에 따라 공통 채널이
달라지기 때문이다.

    ds004504만            → 19채널
    ds004504 + ds004584   → 19채널 (64ch 에서 19개만 선택)
    ds004504 + 웨어러블    → 2채널 (Fp1, Fp2 뿐)

그리고 같은 함수가 저채널 실험에도 쓰인다. 교집합에서 전두엽 서브셋을 한 번 더
고르면 끝이다.

주의
----
공간 필터 커널이 (C, 1) 이라 **채널 순서에 가중치가 매핑**된다. 학습 때 Fp1 이
0번이었는데 추론 때 1번이면 완전히 다른 결과가 나온다. set 은 순서를 보장하지
않으므로 교집합 뒤에 반드시 고정 순서로 정렬해야 하고, 빠뜨리면 에러 없이
성능만 이상하게 나온다.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

# 국제 10-20 표준 순서 (앞 → 뒤, 좌 → 우). 이 순서가 채널 축의 정답이다.
TEN_TWENTY_ORDER: tuple[str, ...] = (
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2",
)

# 자주 쓰는 서브셋 — 저채널 실험용
SUBSETS: dict[str, tuple[str, ...]] = {
    "all19": TEN_TWENTY_ORDER,
    "frontal8": ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "Cz"),
    "frontal4": ("Fp1", "Fp2", "F3", "F4"),
    "forehead2": ("Fp1", "Fp2"),
}

# 데이터셋마다 표기가 다른 채널 이름을 표준으로 맞춘다.
_ALIASES = {
    "T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6",
    "FP1": "Fp1", "FP2": "Fp2",
}


def canonical(name: str) -> str:
    """채널 이름을 10-20 표준 표기로. 'EEG Fp1-LE' 같은 접두·접미도 벗긴다."""
    n = name.strip()
    if n.upper().startswith("EEG "):
        n = n[4:].strip()
    if "-" in n:                      # 'Fp1-LE' → 'Fp1'
        n = n.split("-", 1)[0].strip()
    return _ALIASES.get(n.upper(), _ALIASES.get(n, n))


def sort_canonical(channels: Iterable[str]) -> list[str]:
    """10-20 표준 순서로 정렬. 표준에 없는 채널은 뒤에 알파벳 순으로 붙인다."""
    chans = {canonical(c) for c in channels}
    known = [c for c in TEN_TWENTY_ORDER if c in chans]
    unknown = sorted(chans - set(TEN_TWENTY_ORDER))
    return known + unknown


def common_channels(channel_lists: Sequence[Sequence[str]]) -> list[str]:
    """여러 데이터셋의 채널 목록에서 교집합을 구해 고정 순서로 반환.

    Parameters
    ----------
    channel_lists : 데이터셋별 채널 목록의 목록

    Examples
    --------
    >>> common_channels([["Fp1", "Fp2", "Cz"], ["Fp2", "Fp1", "Pz"]])
    ['Fp1', 'Fp2']
    """
    if not channel_lists:
        raise ValueError("채널 목록이 비어 있습니다")

    sets = [{canonical(c) for c in lst} for lst in channel_lists]
    inter = set.intersection(*sets)
    if not inter:
        raise ValueError(
            "공통 채널이 없습니다. 데이터셋 조합을 확인하세요: "
            + " / ".join(str(sorted(s)) for s in sets)
        )
    return sort_canonical(inter)


def resolve_channels(
    channel_lists: Sequence[Sequence[str]],
    subset: Sequence[str] | str | None = None,
) -> list[str]:
    """교집합을 구한 뒤, subset 이 주어지면 한 번 더 걸러낸다.

    저채널 실험은 subset="frontal4" 처럼 호출하면 된다.
    """
    chans = common_channels(channel_lists)

    if subset is None:
        return chans

    wanted = SUBSETS[subset] if isinstance(subset, str) else tuple(subset)
    wanted = [canonical(c) for c in wanted]

    picked = [c for c in chans if c in wanted]
    missing = [c for c in wanted if c not in chans]
    if not picked:
        raise ValueError(f"subset {subset!r} 의 채널이 교집합에 하나도 없습니다 (교집합: {chans})")
    if missing:
        import warnings

        warnings.warn(
            f"subset {subset!r} 중 {missing} 는 교집합에 없어 제외합니다 "
            f"(사용: {picked})",
            stacklevel=2,
        )
    return sort_canonical(picked)


def channel_indices(available: Sequence[str], wanted: Sequence[str]) -> list[int]:
    """available 에서 wanted 채널을 뽑아낼 인덱스. 순서는 wanted 를 따른다."""
    lookup = {canonical(c): i for i, c in enumerate(available)}
    try:
        return [lookup[canonical(c)] for c in wanted]
    except KeyError as e:
        raise ValueError(f"채널 {e.args[0]!r} 가 없습니다 (가능: {list(lookup)})") from e
