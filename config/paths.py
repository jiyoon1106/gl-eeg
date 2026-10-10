"""경로 설정 — 코드에 절대경로를 박지 않기 위한 단일 창구.

환경변수로 덮어쓸 수 있으므로 서버·Colab·로컬에서 같은 코드가 돈다.

    # csh (연구실 서버)
    setenv GL_DATA_DIR /home/KNUEEhdd1/jjyoon1106/EEG/data

    # bash (Colab 등)
    export GL_DATA_DIR=/content/data
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_path(key: str, default: Path) -> Path:
    raw = os.environ.get(key)
    return Path(raw).expanduser() if raw else default


DATA_DIR = _env_path("GL_DATA_DIR", REPO_ROOT / "data")
RESULTS_DIR = _env_path("GL_RESULTS_DIR", REPO_ROOT / "results")
CONFIG_DIR = REPO_ROOT / "config"
EXP_DIR = CONFIG_DIR / "exp"

# 데이터셋별 원본 위치
DS004504_DIR = DATA_DIR / "ds004504"


def ensure_dirs() -> None:
    """출력 디렉토리만 생성한다. 데이터 디렉토리는 만들지 않는다."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_exp(name: str = "base") -> dict:
    """config/exp/<name>.yaml 을 읽어 dict로 반환."""
    import yaml

    path = EXP_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"실험 설정이 없습니다: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    print(f"REPO_ROOT   = {REPO_ROOT}")
    print(f"DATA_DIR    = {DATA_DIR}   (존재: {DATA_DIR.exists()})")
    print(f"RESULTS_DIR = {RESULTS_DIR}")
    print(f"DS004504    = {DS004504_DIR}  (존재: {DS004504_DIR.exists()})")
