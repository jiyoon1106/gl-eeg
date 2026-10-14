# gl-eeg

공개 생체신호 데이터로 **뇌질환 종류와 진행 정도**를 판별하는 알고리즘.
(주)지엘 산학협력 공모전 · 경북대 5인 팀 · 최종발표 2026.11.26

---

## 설계 원칙 세 줄

1. **한 번 고정할 것과 계속 추가할 것을 분리한다.**
   스키마 · 출력 규격 · 평가 함수는 고정, 어댑터 · 백본 · 실험 스크립트는 추가.
2. **중증도 "단계"는 데이터에 저장하지 않는다.**
   원점수(MMSE 등)만 저장하고, 단계 구분은 `config/exp/*.yaml`에서 만든다.
3. **채널 개수를 코드에 박지 않는다.**
   데이터셋 조합의 공통 채널을 실행 시점에 계산한다(런타임 교집합).

---

## 구조

```
config/
  paths.py              환경변수 기반 경로 (절대경로 하드코딩 금지)
  exp/base.yaml         실험 설정 — 중증도 구간, 채널 서브셋, 모델 인자
gleeg/
  schema.py        ★①  표준 샘플 정의 (EEGSample)
  labels.py             원점수 → 단계 변환 (유일한 지점)
  adapters/             데이터셋별 변환 — 새 데이터셋은 여기에 파일 1개 추가
  preprocess/
    montage.py          런타임 교집합 · 저채널 서브셋
  models/
    backbones.py        Braindecode 래핑 → 임베딩 추출기
    heads.py       ★②  출력 규격 (ModelOutput) + 멀티태스크 · CORAL 헤드
    losses.py           CORAL 손실 · 멀티태스크 가중합 · 증류 손실
    baseline.py         밴드파워 + 부스팅 트리 (비교 기준점, 신경망 아님)
  train/
    loop.py             학습 루프 · 교차검증 (내부 val 분리 · 이중 집계)
  evaluate.py      ★③  피험자 단위 CV + MAE/QWK/민감도
scripts/
  peek_labels.py        데이터 라벨 품질 분석 (받을 때마다 먼저 실행)
  build_dataset.py      원본 → data.npz (어댑터 → 전처리 → epoch)
  run_train.py          학습 실행 → results/<태그>/metrics.json
tests/
  test_smoke.py         실제 데이터 없이 인터페이스 3종 검증
  test_train.py         실제 데이터 없이 학습 루프 검증 (합성 신호)
  test_pipeline.py      가짜 ds004504 로 원본→npz→학습 경로 전체 검증
```

**★ 표시 세 개가 팀 작업이 붙는 접점이다.** 이 세 파일을 바꾸면 팀 전체 코드가
영향을 받으므로, 바꿀 때는 반드시 공유한다.

---

## 시작하기

### 환경 (처음 1회)

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # csh: source .venv/bin/activate.csh
pip install -r requirements.txt
```

GPU가 있고 드라이버가 CUDA 12.x 까지만 지원하면:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 경로 설정

```bash
export GL_DATA_DIR=/path/to/EEG/data        # csh: setenv GL_DATA_DIR /path/...
python config/paths.py                       # 확인
```

### 동작 확인 (데이터 불필요)

```bash
python -m tests.test_smoke
```

인터페이스 3종이 규격대로 동작하는지 더미 텐서로 검증한다.
**데이터 다운로드를 기다리지 않고 작업을 시작할 수 있다.**

### 데이터 받기

```bash
# 먼저 라벨만 (몇 KB)
openneuro-py download --dataset=ds004504 \
  --include=participants.tsv --target-dir=data/ds004504
python -m scripts.peek_labels

# 전체
openneuro-py download --dataset=ds004504 --target-dir=data/ds004504
```

### 학습

```bash
python -m scripts.build_dataset --exp base --source derivatives   # 1회 (약 1분)

python -m scripts.run_train --model baseline                      # 비교 기준점
python -m scripts.run_train --model ShallowFBCSPNet               # 주력
python -m scripts.run_train --model EEGNet --channels frontal4    # 저채널
```

결과는 `results/<태그>/metrics.json` + `config.json` 에 쌓인다. 단계 수나
채널 수가 바뀌면 지표를 직접 비교할 수 없으므로 설정을 함께 남긴다.

**지표는 두 단위로 나온다.**

| | 의미 |
|---|---|
| epoch 단위 | 4초 조각 하나하나를 맞췄는지. 샘플이 8,776개로 보인다 |
| **피험자 단위** | 사람별 확률 평균 후 판정. **실제 표본은 88명** — 보고·발표는 이 수치로 |

조기종료(early stopping)는 train 안에서 다시 떼어낸 val 로만 판단한다
(`inner_val_split`). test 로 조기종료를 결정하면 그 test 성능은 쓸 수 없다.

---

## 데이터셋 라벨 품질 분석 (ds004504)

`scripts/peek_labels.py` 가 찾아낸 것이 **과제 정의를 바꿨다.**

| 발견 | 의미 |
|---|---|
| 대조군 29명 전원 MMSE=30, **std=0** | 실측값이 아니라 만점 기록. MMSE 회귀가 "30이냐 아니냐" 이진 분류로 퇴화한다 |
| MMSE 28·29 구간에 **0명** | 정상/경계 사이가 비어 단계를 나눌 수 없다 |
| 임상 기준 중증(<10)이 **3명** | 피험자 단위 5-fold 시 fold 당 0.6명 → 학습·평가 불가 |
| A군은 전원 ≤23, C군은 전원 30 | 질환 분류와 중증도가 거의 같은 정보 → 멀티태스크 이득 작음 |

**결론**: 5단계 중증도는 이 데이터가 지지하지 않는다.

- **주 과제** = 질환 3분류 (AD 36 / FTD 23 / CN 29 — 균형 양호)
- **보조 과제** = 중증도 3단계 (20 / 39 / 29)
- **5단계 방법론**은 UCI #189(UPDRS, 실측 연속값)에서 검증

MCI를 포함한 데이터셋을 추가하면 `severity_bins` 만 고쳐 5단계로 확장한다.
전처리 재실행도, 코드 수정도 필요 없다.

---

## 모델

```
입력 (C, T)
  ↓
ShallowFBCSPNet          conv → spatial → square → pool → log
  ↓                      = 밴드파워 계산. 치매 지표(서파 증가·알파 감소)와 정렬
임베딩 z (64)
  ├→ Linear(64→3)  softmax   질환
  └→ CoralHead(64, K)        중증도 (순서형, 출력 K-1개)

L = CE + w · CORAL
```

| 모델 | 역할 | 담당 |
|---|---|---|
| `ShallowFBCSPNet` | 주력 백본 | A |
| `EEGNet` | 저채널 스윕 · teacher-student 증류 | D |
| `SincShallowNet` | 학습된 주파수 대역 해석 | D/E |
| 밴드파워 + 부스팅 트리 | 비교 기준점 (제출 모델 아님) | 공통 |

백본 이름은 Braindecode 버전에 따라 다르다(`EEGNet` ↔ `EEGNetv4`).
`build_backbone()` 이 설치된 쪽을 실행 시점에 찾으므로 둘 다 받는다.

**백본마다 인자 이름이 다르다.** ShallowFBCSPNet 의 `filter_time_length` 는
EEGNet 에서 `kernel_length` 다. 모델별 인자는 `backbone_kwargs_by_model` 에
적고, 거기 없으면 `build_backbone()` 이 그 백본이 받지 않는 인자를 걸러낸다
(버린 인자는 로그에 찍는다 — 오타가 조용히 무시되면 안 되므로).
`sfreq` 는 Hz 단위로 필터를 만드는 백본(SincShallowNet · FBCNet)에만 전달된다.

---

## ⚠️ 반드시 지킬 것

**평가는 `gleeg.evaluate` 로만 한다.**
4초 epoch으로 자르면 88명이 수천 샘플이 된다. 같은 사람의 epoch이 train과 test
양쪽에 들어가면 **정확도가 허위로 급등한다.** 이 분야 논문의 최빈 오류다.
`make_splits()` 가 `subject_id` 를 groups로 받아 이를 강제하고,
`assert_no_leakage()` 를 fold마다 호출해 이중 확인한다.

**데이터는 커밋하지 않는다.** `.gitignore` 에 `data/`, `*.npz`, `*.set` 등이
들어 있다. 의료 데이터는 재배포 제약이 있고, 신청형 데이터(CAUEEG 등)는
재배포가 DUA 위반이다.

**노트북에 로직을 쓰지 않는다.** 로직은 `.py`, 노트북은 호출만.

---

## 라이선스 · 출처

- 모델 구현: [Braindecode](https://github.com/braindecode/braindecode) (BSD-3-Clause, 일부 구성요소 CC BY-NC)
- 전처리: [MNE-Python](https://mne.tools) (BSD-3-Clause)
- 데이터: [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504) — 인용 조건 확인 필요

⚠️ Braindecode 일부 구성요소가 **CC BY-NC(비상업적)** 라이선스다. 학생 공모전
범위에서는 문제없으나 상업화 시 검토가 필요하다.
