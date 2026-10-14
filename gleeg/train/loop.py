"""학습 루프 + 교차검증.

세 가지를 코드로 강제한다.

1. 피험자 단위 분할 — evaluate.make_splits() 가 담당
2. train 안에서 다시 val 을 떼어 early stopping — **test 로 조기종료를 결정하면
   그 성능은 못 쓴다.** val 도 피험자 단위로 나눈다.
3. epoch 단위와 **피험자 단위** 지표를 둘 다 보고 — 실제 표본은 88명이다
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .. import evaluate
from ..datasets import Bundle, make_loaders
from ..labels import n_levels
from ..models.backbones import build_backbone, count_params
from ..models.heads import MultiTaskModel
from ..models.losses import coral_predict, multitask_loss


# ----------------------------------------------------------------------
def pick_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def backbone_kwargs(cfg: dict, name: str) -> dict:
    """그 백본에 줄 인자.

    `backbone_kwargs` 는 주력 백본(ShallowFBCSPNet) 기준으로 적혀 있다.
    백본마다 인자 이름이 다르므로, `backbone_kwargs_by_model` 에 해당 모델
    항목이 있으면 **그것으로 대체**한다(병합이 아니다 — 섞으면 어떤 설정으로
    돌았는지 추적이 안 된다). 없으면 build_backbone 이 안 받는 인자를
    걸러낸다.
    """
    m = cfg["model"]
    per = (m.get("backbone_kwargs_by_model") or {}).get(name)
    return dict(per if per is not None else (m.get("backbone_kwargs") or {}))


def build_model(
    cfg: dict,
    n_chans: int,
    n_times: int,
    n_disease: int,
    K: int | None,
    sfreq: float | None = None,
):
    m = cfg["model"]
    backbone = build_backbone(
        m["backbone"],
        n_chans=n_chans,
        n_times=n_times,
        embed_dim=m["embed_dim"],
        sfreq=sfreq,
        **backbone_kwargs(cfg, m["backbone"]),
    )
    return MultiTaskModel(backbone, m["embed_dim"], n_disease, K)


def inner_val_split(
    subject_ids: np.ndarray, strat: np.ndarray, train_idx: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """train 안에서 val 을 떼어낸다 — 피험자 단위로.

    early stopping 을 test 로 결정하면 test 성능이 낙관적으로 부풀려진다.
    """
    sub_tr = subject_ids[train_idx]
    inner = evaluate.make_splits(sub_tr, strat[train_idx], n_splits=5, seed=seed)
    tr_in, va_in = inner[0]
    return train_idx[tr_in], train_idx[va_in]


# ----------------------------------------------------------------------
@dataclass
class FoldResult:
    fold: int
    best_epoch: int
    metrics_epoch: dict
    metrics_subject: dict
    history: list[dict] = field(default_factory=list)


@torch.no_grad()
def _predict(model, loader, device) -> dict:
    model.eval()
    P, yd, yl, lp, idx = [], [], [], [], []
    for b in loader:
        out = model(b["x"].to(device))
        P.append(F.softmax(out.disease_logits, dim=1).cpu().numpy())
        yd.append(b["y_disease"].numpy())
        yl.append(b["y_level"].numpy())
        if out.severity_logits is not None:
            lp.append(coral_predict(out.severity_logits).cpu().numpy())
        idx.append(b["idx"].numpy())
    return {
        "probs": np.concatenate(P),
        "y_disease": np.concatenate(yd),
        "y_level": np.concatenate(yl),
        "level_pred": np.concatenate(lp) if lp else None,
        "idx": np.concatenate(idx),
    }


def train_fold(
    bundle: Bundle,
    cfg: dict,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: int,
    device: torch.device,
    verbose: bool = True,
) -> FoldResult:
    t = cfg["train"]
    classes = bundle.meta["disease_classes"]
    bins = bundle.meta["severity_bins"]
    K = n_levels(bins)
    seed = cfg["seed"] + fold

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ① train 안에서 val 분리 (피험자 단위)
    tr, va = inner_val_split(bundle.subject_id, bundle.y_disease, train_idx, seed)
    evaluate.assert_no_leakage(bundle.subject_id, tr, va)
    evaluate.assert_no_leakage(bundle.subject_id, tr, test_idx)

    dl_tr, _ = make_loaders(bundle, tr, tr[:1], t["batch_size"], num_workers=0, seed=seed)
    _, dl_va = make_loaders(bundle, va[:1], va, t["batch_size"], num_workers=0, seed=seed)
    _, dl_te = make_loaders(bundle, test_idx, test_idx, t["batch_size"], num_workers=0, seed=seed)

    model = build_model(
        cfg, len(bundle.channels), bundle.X.shape[2], len(classes), K, bundle.sfreq
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["max_epochs"])

    if verbose:
        n_s = lambda i: len(np.unique(bundle.subject_id[i]))  # noqa: E731
        print(
            f"  fold {fold}: train {n_s(tr)}명 / val {n_s(va)}명 / test {n_s(test_idx)}명 "
            f"· 파라미터 {count_params(model):,}"
        )

    best_f1, best_epoch, best_state, patience = -1.0, -1, None, 0
    history: list[dict] = []
    t0 = time.time()

    for ep in range(t["max_epochs"]):
        model.train()
        tot = 0.0
        for b in dl_tr:
            opt.zero_grad()
            out = model(b["x"].to(device))
            loss, _ = multitask_loss(
                out,
                b["y_disease"].to(device),
                b["y_level"].to(device),
                K,
                cfg["loss"]["severity_weight"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
        sched.step()

        # val 로 조기종료 판단 — 피험자 단위 F1 기준
        pv = _predict(model, dl_va, device)
        pooled = evaluate.pool_by_subject(
            bundle.subject_id[pv["idx"]], pv["y_disease"], pv["probs"]
        )
        from sklearn.metrics import f1_score

        f1 = f1_score(pooled["y_true"], pooled["y_pred"], average="macro", zero_division=0)
        history.append({"epoch": ep, "train_loss": tot / max(1, len(dl_tr)), "val_f1": float(f1)})

        if f1 > best_f1 + 1e-4:
            best_f1, best_epoch, patience = f1, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= t["early_stopping_patience"]:
                break

        if verbose and (ep % 10 == 0 or ep == t["max_epochs"] - 1):
            print(
                f"    ep {ep:>3} loss {history[-1]['train_loss']:.4f} "
                f"val_f1(subject) {f1:.4f}  best {best_f1:.4f}@{best_epoch}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    # ② test 평가 — epoch 단위와 피험자 단위 둘 다
    pt = _predict(model, dl_te, device)
    yp = pt["probs"].argmax(axis=1)
    m_epoch = evaluate.compute_metrics(
        pt["y_disease"], yp, classes, pt["y_level"], pt["level_pred"], n_levels=K
    )
    pooled = evaluate.pool_by_subject(
        bundle.subject_id[pt["idx"]], pt["y_disease"], pt["probs"],
        pt["y_level"], pt["level_pred"],
    )
    m_subj = evaluate.compute_metrics(
        pooled["y_true"], pooled["y_pred"], classes,
        pooled.get("level_true"), pooled.get("level_pred"), n_levels=K,
    )

    if verbose:
        print(
            f"    -> epoch단위 acc {m_epoch['disease_acc']:.4f} | "
            f"피험자단위 acc {m_subj['disease_acc']:.4f} "
            f"F1 {m_subj['disease_f1_macro']:.4f} "
            f"민감도 {m_subj.get('screening_sensitivity', float('nan')):.4f} "
            f"({time.time()-t0:.0f}s, best ep {best_epoch})"
        )

    return FoldResult(fold, best_epoch, m_epoch, m_subj, history)


# ----------------------------------------------------------------------
def cross_validate(bundle: Bundle, cfg: dict, device: torch.device | None = None) -> dict:
    device = device or pick_device(cfg["train"]["device"])
    splits = evaluate.make_splits(
        bundle.subject_id, bundle.y_disease, cfg["eval"]["n_splits"], cfg["seed"]
    )
    print(f"\n  장치 {device} · {len(splits)}-fold (피험자 단위)")

    results = [
        train_fold(bundle, cfg, tr, te, i, device) for i, (tr, te) in enumerate(splits)
    ]
    return {
        "epoch_level": evaluate.aggregate_folds([r.metrics_epoch for r in results]),
        "subject_level": evaluate.aggregate_folds([r.metrics_subject for r in results]),
        "best_epochs": [r.best_epoch for r in results],
        "history": [r.history for r in results],
    }


# ----------------------------------------------------------------------
def cross_validate_baseline(bundle: Bundle, cfg: dict) -> dict:
    """고전 베이스라인 — 밴드파워 + 부스팅 트리. 비교 기준점.

    신경망과 **같은 지표**가 나와야 비교가 성립하므로 중증도도 같이 낸다.
    다만 순서형 헤드 대신 원점수 회귀 후 구간화로 처리한다 — 고전 방식에서
    순서 정보를 다루는 통상적인 방법이다.
    """
    from ..models.baseline import bandpower_features, make_classifier, make_regressor

    classes = bundle.meta["disease_classes"]
    bins = bundle.meta["severity_bins"]
    K = n_levels(bins)
    edges = bins["edges"]

    print("\n  밴드파워 특징 추출 중...")
    feats = bandpower_features(bundle.X.astype(np.float32), bundle.sfreq)
    print(f"  특징 {feats.shape}")

    splits = evaluate.make_splits(
        bundle.subject_id, bundle.y_disease, cfg["eval"]["n_splits"], cfg["seed"]
    )
    has_sev = np.isfinite(bundle.severity_score) & (bundle.y_level >= 0)

    fe, fs = [], []
    for i, (tr, te) in enumerate(splits):
        evaluate.assert_no_leakage(bundle.subject_id, tr, te)
        clf = make_classifier(cfg["seed"] + i).fit(feats[tr], bundle.y_disease[tr])
        probs = clf.predict_proba(feats[te])
        yp = probs.argmax(axis=1)

        # 중증도: 원점수 회귀 -> 같은 edges 로 구간화
        lp = np.full(len(te), evaluate.IGNORE_LEVEL, dtype=np.int64)
        tr_s, te_s = tr[has_sev[tr]], te[has_sev[te]]
        if len(tr_s) and len(te_s):
            reg = make_regressor(cfg["seed"] + i).fit(feats[tr_s], bundle.severity_score[tr_s])
            pred_score = reg.predict(feats[te_s])
            lp[has_sev[te]] = np.digitize(pred_score, edges)

        fe.append(evaluate.compute_metrics(
            bundle.y_disease[te], yp, classes, bundle.y_level[te], lp, n_levels=K
        ))
        pooled = evaluate.pool_by_subject(
            bundle.subject_id[te], bundle.y_disease[te], probs, bundle.y_level[te], lp
        )
        fs.append(evaluate.compute_metrics(
            pooled["y_true"], pooled["y_pred"], classes,
            pooled.get("level_true"), pooled.get("level_pred"), n_levels=K,
        ))
        print(
            f"  fold {i}: epoch단위 {fe[-1]['disease_acc']:.4f} | "
            f"피험자단위 {fs[-1]['disease_acc']:.4f} "
            f"(중증도 MAE {fs[-1].get('severity_mae', float('nan')):.3f})",
            flush=True,
        )

    return {
        "epoch_level": evaluate.aggregate_folds(fe),
        "subject_level": evaluate.aggregate_folds(fs),
        "n_levels": K,
    }
