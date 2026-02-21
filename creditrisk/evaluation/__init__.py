"""
creditrisk.evaluation
=====================
模型评估：KS、AUC、PSI、Lift、通过率-坏率曲线。
所有函数均返回 (value, detail_df) 便于进一步分析。
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve


# ── KS ─────────────────────────────────────────────────────────────────────

def ks_stat(y_true: np.ndarray,
            y_score: np.ndarray) -> Tuple[float, float]:
    """
    计算 KS 统计量及对应阈值。

    Returns
    -------
    ks : float
    threshold : float   对应最大 KS 的分数阈值

    Examples
    --------
    >>> ks, thr = ks_stat(y_true, y_score)
    >>> print(f"KS={ks:.4f}  threshold={thr:.4f}")
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    diff = tpr - fpr
    idx  = np.argmax(diff)
    return float(diff[idx]), float(thresholds[idx])


def ks_table(y_true: np.ndarray,
             y_score: np.ndarray,
             n_deciles: int = 10) -> pd.DataFrame:
    """
    按十分位生成 KS 明细表（评分卡标配输出）。

    Returns
    -------
    DataFrame 包含：decile, score_range, count, bad, good,
                    bad_rate, cum_bad_pct, cum_good_pct, ks
    """
    df = pd.DataFrame({"score": y_score, "label": y_true})
    # 从高分到低分排列（高分=高风险）
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["decile"] = (df.index // (len(df) / n_deciles)).astype(int) + 1
    df["decile"] = df["decile"].clip(upper=n_deciles)

    total_bad  = df["label"].sum()
    total_good = len(df) - total_bad

    tbl = (df.groupby("decile")
             .agg(score_min=("score","min"), score_max=("score","max"),
                  count=("label","count"), bad=("label","sum"))
             .reset_index())
    tbl["good"]          = tbl["count"] - tbl["bad"]
    tbl["bad_rate"]      = tbl["bad"] / tbl["count"]
    tbl["cum_bad_pct"]   = tbl["bad"].cumsum() / total_bad
    tbl["cum_good_pct"]  = tbl["good"].cumsum() / total_good
    tbl["ks"]            = (tbl["cum_bad_pct"] - tbl["cum_good_pct"]).abs()
    return tbl


# ── PSI ────────────────────────────────────────────────────────────────────

def psi(expected: np.ndarray,
        actual: np.ndarray,
        bins: int = 10,
        eps: float = 1e-6) -> Tuple[float, pd.DataFrame]:
    """
    计算 PSI（群体稳定性指标）。

    Parameters
    ----------
    expected : 基准期分数（训练集）
    actual   : 监控期分数（线上实际）

    Returns
    -------
    psi_value : float
    detail    : DataFrame，逐箱明细

    PSI 判断
    --------
    < 0.10   稳定
    0.10~0.20  轻微漂移，关注
    > 0.20   显著漂移，触发重建
    """
    breakpoints = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))

    exp_cnt = np.histogram(expected, bins=breakpoints)[0]
    act_cnt = np.histogram(actual,   bins=breakpoints)[0]

    exp_pct = exp_cnt / (len(expected) + eps) + eps
    act_pct = act_cnt / (len(actual)   + eps) + eps

    psi_bins = (act_pct - exp_pct) * np.log(act_pct / exp_pct)

    detail = pd.DataFrame({
        "bin_low":   breakpoints[:-1],
        "bin_high":  breakpoints[1:],
        "exp_pct":   exp_pct,
        "act_pct":   act_pct,
        "psi":       psi_bins,
        "status":    np.where(psi_bins > 0.025, "⚠️", "✅"),
    })
    return float(psi_bins.sum()), detail


# ── Lift / 捕获率 ──────────────────────────────────────────────────────────

def lift_table(y_true: np.ndarray,
               y_score: np.ndarray,
               n_bins: int = 10) -> pd.DataFrame:
    """
    Lift 表：按评分排序后，各分位的坏客户捕获率与 Lift 值。

    Examples
    --------
    >>> tbl = lift_table(y_true, y_score)
    >>> print(tbl[["decile","cum_bad_capture","lift"]].to_string())
    """
    df = pd.DataFrame({"score": y_score, "label": y_true})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    total_bad = df["label"].sum()

    df["decile"] = pd.qcut(df.index / len(df), q=n_bins,
                            labels=range(1, n_bins + 1))
    tbl = df.groupby("decile", observed=True).agg(
        count=("label","count"), bad=("label","sum")
    ).reset_index()
    tbl["bad_rate"]        = tbl["bad"] / tbl["count"]
    tbl["cum_bad_capture"] = tbl["bad"].cumsum() / total_bad
    tbl["pct_population"]  = tbl["count"].cumsum() / len(df)
    tbl["lift"]            = tbl["cum_bad_capture"] / tbl["pct_population"]
    return tbl


# ── 综合评估报告 ───────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray,
             y_score: np.ndarray,
             label: str = "Model") -> dict:
    """
    一键输出完整评估指标。

    Returns
    -------
    dict with keys: auc, gini, ks, ks_threshold, ap

    Examples
    --------
    >>> metrics = evaluate(y_val, y_pred, label="LightGBM v1")
    """
    from sklearn.metrics import average_precision_score

    auc  = roc_auc_score(y_true, y_score)
    ks, thr = ks_stat(y_true, y_score)
    ap   = average_precision_score(y_true, y_score)

    metrics = dict(label=label, auc=auc, gini=2*auc-1, ks=ks,
                   ks_threshold=thr, ap=ap)

    bar  = "=" * 42
    print(f"\n{bar}")
    print(f"  {label}")
    print(bar)
    print(f"  AUC  : {auc:.4f}")
    print(f"  GINI : {2*auc-1:.4f}")
    print(f"  KS   : {ks:.4f}  (thr={thr:.4f})")
    print(f"  AP   : {ap:.4f}")
    print(bar)
    return metrics


def compare_models(results: List[dict]) -> pd.DataFrame:
    """
    对比多个模型的评估结果。

    Parameters
    ----------
    results : list of dicts from evaluate()

    Examples
    --------
    >>> compare_models([
    ...     evaluate(y, pred_lr,   "LR Scorecard"),
    ...     evaluate(y, pred_lgbm, "LightGBM"),
    ... ])
    """
    df = pd.DataFrame(results)[["label","auc","gini","ks","ap"]]
    df = df.sort_values("ks", ascending=False).reset_index(drop=True)
    return df
