"""
creditrisk.utils
================
通用工具：可视化、PSI监控、计时器、日志。
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Optional


# ── 可视化 ─────────────────────────────────────────────────────────────────

def plot_roc_pr(y_true, pred_dict: dict, figsize=(13, 5)):
    """
    同时绘制 ROC 曲线和 PR 曲线（支持多模型对比）。

    Parameters
    ----------
    pred_dict : {"模型名": y_score, ...}

    Examples
    --------
    >>> plot_roc_pr(y_val, {
    ...     "LR Scorecard": pred_lr,
    ...     "LightGBM":     pred_lgbm,
    ... })
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for name, y_score in pred_dict.items():
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

        # PR
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(rec, prec)
        axes[1].plot(rec, prec, label=f"{name} (AP={pr_auc:.3f})")

    axes[0].plot([0,1],[0,1],"k--", alpha=0.4)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC 曲线"); axes[0].legend()

    axes[1].axhline(y=y_true.mean(), color="k", linestyle="--", alpha=0.4,
                    label=f"随机 ({y_true.mean():.3f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("PR 曲线"); axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_score_dist(y_true, y_score, bins=40, figsize=(13, 4)):
    """
    评分分布对比图（好坏客户分布 + 各分段坏率）。
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 左图：好坏分布
    axes[0].hist(y_score[y_true==0], bins=bins, alpha=0.6,
                 color="steelblue", label="好客户", density=True)
    axes[0].hist(y_score[y_true==1], bins=bins, alpha=0.6,
                 color="tomato",    label="坏客户", density=True)
    axes[0].set_xlabel("预测概率"); axes[0].set_ylabel("密度")
    axes[0].set_title("好坏客户分布"); axes[0].legend()

    # 右图：各分段坏率
    df = pd.DataFrame({"score": y_score, "label": y_true})
    df["band"] = pd.qcut(df["score"], q=10, duplicates="drop")
    bad_rate   = df.groupby("band", observed=True)["label"].mean()
    axes[1].bar(range(len(bad_rate)), bad_rate.values * 100, color="steelblue")
    axes[1].set_xticks(range(len(bad_rate)))
    axes[1].set_xticklabels(
        [str(b) for b in bad_rate.index], rotation=45, ha="right", fontsize=8
    )
    axes[1].set_ylabel("坏率 (%)"); axes[1].set_title("各分段坏率")

    plt.tight_layout()
    plt.show()


def plot_ks_curve(y_true, y_score, label="Model"):
    """绘制 KS 曲线"""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ks = max(tpr - fpr)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, tpr, label="累计坏率（TPR）", color="tomato")
    plt.plot(thresholds, fpr, label="累计好率（FPR）", color="steelblue")
    plt.plot(thresholds, tpr - fpr, label=f"KS = {ks:.4f}", color="green", ls="--")
    plt.axvline(thresholds[np.argmax(tpr - fpr)], color="gray", ls=":", alpha=0.7)
    plt.xlabel("评分阈值"); plt.gca().invert_xaxis()
    plt.title(f"KS 曲线 — {label}"); plt.legend()
    plt.tight_layout(); plt.show()


def plot_vintage(vintage_df: pd.DataFrame, title: str = "Vintage Analysis"):
    """绘制 Vintage 曲线"""
    fig, ax = plt.subplots(figsize=(14, 6))
    for cohort in vintage_df.index:
        ax.plot(vintage_df.columns,
                vintage_df.loc[cohort] * 100,
                marker="o", markersize=3, label=str(cohort))
    ax.set_xlabel("账龄（月）"); ax.set_ylabel("累计坏率 (%)"); ax.set_title(title)
    ax.legend(loc="upper left", ncol=4, fontsize=8)
    plt.tight_layout(); plt.show()


def plot_feature_importance(importance_df: pd.DataFrame,
                              top_n: int = 30,
                              title: str = "Feature Importance"):
    """绘制特征重要性水平条形图"""
    df = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, top_n * 0.28 + 1))
    ax.barh(df.index, df.iloc[:, 0], color="steelblue", alpha=0.85)
    ax.set_xlabel("Importance"); ax.set_title(title)
    plt.tight_layout(); plt.show()


# ── 计时器 ─────────────────────────────────────────────────────────────────

class Timer:
    """
    简单计时器，用于记录各建模步骤的耗时。

    Examples
    --------
    >>> with Timer("特征工程"):
    ...     X = build_features(df)
    """
    def __init__(self, name: str = ""):
        self.name = name

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self._start
        print(f"[{self.name}] 耗时 {elapsed:.1f}s")


# ── 简单日志 ───────────────────────────────────────────────────────────────

def log_step(msg: str, level: str = "INFO") -> None:
    """打印带时间戳的步骤日志"""
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S")
    icons = {"INFO": "ℹ️", "WARN": "⚠️", "ERROR": "❌", "OK": "✅"}
    print(f"[{ts}] {icons.get(level, '')} {msg}")


# ── 数据概况 ───────────────────────────────────────────────────────────────

def data_overview(df: pd.DataFrame, target: Optional[str] = None) -> pd.DataFrame:
    """
    快速输出数据集概况（类型、缺失率、唯一值数、与目标的相关性）。
    """
    rows = []
    for col in df.columns:
        row = {
            "feature":      col,
            "dtype":        str(df[col].dtype),
            "missing_rate": df[col].isnull().mean(),
            "nunique":      df[col].nunique(),
        }
        if target and df[col].dtype.kind in "iufc" and col != target:
            row["corr_with_target"] = df[col].corr(df[target])
        rows.append(row)

    overview = pd.DataFrame(rows).sort_values("missing_rate", ascending=False)
    print(f"Shape: {df.shape}  |  Target bad rate: "
          f"{df[target].mean():.4%}" if target else f"Shape: {df.shape}")
    return overview
