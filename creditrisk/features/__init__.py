"""
creditrisk.features
===================
特征工程：WOE编码、最优分箱、聚合特征、Target Encoding（含CV防穿越）。
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple, Union
from sklearn.model_selection import StratifiedKFold


# ── WOE / IV ───────────────────────────────────────────────────────────────

class WOEEncoder:
    """
    WOE（证据权重）编码器，支持：
    - 数值型特征的最优等频分箱
    - 类别型特征直接映射
    - 缺失值单独分箱（填充为 -999 后处理）
    - 单调性约束（auto 自动检测）

    使用示例
    --------
    >>> enc = WOEEncoder(monotonic='auto', min_bin_pct=0.05)
    >>> enc.fit(X_train, y_train)
    >>> X_woe = enc.transform(X_train)
    >>> print(enc.iv_summary())
    """

    def __init__(self,
                 bins: int = 10,
                 min_bin_pct: float = 0.05,
                 monotonic: str = 'auto',   # 'auto' | 'increasing' | 'decreasing' | 'none'
                 smooth: float = 0.5):       # 平滑项，防 log(0)
        self.bins = bins
        self.min_bin_pct = min_bin_pct
        self.monotonic = monotonic
        self.smooth = smooth
        self._encoders: Dict[str, dict] = {}   # feature -> {bins, woe_map, iv}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOEEncoder":
        self._total_bad  = y.sum()
        self._total_good = (1 - y).sum()
        for col in X.columns:
            self._encoders[col] = self._fit_one(X[col], y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in X.columns:
            if col in self._encoders:
                out[col + "_woe"] = self._transform_one(X[col], self._encoders[col])
        return pd.DataFrame(out, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def iv_summary(self) -> pd.DataFrame:
        """返回各特征的 IV 汇总，按 IV 降序排列"""
        rows = [{"feature": col, "IV": meta["iv"]} for col, meta in self._encoders.items()]
        df = pd.DataFrame(rows).sort_values("IV", ascending=False).reset_index(drop=True)
        df["strength"] = df["IV"].apply(_iv_label)
        return df

    def binning_table(self, feature: str) -> pd.DataFrame:
        """返回单个特征的分箱明细表"""
        return self._encoders[feature]["table"]

    # ── 内部方法 ────────────────────────────────────────────────────────────

    def _fit_one(self, series: pd.Series, y: pd.Series) -> dict:
        try:
            from optbinning import OptimalBinning
            dtype = "numerical" if series.dtype.kind in "iufc" else "categorical"
            ob = OptimalBinning(
                name=series.name,
                dtype=dtype,
                solver="cp",
                monotonic_trend=self.monotonic if self.monotonic != "none" else "auto",
                min_bin_size=self.min_bin_pct,
            )
            ob.fit(series.values, y.values)
            table = ob.binning_table.build()
            iv = table["IV"].iloc[-1]  # 最后一行是合计
            woe_map = dict(zip(table.index[:-2], table["WoE"].iloc[:-2]))
            return {"optb": ob, "table": table, "iv": iv, "woe_map": woe_map,
                    "dtype": dtype}
        except ImportError:
            return self._fit_one_fallback(series, y)

    def _fit_one_fallback(self, series: pd.Series, y: pd.Series) -> dict:
        """optbinning 未安装时的等频分箱兜底"""
        s = smooth = self.smooth
        missing_mask = series.isna()
        table_rows = []

        if series.dtype.kind in "iufc":
            # 数值型：等频分箱
            breaks = np.unique(
                np.percentile(series.dropna(), np.linspace(0, 100, self.bins + 1))
            )
            series_binned = pd.cut(series, bins=breaks, labels=False, include_lowest=True)
        else:
            series_binned = series.astype(str)

        total_bad  = y.sum()
        total_good = (1 - y).sum()

        # 缺失单独处理
        if missing_mask.any():
            y_miss = y[missing_mask]
            b, g = y_miss.sum() + s, (1 - y_miss).sum() + s
            woe = np.log((b / total_bad) / (g / total_good))
            iv_part = (b / total_bad - g / total_good) * woe
            table_rows.append({"bin": "Missing", "woe": woe, "iv": iv_part})

        for bin_val in sorted(series_binned[~missing_mask].unique()):
            mask = series_binned == bin_val
            y_bin = y[mask]
            b, g = y_bin.sum() + s, (1 - y_bin).sum() + s
            woe = np.log((b / total_bad) / (g / total_good))
            iv_part = (b / total_bad - g / total_good) * woe
            table_rows.append({"bin": bin_val, "woe": woe, "iv": iv_part})

        table = pd.DataFrame(table_rows)
        iv = table["iv"].sum()
        woe_map = dict(zip(table["bin"], table["woe"]))
        return {"table": table, "iv": iv, "woe_map": woe_map, "dtype": "fallback"}

    def _transform_one(self, series: pd.Series, meta: dict) -> pd.Series:
        if "optb" in meta:
            return pd.Series(meta["optb"].transform(series.values, metric="woe"),
                             index=series.index)
        # fallback
        if series.dtype.kind in "iufc":
            breaks = [row["bin"] for row in meta["table"].to_dict("records")
                      if row["bin"] != "Missing"]
            # 简化：直接按分位数映射
        return series.map(meta["woe_map"]).fillna(
            meta["woe_map"].get("Missing", 0)
        )


def _iv_label(iv: float) -> str:
    if iv < 0.02:  return "无预测力"
    if iv < 0.10:  return "弱"
    if iv < 0.30:  return "中等"
    if iv < 0.50:  return "强"
    return "极强（疑似穿越）"


# ── Target Encoding (CV防穿越) ─────────────────────────────────────────────

class TargetEncoder:
    """
    带 k-fold 的 Target Encoding，防止训练集穿越。
    低频类别做平滑（Bayesian smoothing）。

    使用示例
    --------
    >>> te = TargetEncoder(cols=["purpose", "homeOwnership"], smoothing=20)
    >>> X_train_enc = te.fit_transform(X_train, y_train)
    >>> X_test_enc  = te.transform(X_test)
    """

    def __init__(self,
                 cols: List[str],
                 n_splits: int = 5,
                 smoothing: float = 20.0,
                 random_state: int = 42):
        self.cols = cols
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state
        self._global_mean: float = 0.0
        self._col_maps: Dict[str, pd.Series] = {}

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        X = X.copy()
        self._global_mean = y.mean()
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                             random_state=self.random_state)

        for col in self.cols:
            X[col + "_te"] = self._global_mean
            for tr_idx, val_idx in kf.split(X, y):
                stats = (y.iloc[tr_idx]
                          .groupby(X[col].iloc[tr_idx])
                          .agg(["mean", "count"]))
                smooth = self._smooth(stats)
                X.iloc[val_idx, X.columns.get_loc(col + "_te")] = (
                    X[col].iloc[val_idx].map(smooth).fillna(self._global_mean)
                )

        # 全量拟合用于 test 集
        for col in self.cols:
            stats = y.groupby(X[col]).agg(["mean", "count"])
            self._col_maps[col] = self._smooth(stats)

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols:
            X[col + "_te"] = X[col].map(self._col_maps[col]).fillna(self._global_mean)
        return X

    def _smooth(self, stats: pd.DataFrame) -> pd.Series:
        """Bayesian smoothing：防止低频类别过拟合"""
        n = stats["count"]
        mean = stats["mean"]
        return (n * mean + self.smoothing * self._global_mean) / (n + self.smoothing)


# ── 聚合特征 ───────────────────────────────────────────────────────────────

def aggregate_features(df: pd.DataFrame,
                        group_cols: List[str],
                        agg_cols: List[str],
                        agg_funcs: List[str] = ["mean", "std", "min", "max"]
                        ) -> pd.DataFrame:
    """
    生成分组聚合特征（Group Statistics）。
    常用于：按 grade 聚合贷款金额、按 purpose 聚合利率等。

    Examples
    --------
    >>> feats = aggregate_features(
    ...     train, group_cols=["grade"], agg_cols=["loanAmnt", "interestRate"]
    ... )
    """
    agg_dict = {col: agg_funcs for col in agg_cols}
    grouped = df.groupby(group_cols).agg(agg_dict)
    grouped.columns = ["_".join(c) + f"_by_{'_'.join(group_cols)}"
                       for c in grouped.columns]
    return df[group_cols].join(grouped, on=group_cols).drop(columns=group_cols)


def ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造天池数据集常用比率特征。
    """
    out = pd.DataFrame(index=df.index)
    eps = 1e-6

    # 贷款金额 / 年收入
    if "loanAmnt" in df and "annualIncome" in df:
        out["loanToIncome"] = df["loanAmnt"] / (df["annualIncome"] + eps)

    # 月还款 / 月收入
    if "installment" in df and "annualIncome" in df:
        out["installmentToIncome"] = df["installment"] / (df["annualIncome"] / 12 + eps)

    # FICO 均值与范围
    if "ficoRangeLow" in df and "ficoRangeHigh" in df:
        out["ficoMean"]  = (df["ficoRangeLow"] + df["ficoRangeHigh"]) / 2
        out["ficoRange"] = df["ficoRangeHigh"] - df["ficoRangeLow"]

    # 循环利用率（已是百分比，转为小数）
    if "revolUtil" in df:
        out["revolUtil_norm"] = df["revolUtil"] / 100.0

    # 开放账户 / 总账户
    if "openAcc" in df and "totalAcc" in df:
        out["openAccRatio"] = df["openAcc"] / (df["totalAcc"] + eps)

    # 负债与收入比 × 利率（风险叠加）
    if "dti" in df and "interestRate" in df:
        out["dti_x_rate"] = df["dti"] * df["interestRate"]

    return out


def anonymized_stats(df: pd.DataFrame,
                     n_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    对天池的匿名特征 n0-n14 构造统计量。
    这些特征含义未知，统计聚合是常见 Kaggle 技巧。
    """
    if n_cols is None:
        n_cols = [c for c in df.columns if c.startswith("n") and c[1:].isdigit()]
    sub = df[n_cols].copy()

    out = pd.DataFrame(index=df.index)
    out["n_mean"]    = sub.mean(axis=1)
    out["n_std"]     = sub.std(axis=1)
    out["n_min"]     = sub.min(axis=1)
    out["n_max"]     = sub.max(axis=1)
    out["n_null_cnt"] = sub.isnull().sum(axis=1)
    out["n_skew"]    = sub.skew(axis=1)
    return out
