"""
creditrisk.selection
====================
特征选择：Null Importance（空重要性）、对抗验证（Adversarial Validation）。
两种方法均来自顶级 Kaggle 竞赛的核心技巧，在信贷风控中同样有效。
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List, Optional, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# ── Null Importance ────────────────────────────────────────────────────────

class NullImportanceSelector:
    """
    Null Importance（空重要性）特征选择。

    原理：
    1. 用真实标签训练模型，记录特征重要性（actual importance）
    2. 将标签随机打乱 N 次，每次训练并记录重要性（null distribution）
    3. 真实重要性 >> 空重要性 → 特征有效
       真实重要性 ≈ 空重要性 → 特征是噪声，应删除

    来源：Kaggle Grand Master Oliver Fouché 的技巧，
    在 Home Credit Default Risk 竞赛中被广泛采用。

    使用示例
    --------
    >>> selector = NullImportanceSelector(n_runs=50, threshold=0.75)
    >>> selector.fit(X_train, y_train)
    >>> selected = selector.selected_features_
    >>> selector.plot_importance(top_n=30)
    """

    def __init__(self,
                 n_runs: int = 50,
                 threshold: float = 0.75,
                 lgbm_params: Optional[dict] = None,
                 random_state: int = 42):
        """
        Parameters
        ----------
        n_runs    : 打乱标签的轮数（越多越准确，但越慢；50~100 为佳）
        threshold : 保留阈值：actual_score_pct > threshold 的特征被保留
                    actual_score_pct = actual_importance 在 null 分布中的百分位数
        """
        self.n_runs    = n_runs
        self.threshold = threshold
        self.random_state = random_state
        self.lgbm_params = lgbm_params or {
            "objective":    "binary",
            "metric":       "auc",
            "num_leaves":   31,
            "n_estimators": 200,
            "learning_rate":0.05,
            "verbosity":    -1,
        }
        self.actual_importance_: Optional[pd.Series] = None
        self.null_importance_:   Optional[pd.DataFrame] = None
        self.selected_features_: Optional[List[str]] = None
        self.score_df_:          Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NullImportanceSelector":
        print("Step 1/2: 训练真实标签模型...")
        self.actual_importance_ = self._get_importance(X, y, shuffle=False)

        print(f"Step 2/2: 训练 {self.n_runs} 轮打乱标签模型...")
        null_imps = []
        for i in range(self.n_runs):
            imp = self._get_importance(X, y, shuffle=True, seed=i)
            null_imps.append(imp)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{self.n_runs} 完成")

        self.null_importance_ = pd.concat(null_imps, axis=1)
        self.null_importance_.columns = [f"run_{i}" for i in range(self.n_runs)]

        self._compute_scores()
        return self

    def _compute_scores(self) -> None:
        """计算每个特征的实际重要性在 null 分布中的百分位数"""
        rows = []
        for feat in self.actual_importance_.index:
            actual = self.actual_importance_[feat]
            null   = self.null_importance_.loc[feat].values
            # 实际值 > null 分布的百分位（越高越好）
            score = (null < actual).mean()
            rows.append({"feature": feat,
                         "actual_importance": actual,
                         "null_mean": null.mean(),
                         "null_std":  null.std(),
                         "score_pct": score})

        self.score_df_ = (pd.DataFrame(rows)
                          .sort_values("score_pct", ascending=False)
                          .reset_index(drop=True))

        self.selected_features_ = (
            self.score_df_[self.score_df_["score_pct"] >= self.threshold]["feature"]
            .tolist()
        )
        n = len(self.selected_features_)
        total = len(self.actual_importance_)
        print(f"\n保留特征：{n} / {total}  (threshold={self.threshold})")

    def _get_importance(self, X, y, shuffle, seed=None) -> pd.Series:
        y_ = y.sample(frac=1, random_state=seed).values if shuffle else y.values
        model = lgb.LGBMClassifier(**self.lgbm_params, random_state=seed or self.random_state)
        model.fit(X, y_)
        return pd.Series(
            model.feature_importances_, index=X.columns, name="importance"
        )

    def plot_importance(self, top_n: int = 30) -> None:
        """可视化真实重要性 vs 空重要性分布"""
        import matplotlib.pyplot as plt

        df = self.score_df_.head(top_n)
        fig, ax = plt.subplots(figsize=(10, top_n * 0.3 + 1))

        y_pos = range(len(df))
        ax.barh(y_pos, df["actual_importance"], color="steelblue",
                alpha=0.8, label="Actual")
        ax.barh(y_pos, df["null_mean"], color="salmon",
                alpha=0.6, label="Null (mean)")
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(df["feature"])
        ax.invert_yaxis()
        ax.set_xlabel("Importance (gain)")
        ax.set_title(f"Null Importance (top {top_n})")
        ax.legend()
        plt.tight_layout()
        plt.show()


# ── Adversarial Validation ─────────────────────────────────────────────────

class AdversarialValidator:
    """
    对抗验证（Adversarial Validation）。

    原理：
    训练一个二分类模型来区分「训练集」vs「测试集」。
    - 如果模型能轻松区分（AUC >> 0.5），说明训练集与测试集分布不同
    - 对区分能力强的特征进行排查或删除

    在信贷风控中的应用：
    1. 识别时间漂移严重的特征（训练期 vs 上线期分布差异大）
    2. 找到导致模型线上泛化差的根源特征
    3. 在 A/B 测试设计中验证两组客群的特征分布一致性

    使用示例
    --------
    >>> av = AdversarialValidator()
    >>> av.fit(X_train, X_test)
    >>> print(f"AUC={av.auc_:.4f}")  # 越接近0.5越好
    >>> av.report()
    """

    def __init__(self,
                 lgbm_params: Optional[dict] = None,
                 n_splits: int = 5,
                 random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.lgbm_params = lgbm_params or {
            "objective":    "binary",
            "metric":       "auc",
            "num_leaves":   63,
            "n_estimators": 200,
            "learning_rate":0.05,
            "verbosity":    -1,
        }
        self.auc_:      Optional[float] = None
        self.feature_importance_: Optional[pd.DataFrame] = None

    def fit(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> "AdversarialValidator":
        """
        Parameters
        ----------
        X_train : 训练集特征（不含标签）
        X_test  : 测试集/线上集特征
        """
        # 合并，train=0, test=1
        X = pd.concat([X_train, X_test], ignore_index=True)
        y = np.array([0] * len(X_train) + [1] * len(X_test))

        kf    = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                random_state=self.random_state)
        aucs  = []
        imps  = []

        for fold, (tr, val) in enumerate(kf.split(X, y)):
            model = lgb.LGBMClassifier(**self.lgbm_params,
                                        random_state=self.random_state)
            model.fit(X.iloc[tr], y[tr])
            pred = model.predict_proba(X.iloc[val])[:, 1]
            auc  = roc_auc_score(y[val], pred)
            aucs.append(auc)
            imps.append(pd.Series(model.feature_importances_, index=X.columns))

        self.auc_ = np.mean(aucs)
        self.feature_importance_ = (
            pd.concat(imps, axis=1)
            .mean(axis=1)
            .sort_values(ascending=False)
            .rename("importance")
            .to_frame()
        )
        return self

    def report(self, top_n: int = 20) -> pd.DataFrame:
        """
        输出对抗验证报告。

        Returns
        -------
        DataFrame：区分能力最强的前 top_n 个特征（即分布漂移最严重的特征）
        """
        status = (
            "✅ 训练集与测试集分布相似，模型可泛化"
            if self.auc_ < 0.6
            else "⚠️  训练集与测试集存在分布差异，建议检查高重要性特征"
            if self.auc_ < 0.75
            else "❌ 严重分布漂移，模型线上效果可能大幅衰退"
        )
        print(f"\n对抗验证 AUC = {self.auc_:.4f}  {status}")
        top = self.feature_importance_.head(top_n)
        print(f"\n漂移最严重的前 {top_n} 个特征（建议考虑删除或转换）：")
        print(top.to_string())
        return top

    def get_leaked_features(self, auc_threshold: float = 0.7,
                             imp_quantile: float = 0.8) -> List[str]:
        """
        返回疑似时间泄漏/分布漂移严重的特征列表。
        仅在整体 AUC > auc_threshold 时才有意义。
        """
        if self.auc_ < auc_threshold:
            return []
        cutoff = self.feature_importance_["importance"].quantile(imp_quantile)
        return self.feature_importance_[
            self.feature_importance_["importance"] >= cutoff
        ].index.tolist()
