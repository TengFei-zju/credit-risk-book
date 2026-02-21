"""
creditrisk.models
=================
模型封装：评分卡（LR + WOE）、LightGBM 带 OOF、评分转换。
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from creditrisk.evaluation import evaluate


# ── 评分卡 ─────────────────────────────────────────────────────────────────

class Scorecard:
    """
    标准信用评分卡：WOE编码 + 逻辑回归 + 刻度转换。

    参数
    ----
    base_score : int    基准分（默认600）
    base_odds  : float  基准好坏比（默认50:1，即 odds=50）
    pdo        : int    好坏比翻倍对应的分数变化（默认20）

    使用示例
    --------
    >>> from creditrisk.features import WOEEncoder
    >>> enc = WOEEncoder()
    >>> X_woe = enc.fit_transform(X_train, y_train)
    >>> sc  = Scorecard()
    >>> sc.fit(X_woe, y_train)
    >>> scores = sc.predict_score(X_woe)
    >>> sc.card_table(enc)         # 输出评分卡明细
    """

    def __init__(self,
                 base_score: int   = 600,
                 base_odds: float  = 50.0,
                 pdo: int          = 20,
                 C: float          = 0.1,
                 max_iter: int     = 1000):
        self.base_score = base_score
        self.base_odds  = base_odds
        self.pdo        = pdo
        self.C          = C
        self.max_iter   = max_iter
        self._lr: Optional[LogisticRegression] = None
        self._B: float = pdo / np.log(2)
        self._A: float = base_score + self._B * np.log(base_odds)

    def fit(self, X_woe: pd.DataFrame, y: pd.Series) -> "Scorecard":
        self._feature_names = list(X_woe.columns)
        self._lr = LogisticRegression(
            C=self.C, max_iter=self.max_iter,
            solver="lbfgs", class_weight="balanced"
        )
        self._lr.fit(X_woe, y)
        return self

    def predict_proba(self, X_woe: pd.DataFrame) -> np.ndarray:
        return self._lr.predict_proba(X_woe)[:, 1]

    def predict_score(self, X_woe: pd.DataFrame) -> np.ndarray:
        """将 log-odds 转换为整数评分"""
        log_odds = np.log(
            self._lr.predict_proba(X_woe)[:, 1] /
            (1 - self._lr.predict_proba(X_woe)[:, 1] + 1e-9)
        )
        return np.round(self._A - self._B * log_odds).astype(int)

    def card_table(self, woe_encoder) -> pd.DataFrame:
        """
        生成评分卡明细表（每个分箱对应的分值）。
        """
        n  = len(self._feature_names)
        B  = self._B
        A  = self._A

        rows = []
        for i, feat in enumerate(self._feature_names):
            coef = self._lr.coef_[0][i]
            # 截距均摊到每个变量
            intercept_part = self._lr.intercept_[0] / n
            tbl = woe_encoder.binning_table(feat)
            for _, row in tbl.iterrows():
                points = round(A / n - B * (coef * row.get("WoE", row.get("woe", 0))
                                             + intercept_part))
                rows.append({
                    "feature":  feat,
                    "bin":      row.name if hasattr(row, "name") else row.get("bin",""),
                    "woe":      round(row.get("WoE", row.get("woe", 0)), 4),
                    "points":   points,
                })
        return pd.DataFrame(rows)

    def score_distribution(self, X_woe: pd.DataFrame,
                            y: pd.Series) -> pd.DataFrame:
        """各分段的通过率与坏率"""
        scores  = self.predict_score(X_woe)
        df      = pd.DataFrame({"score": scores, "label": y})
        df["band"] = pd.cut(df["score"], bins=10)
        tbl = df.groupby("band", observed=True).agg(
            count=("label","count"), bad=("label","sum")
        )
        tbl["pass_cum"] = tbl["count"].cumsum() / len(df)
        tbl["bad_rate"] = tbl["bad"] / tbl["count"]
        return tbl


# ── LightGBM + OOF ─────────────────────────────────────────────────────────

import lightgbm as lgb


class LGBMWithOOF:
    """
    LightGBM 带 OOF（Out-of-Fold）预测，支持：
    - 分层 K-Fold 训练
    - Early stopping
    - 按月时序 K-Fold（可选）

    使用示例
    --------
    >>> lgbm = LGBMWithOOF(n_splits=5, params=lgbm_params)
    >>> oof_pred, test_pred = lgbm.fit_predict(X_train, y_train, X_test)
    >>> from creditrisk.evaluation import evaluate
    >>> evaluate(y_train, oof_pred, "LightGBM OOF")
    """

    DEFAULT_PARAMS = {
        "objective":         "binary",
        "metric":            "auc",
        "verbosity":         -1,
        "num_leaves":        31,
        "min_child_samples": 100,
        "learning_rate":     0.05,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "lambda_l1":         0.1,
        "lambda_l2":         0.1,
        "n_estimators":      2000,
    }

    def __init__(self,
                 n_splits: int = 5,
                 params: Optional[dict] = None,
                 random_state: int = 42,
                 early_stopping_rounds: int = 50):
        self.n_splits = n_splits
        self.params   = {**self.DEFAULT_PARAMS, **(params or {})}
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.models_: List[lgb.Booster] = []
        self.oof_scores_: List[float] = []
        self.feature_importance_: Optional[pd.DataFrame] = None

    def fit_predict(self,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    X_test: pd.DataFrame,
                    feature_cols: Optional[List[str]] = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        oof_pred  : OOF 预测（与 y_train 等长，可直接送 evaluate）
        test_pred : 测试集预测（各 fold 均值）
        """
        if feature_cols is None:
            feature_cols = list(X_train.columns)

        kf   = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                               random_state=self.random_state)
        oof  = np.zeros(len(X_train))
        test_preds = np.zeros((len(X_test), self.n_splits))
        imp_list = []

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            # 动态设置正负样本权重
            pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
            fold_params = {**self.params, "scale_pos_weight": pos_weight}

            dtrain = lgb.Dataset(X_tr[feature_cols], label=y_tr)
            dval   = lgb.Dataset(X_val[feature_cols], label=y_val)

            model = lgb.train(
                params=fold_params,
                train_set=dtrain,
                valid_sets=[dval],
                num_boost_round=self.params.get("n_estimators", 2000),
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(200),
                ],
            )
            self.models_.append(model)

            oof_fold   = model.predict(X_val[feature_cols])
            oof[val_idx] = oof_fold
            test_preds[:, fold] = model.predict(X_test[feature_cols])

            from sklearn.metrics import roc_auc_score
            fold_auc = roc_auc_score(y_val, oof_fold)
            self.oof_scores_.append(fold_auc)
            print(f"  Fold {fold+1}/{self.n_splits}  AUC={fold_auc:.4f}  "
                  f"trees={model.num_trees()}")

            imp_list.append(pd.Series(
                model.feature_importance(importance_type="gain"),
                index=feature_cols, name=f"fold{fold+1}"
            ))

        print(f"\n  OOF AUC = {np.mean(self.oof_scores_):.4f} "
              f"± {np.std(self.oof_scores_):.4f}")

        self.feature_importance_ = (
            pd.concat(imp_list, axis=1)
            .mean(axis=1)
            .sort_values(ascending=False)
            .rename("importance_gain")
            .to_frame()
        )
        return oof, test_preds.mean(axis=1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """用所有 fold 模型的均值预测"""
        preds = np.stack([m.predict(X) for m in self.models_], axis=1)
        return preds.mean(axis=1)
