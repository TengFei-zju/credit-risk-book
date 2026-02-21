"""
creditrisk.ensemble
===================
模型融合：OOF Stacking、Rank-based Blending、Pseudo-labeling。
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Callable, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# ── Rank-based Blending ────────────────────────────────────────────────────

def rank_blend(pred_list: List[np.ndarray],
               weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Rank-based Blending（基于排名的融合）。

    比直接平均概率更稳健，因为不同模型的概率尺度可能不同，
    而 rank 是统一的。Kaggle 竞赛中最常用的融合方式之一。

    Parameters
    ----------
    pred_list : list of 1-D arrays，各模型预测概率
    weights   : 各模型权重（None = 等权）

    Examples
    --------
    >>> blended = rank_blend([pred_lr, pred_lgbm, pred_xgb], weights=[1, 2, 2])
    """
    if weights is None:
        weights = [1.0] * len(pred_list)

    ranks = [pd.Series(p).rank(pct=True).values * w
             for p, w in zip(pred_list, weights)]
    return np.stack(ranks, axis=1).sum(axis=1) / sum(weights)


def prob_blend(pred_list: List[np.ndarray],
               weights: Optional[List[float]] = None) -> np.ndarray:
    """加权概率平均融合"""
    if weights is None:
        weights = [1.0] * len(pred_list)
    total = sum(weights)
    return sum(p * w for p, w in zip(pred_list, weights)) / total


# ── OOF Stacking ──────────────────────────────────────────────────────────

class StackingEnsemble:
    """
    OOF Stacking（基于 Out-of-Fold 预测的叠加集成）。

    第一层（Base Layer）：
      多个 LGBMWithOOF 模型，各自产生 OOF 预测和测试集预测
    第二层（Meta Layer）：
      用 OOF 预测作为特征训练元模型（默认 Logistic Regression）

    使用示例
    --------
    >>> from creditrisk.models import LGBMWithOOF
    >>> base_models = [
    ...     LGBMWithOOF(params=params_v1, n_splits=5),
    ...     LGBMWithOOF(params=params_v2, n_splits=5),
    ... ]
    >>> stack = StackingEnsemble(base_models)
    >>> final_pred = stack.fit_predict(X_tr, y_tr, X_test)
    """

    def __init__(self,
                 base_models: list,
                 meta_model=None,
                 use_features_in_meta: bool = False):
        """
        Parameters
        ----------
        base_models : list of models with fit_predict(X_tr, y_tr, X_test) API
        meta_model  : 元模型（None = Logistic Regression）
        use_features_in_meta : 是否将原始特征也传入元模型
        """
        self.base_models = base_models
        self.meta_model  = meta_model or LogisticRegression(C=0.01, max_iter=1000)
        self.use_features_in_meta = use_features_in_meta

    def fit_predict(self,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    X_test: pd.DataFrame) -> np.ndarray:
        """
        Returns
        -------
        test_pred : 最终测试集预测概率
        """
        train_meta = np.zeros((len(X_train), len(self.base_models)))
        test_meta  = np.zeros((len(X_test),  len(self.base_models)))

        for i, model in enumerate(self.base_models):
            print(f"\n=== Base Model {i+1}/{len(self.base_models)} ===")
            oof, test_pred = model.fit_predict(X_train, y_train, X_test)
            train_meta[:, i] = oof
            test_meta[:, i]  = test_pred

            auc = roc_auc_score(y_train, oof)
            print(f"Base {i+1} OOF AUC = {auc:.4f}")

        # 元模型训练
        if self.use_features_in_meta:
            train_meta = np.hstack([train_meta, X_train.values])
            test_meta  = np.hstack([test_meta,  X_test.values])

        print("\n=== Meta Model ===")
        self.meta_model.fit(train_meta, y_train)
        final_pred = self.meta_model.predict_proba(test_meta)[:, 1]

        oof_meta = self.meta_model.predict_proba(train_meta)[:, 1]
        auc_meta = roc_auc_score(y_train, oof_meta)
        print(f"Meta Model OOF AUC = {auc_meta:.4f}")

        return final_pred


# ── Pseudo-labeling ────────────────────────────────────────────────────────

def pseudo_label(model,
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 X_test: pd.DataFrame,
                 confidence_threshold: float = 0.05,
                 bad_multiplier: float = 2.0,
                 random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Pseudo-labeling（伪标签）半监督学习。

    用于信贷风控时的逻辑：
    - 对测试集高置信度预测（极低风险或极高风险）的样本赋予伪标签
    - 混入训练集重新训练，扩充有效样本量

    在信贷建模中，这等价于一种「拒绝推断」：
    将测试集中高分段（预测好客户）和低分段（预测坏客户）的样本纳入训练。

    Parameters
    ----------
    confidence_threshold : 判定为"好客户"的概率阈值（< threshold = 好，> 1-threshold = 坏）
    bad_multiplier       : 对坏客户伪标签的保守放大系数

    Returns
    -------
    X_combined, y_combined : 包含伪标签的扩充训练集
    """
    # 初始训练并预测测试集
    test_probs = model.predict(X_test)

    # 高置信度好客户
    good_mask = test_probs < confidence_threshold
    # 高置信度坏客户（更保守，用更高阈值）
    bad_mask  = test_probs > (1 - confidence_threshold / bad_multiplier)

    pseudo_X = pd.concat([
        X_test[good_mask],
        X_test[bad_mask],
    ], ignore_index=True)
    pseudo_y = pd.Series(
        [0] * good_mask.sum() + [1] * bad_mask.sum()
    )

    n_good = good_mask.sum()
    n_bad  = bad_mask.sum()
    print(f"伪标签：好客户 {n_good} 条  坏客户 {n_bad} 条  "
          f"({(n_good+n_bad)/len(X_test):.1%} 的测试集)")

    X_combined = pd.concat([X_train, pseudo_X], ignore_index=True)
    y_combined = pd.concat([y_train, pseudo_y], ignore_index=True)
    return X_combined, y_combined
