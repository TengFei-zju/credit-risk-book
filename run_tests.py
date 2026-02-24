"""
信贷风控建模：打工人手册 — 代码快速测试
验证 creditrisk 包的核心模块可以正常导入和运行
无需真实数据（用合成数据）
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def make_credit_data(n=5000, seed=42):
    """生成模拟信贷数据"""
    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=n, n_features=20, n_informative=10,
        n_redundant=4, flip_y=0.05, class_sep=0.8,
        random_state=seed
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(20)])
    df["isDefault"] = y
    # 模拟天池字段
    df["grade"]      = rng.choice([1,2,3,4,5,6,7], size=n)
    df["loanAmnt"]   = rng.lognormal(10, 0.5, n)
    df["annualIncome"]= rng.lognormal(11, 0.6, n)
    return df


def test_data_module():
    from creditrisk.data import get_feature_cols, TARGET
    df = make_credit_data()
    feats = get_feature_cols(df)
    assert TARGET not in feats
    assert len(feats) == df.shape[1] - 1
    print("✅ creditrisk.data — OK")


def test_features_module():
    from creditrisk.features import WOEEncoder, ratio_features, anonymized_stats
    df = make_credit_data(1000)
    X = df.drop(columns=["isDefault"])
    y = df["isDefault"]

    # WOE编码
    enc = WOEEncoder(bins=5)
    X_woe = enc.fit_transform(X[["feat_0","feat_1","feat_2"]], y)
    assert X_woe.shape[0] == 1000
    iv_df = enc.iv_summary()
    assert "IV" in iv_df.columns
    print("✅ creditrisk.features.WOEEncoder — OK")

    # 匿名特征统计
    n_cols = [c for c in X.columns if c.startswith("feat_")]
    anon = anonymized_stats(X, n_cols=n_cols)
    assert "n_mean" in anon.columns
    print("✅ creditrisk.features.anonymized_stats — OK")


def test_evaluation_module():
    from creditrisk.evaluation import ks_stat, psi, evaluate, lift_table
    rng = np.random.default_rng(0)
    y     = rng.integers(0, 2, 500)
    score = rng.uniform(0, 1, 500)

    ks, thr = ks_stat(y, score)
    assert 0 <= ks <= 1
    print(f"✅ creditrisk.evaluation.ks_stat — KS={ks:.4f}")

    psi_val, psi_df = psi(score[:250], score[250:])
    assert psi_val >= 0
    print(f"✅ creditrisk.evaluation.psi — PSI={psi_val:.4f}")

    metrics = evaluate(y, score, label="test")
    assert "auc" in metrics
    print("✅ creditrisk.evaluation.evaluate — OK")


def test_models_module():
    from creditrisk.models import LGBMWithOOF
    df = make_credit_data(2000)
    X  = df.drop(columns=["isDefault", "grade", "loanAmnt", "annualIncome"])
    y  = df["isDefault"]
    X_test = X.sample(200, random_state=42)

    lgbm = LGBMWithOOF(n_splits=3, params={"n_estimators": 50, "verbosity": -1})
    oof, test_pred = lgbm.fit_predict(X, pd.Series(y), X_test)

    assert len(oof) == len(y)
    assert len(test_pred) == len(X_test)
    print("✅ creditrisk.models.LGBMWithOOF — OK")


def test_selection_module():
    from creditrisk.selection import AdversarialValidator
    df = make_credit_data(1000)
    X  = df.drop(columns=["isDefault"])
    X_test = X.sample(200, random_state=99)

    av = AdversarialValidator(n_splits=3,
                              lgbm_params={"n_estimators": 30, "verbosity": -1})
    av.fit(X, X_test)
    assert av.auc_ is not None
    print(f"✅ creditrisk.selection.AdversarialValidator — AUC={av.auc_:.4f}")


def test_ensemble_module():
    from creditrisk.ensemble import rank_blend, prob_blend
    rng = np.random.default_rng(42)
    p1  = rng.uniform(0, 1, 1000)
    p2  = rng.uniform(0, 1, 1000)

    blended = rank_blend([p1, p2])
    assert len(blended) == 1000
    print("✅ creditrisk.ensemble.rank_blend — OK")


if __name__ == "__main__":
    print("=" * 50)
    print("creditrisk 包 — 快速测试")
    print("=" * 50)
    test_data_module()
    test_features_module()
    test_evaluation_module()
    test_models_module()
    test_selection_module()
    test_ensemble_module()
    print("\n" + "=" * 50)
    print("✅ 所有测试通过！代码库可正常运行。")
    print("=" * 50)
