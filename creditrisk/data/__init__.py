"""
creditrisk.data
==============
天池数据集加载、标签定义、时序划分。

数据集来源：
  阿里云天池「金融风控-贷款违约预测」
  https://tianchi.aliyun.com/competition/entrance/531830
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Union


# ── 天池数据集字段说明 ──────────────────────────────────────────────────────
TIANCHI_DTYPES = {
    "id":                    "int64",
    "loanAmnt":              "float64",
    "term":                  "int64",       # 36 / 60 (月)
    "interestRate":          "float64",
    "installment":           "float64",
    "grade":                 "category",    # A-G
    "subGrade":              "category",    # A1-G5
    "employmentTitle":       "float64",
    "employmentLength":      "category",    # '1 year', '10+ years', ...
    "homeOwnership":         "category",
    "annualIncome":          "float64",
    "verificationStatus":    "category",
    "issueDate":             "object",
    "isDefault":             "int64",       # TARGET: 1=违约
    "purpose":               "category",
    "postCode":              "float64",
    "regionCode":            "float64",
    "dti":                   "float64",
    "delinquency_2years":    "float64",
    "ficoRangeLow":          "float64",
    "ficoRangeHigh":         "float64",
    "openAcc":               "float64",
    "pubRec":                "float64",
    "pubRecBankruptcies":    "float64",
    "revolBal":              "float64",
    "revolUtil":             "float64",
    "totalAcc":              "float64",
    "initialListStatus":     "category",    # w / f
    "applicationType":       "category",    # Individual / Joint App
    "earliesCreditLine":     "object",
    "title":                 "float64",
    "policyCode":            "float64",     # 全为1，无信息量
    # 匿名特征 n0-n14
    **{f"n{i}": "float64" for i in range(15)},
}

TARGET = "isDefault"

CATEGORICAL_COLS = [
    "grade", "subGrade", "employmentLength", "homeOwnership",
    "verificationStatus", "purpose", "initialListStatus", "applicationType",
]

DATE_COLS = ["issueDate", "earliesCreditLine"]

# policyCode 全为1，对预测无贡献，直接丢弃
DROP_COLS = ["id", "policyCode"]


# ── 加载函数 ───────────────────────────────────────────────────────────────

def load_tianchi(data_dir: Union[str, Path],
                 nrows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载天池贷款违约预测数据集。

    Parameters
    ----------
    data_dir : str | Path
        存放 train.csv 和 testA.csv 的目录路径
    nrows : int, optional
        调试时加载前 N 行（None = 全量）

    Returns
    -------
    train, test : pd.DataFrame

    Examples
    --------
    >>> train, test = load_tianchi("data/")
    >>> train.shape, test.shape
    ((800000, 47), (200000, 46))
    """
    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"
    test_path  = data_dir / "testA.csv"

    assert train_path.exists(), f"找不到训练集：{train_path}"
    assert test_path.exists(),  f"找不到测试集：{test_path}"

    kw = dict(nrows=nrows)
    train = pd.read_csv(train_path, **kw)
    test  = pd.read_csv(test_path,  **kw)

    train = _preprocess_raw(train, is_train=True)
    test  = _preprocess_raw(test,  is_train=False)

    print(f"训练集：{train.shape}  |  坏率：{train[TARGET].mean():.4%}")
    print(f"测试集：{test.shape}")
    return train, test


def _preprocess_raw(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    """内部：原始字段清洗（类型转换、日期解析）"""
    df = df.copy()

    # 丢弃无用列
    drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop)

    # 就业年限：文本 → 数值
    df["employmentLength"] = _parse_employment_length(df["employmentLength"])

    # 日期 → 数值特征
    if "issueDate" in df.columns:
        df["issueDate_month"] = pd.to_datetime(df["issueDate"], errors="coerce").dt.month
        df["issueDate_year"]  = pd.to_datetime(df["issueDate"], errors="coerce").dt.year
        df = df.drop(columns=["issueDate"])

    if "earliesCreditLine" in df.columns:
        # 格式如 "Dec-2000"
        parsed = pd.to_datetime(df["earliesCreditLine"], format="%b-%Y", errors="coerce")
        ref = pd.Timestamp("2020-01-01")
        df["creditHistory_months"] = (ref - parsed).dt.days // 30
        df = df.drop(columns=["earliesCreditLine"])

    # grade 编码为有序整数（A=1, B=2, ...）
    grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    if "grade" in df.columns:
        df["grade"] = df["grade"].map(grade_map)

    # subGrade: A1=1, A2=2, ..., G5=35
    if "subGrade" in df.columns:
        df["subGrade"] = df["subGrade"].apply(_parse_subgrade)

    return df


def _parse_employment_length(series: pd.Series) -> pd.Series:
    """'10+ years' → 10, '< 1 year' → 0, NaN → -1"""
    mapping = {
        "< 1 year": 0,
        "1 year":   1, "2 years":  2, "3 years":  3, "4 years":  4,
        "5 years":  5, "6 years":  6, "7 years":  7, "8 years":  8,
        "9 years":  9, "10+ years":10,
    }
    return series.map(mapping).fillna(-1).astype("int8")


def _parse_subgrade(sg) -> int:
    if pd.isna(sg):
        return -1
    grade_order = {"A": 0, "B": 5, "C": 10, "D": 15, "E": 20, "F": 25, "G": 30}
    try:
        return grade_order.get(sg[0], 0) + int(sg[1])
    except (IndexError, ValueError):
        return -1


# ── 数据集划分 ──────────────────────────────────────────────────────────────

def train_val_split(train: pd.DataFrame,
                    val_ratio: float = 0.2,
                    stratify: bool = True,
                    random_state: int = 42
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按比例随机划分训练集/验证集（分层抽样保持坏率）。

    Examples
    --------
    >>> tr, val = train_val_split(train, val_ratio=0.2)
    """
    from sklearn.model_selection import train_test_split
    stratify_col = train[TARGET] if stratify else None
    tr, val = train_test_split(
        train, test_size=val_ratio, stratify=stratify_col,
        random_state=random_state
    )
    return tr.reset_index(drop=True), val.reset_index(drop=True)


def time_split(train: pd.DataFrame,
               date_col: str = "issueDate_year",
               val_years: Union[int, list] = 2019
               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按时间划分，模拟真实上线场景（训练集 < 验证集时间）。

    Parameters
    ----------
    val_years : int | list
        作为验证集的年份

    Examples
    --------
    >>> tr, val = time_split(train, val_years=2019)
    """
    if isinstance(val_years, int):
        val_years = [val_years]
    mask = train[date_col].isin(val_years)
    return train[~mask].copy(), train[mask].copy()


def get_feature_cols(df: pd.DataFrame) -> list:
    """返回特征列（排除目标列）"""
    return [c for c in df.columns if c != TARGET]
