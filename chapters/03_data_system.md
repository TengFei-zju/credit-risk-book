# 第三章 数据体系与数据治理

## 3.1 风控数据的分类框架

```
风控数据体系
├── 申请人维度
│   ├── 基本信息（年龄、性别、学历、职业）
│   ├── 联系信息（手机、地址）
│   └── 身份信息（身份证、银行卡）
├── 信用历史维度
│   ├── 人行征信（查询记录、还款记录、负债情况）
│   ├── 百行征信（互金记录）
│   └── 内部历史（自有历史贷款记录）
├── 资产与负债维度
│   ├── 收入证明（工资流水、税单）
│   ├── 房产/车辆
│   └── 负债情况（信用卡、贷款）
├── 行为数据
│   ├── App使用行为（埋点、设备信息）
│   ├── 还款行为（历史还款模式）
│   └── 互动行为（客服、APP活跃度）
└── 外部数据
    ├── 运营商数据
    ├── 电商/支付数据
    └── 司法/黑名单数据
```

---

## 3.2 数据探索性分析（EDA）

建模前的EDA是识别数据问题、发现特征机会的关键步骤。

### 3.2.1 单变量分析

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def eda_univariate(df, col, target='label'):
    """单变量分析：缺失率、分布、与Y的关系"""
    report = {}

    # 基础统计
    report['dtype'] = str(df[col].dtype)
    report['missing_rate'] = df[col].isnull().mean()
    report['nunique'] = df[col].nunique()

    if df[col].dtype in ['object', 'category']:
        report['top5'] = df[col].value_counts(normalize=True).head()
    else:
        report['mean'] = df[col].mean()
        report['std'] = df[col].std()
        report['pct'] = df[col].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()

    # 与Y的相关性（仅数值型）
    if df[col].dtype not in ['object', 'category']:
        report['corr_with_y'] = df[col].corr(df[target])

    return report

# 批量EDA报告
def eda_report(df, features, target='label'):
    return pd.DataFrame([
        {'feature': col, **eda_univariate(df, col, target)}
        for col in features
    ])
```

### 3.2.2 目标变量分析

```python
def analyze_target(df, target='label', time_col='apply_date'):
    """分析Y变量的时间分布和基础坏率"""
    monthly = df.groupby(df[time_col].dt.to_period('M')).agg(
        total=(target, 'count'),
        bad=(target, 'sum'),
    )
    monthly['bad_rate'] = monthly['bad'] / monthly['total']

    print(f"总样本量：{len(df):,}")
    print(f"坏样本量：{df[target].sum():,}")
    print(f"整体坏率：{df[target].mean():.2%}")
    print("\n按月坏率分布：")
    print(monthly)

    return monthly
```

---

## 3.3 缺失值处理

### 3.3.1 缺失类型识别

风控数据的缺失往往**不是随机的（not MCAR）**，缺失本身就是信号：

| 缺失类型 | 示例 | 建模策略 |
|---------|------|---------|
| **MCAR**（完全随机缺失） | 系统故障丢数据 | 均值/中位数填充 |
| **MAR**（随机缺失） | 高学历客户不填学历 | 分组统计填充 |
| **MNAR**（非随机缺失） | 征信空白=无贷款记录 | **缺失本身作为特征** |

```python
# 将缺失转为特征的常见做法
df['has_credit_history'] = df['credit_score'].notna().astype(int)
df['credit_score_filled'] = df['credit_score'].fillna(-1)  # -1表示无记录

# WOE编码时，缺失单独分箱处理（见第四章）
```

### 3.3.2 缺失率阈值

| 缺失率 | 处理建议 |
|--------|---------|
| < 5% | 均值/众数填充，通常不影响模型 |
| 5%~30% | 增加缺失标志位，考虑分组填充 |
| 30%~70% | 谨慎使用，需验证缺失是否携带信息 |
| > 70% | 通常直接删除，除非缺失本身具有强预测力 |

---

## 3.4 异常值处理

### 3.4.1 异常值识别

```python
def detect_outliers(series, method='iqr', threshold=3):
    """
    异常值检测
    method: 'iqr' | 'zscore' | 'percentile'
    """
    if method == 'iqr':
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - threshold * IQR, Q3 + threshold * IQR
    elif method == 'zscore':
        mean, std = series.mean(), series.std()
        lower = mean - threshold * std
        upper = mean + threshold * std
    elif method == 'percentile':
        lower = series.quantile(0.01)
        upper = series.quantile(0.99)

    return lower, upper

# 缩尾（Winsorization）
def winsorize(df, col, lower_pct=0.01, upper_pct=0.99):
    lower = df[col].quantile(lower_pct)
    upper = df[col].quantile(upper_pct)
    df[col] = df[col].clip(lower=lower, upper=upper)
    return df
```

### 3.4.2 风控数据特有的"异常"

- **收入为0**：可能是真实无收入，也可能是填写错误 → 业务核实
- **年龄<18或>70**：可能是数据错误，也可能是实际边缘客群 → 准入规则处理
- **申请额度远超收入**：可能是欺诈信号 → 单独标记

---

## 3.5 Vintage分析

Vintage分析是风控数据分析的核心工具，用于理解资产质量的时间演变规律。

```python
def vintage_analysis(df, cohort_col='apply_month', age_col='loan_age',
                     bad_col='is_bad'):
    """
    Vintage分析：各放款月份的逾期率随账龄变化
    """
    vintage = df.pivot_table(
        values=bad_col,
        index=cohort_col,
        columns=age_col,
        aggfunc='mean'
    )
    return vintage

# 可视化
def plot_vintage(vintage_df, title='Vintage Analysis'):
    fig, ax = plt.subplots(figsize=(14, 7))
    for cohort in vintage_df.index:
        ax.plot(vintage_df.columns,
                vintage_df.loc[cohort] * 100,
                label=str(cohort), marker='o', markersize=3)
    ax.set_xlabel('账龄（月）')
    ax.set_ylabel('累计逾期率（%）')
    ax.set_title(title)
    ax.legend(loc='upper left', ncol=3, fontsize=8)
    plt.tight_layout()
    return fig
```

**Vintage图能回答的问题**：
- 近期放款质量是否在恶化（各cohort曲线是否上移）
- 资产的风险成熟期在哪（曲线何时趋于平稳）
- 特定时期（如疫情）的冲击是否已消散

---

## 3.6 样本不平衡处理

信用风险建模的坏率通常很低（1%~10%），样本不平衡是常态。

### 3.6.1 常用处理策略

```python
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# 方法一：欠采样（下采样好样本）
def under_sample(X, y, ratio=10):
    """好坏比例控制在 ratio:1"""
    df = pd.concat([X, y], axis=1)
    good = df[y == 0].sample(n=int(df[y == 1].sum() * ratio), random_state=42)
    bad = df[y == 1]
    return pd.concat([good, bad]).sample(frac=1, random_state=42)

# 方法二：过采样（SMOTE合成少数类）
# 注意：SMOTE在风控中需谨慎，合成样本可能与真实坏客户分布有偏差
smote = SMOTE(sampling_strategy=0.1, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 方法三：class_weight（最推荐，无需改变数据）
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')
# 或自定义权重
model = LogisticRegression(class_weight={0: 1, 1: 10})
```

**实践建议**：
- 评分卡/LR建模优先使用 `class_weight`
- 树模型（XGBoost/LightGBM）使用 `scale_pos_weight`
- 评估时始终用真实分布的测试集，不要在平衡后的数据上算AUC

---

## 3.7 特征覆盖率分析

在特征上线前，必须评估其**线上覆盖率**（即实际打分时能取到该特征的比例）。

```python
def feature_coverage_report(df_online_sample, features):
    """
    评估特征在在线流量中的覆盖率
    df_online_sample: 上线后的真实流量样本
    """
    report = []
    for feat in features:
        row = {
            'feature': feat,
            'missing_rate': df_online_sample[feat].isnull().mean(),
            'coverage_rate': df_online_sample[feat].notna().mean(),
            'zero_rate': (df_online_sample[feat] == 0).mean(),
        }
        report.append(row)

    return pd.DataFrame(report).sort_values('coverage_rate')
```

**覆盖率低的特征的处理**：
- 覆盖率 < 50%：慎重入模，缺失填充策略要充分测试
- 覆盖率 < 20%：通常不建议入模，或作为辅助特征
- 关注覆盖率的**人群偏差**：某类客群覆盖率低可能导致模型对其失效

---

## 3.8 数据治理基础

### 3.8.1 特征平台

规模化的风控团队需要特征平台统一管理特征：

```
特征平台核心能力：
├── 特征注册与文档（元数据管理）
├── 特征计算（离线批量 + 实时流式）
├── 特征存储（历史回溯 + 在线服务）
├── 特征监控（覆盖率、分布漂移告警）
└── 特征回溯（训练时能拉到任意时间点的特征值）
```

### 3.8.2 训练数据一致性

**训练集特征 = 线上特征**：这是工程上最难保证、出问题最多的地方。

常见的 training-serving skew 原因：
- 离线特征用了近实时数据，线上用了实时数据（时间差）
- 离线fillna的逻辑与线上不一致
- 离线特征计算的时区与线上不同

---

> **本章小结**：风控数据的质量决定了模型效果的上限。EDA、缺失处理、Vintage分析不是"数据清洗的繁琐工作"，而是理解资产风险特征的核心手段。特征平台和数据一致性是规模化建模的工程基础。
