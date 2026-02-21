# 第四章 特征工程

## 4.1 特征工程在风控中的核心地位

在信贷风控领域，**特征工程的重要性往往高于模型选择**。一个精心设计的特征集+简单的逻辑回归，通常优于一个粗糙特征集+复杂的深度学习模型。原因在于：

1. 风险信号往往隐藏在业务逻辑中，需要领域知识显式构造
2. 模型可解释性要求限制了模型复杂度，特征质量更关键
3. 数据量通常有限，复杂模型容易过拟合

---

## 4.2 WOE编码与信息价值（IV）

WOE（Weight of Evidence，证据权重）是评分卡建模的核心编码方式。

### 4.2.1 WOE计算原理

$$WOE_i = \ln\left(\frac{P(X=i|Y=1)}{P(X=i|Y=0)}\right) = \ln\left(\frac{Bad_i/Bad_{total}}{Good_i/Good_{total}}\right)$$

$$IV = \sum_i (P(Y=1|X=i) - P(Y=0|X=i)) \times WOE_i$$

**WOE直觉理解**：
- WOE > 0：该分箱中坏客户比例高于总体
- WOE < 0：该分箱中坏客户比例低于总体
- WOE = 0：该分箱与总体无差异

**IV判断标准**：

| IV值 | 预测能力 |
|------|---------|
| < 0.02 | 无预测力，通常丢弃 |
| 0.02~0.1 | 弱预测力 |
| 0.1~0.3 | 中等预测力 |
| 0.3~0.5 | 强预测力 |
| > 0.5 | 预测力极强，需检查是否穿越 |

```python
import numpy as np
import pandas as pd

def calculate_woe_iv(df, feature, target, bins=10, min_bin_pct=0.05):
    """
    计算特征的WOE和IV
    支持数值型（分箱）和类别型
    """
    df = df[[feature, target]].copy()

    # 数值型自动分箱
    if df[feature].dtype in ['float64', 'int64']:
        df['bin'] = pd.qcut(df[feature], q=bins, duplicates='drop')
    else:
        df['bin'] = df[feature]

    # 计算每个分箱的好坏数量
    stats = df.groupby('bin', observed=True)[target].agg(
        bad='sum',
        total='count'
    ).reset_index()
    stats['good'] = stats['total'] - stats['bad']

    # 处理零值（防止log(0)）
    stats['bad'] = stats['bad'].replace(0, 0.5)
    stats['good'] = stats['good'].replace(0, 0.5)

    total_bad = stats['bad'].sum()
    total_good = stats['good'].sum()

    stats['bad_rate'] = stats['bad'] / stats['total']
    stats['pct_bad'] = stats['bad'] / total_bad
    stats['pct_good'] = stats['good'] / total_good
    stats['woe'] = np.log(stats['pct_bad'] / stats['pct_good'])
    stats['iv'] = (stats['pct_bad'] - stats['pct_good']) * stats['woe']

    iv_total = stats['iv'].sum()
    return stats, iv_total
```

### 4.2.2 最优分箱

好的分箱需满足：
- **单调性**：WOE随特征值单调递增或递减（业务逻辑一致）
- **显著性**：每个分箱的好坏样本量足够
- **覆盖率**：缺失值单独分箱

```python
from optbinning import OptimalBinning

def optimal_woe_binning(df, feature, target):
    """使用optbinning库进行最优分箱"""
    optb = OptimalBinning(
        name=feature,
        dtype='numerical',
        solver='cp',
        monotonic_trend='auto',   # 自动检测单调方向
        min_bin_size=0.05,        # 最小分箱比例5%
    )
    optb.fit(df[feature].values, df[target].values)

    # 获取分箱统计表
    binning_table = optb.binning_table.build()
    return optb, binning_table
```

---

## 4.3 数值型特征处理

### 4.3.1 统计聚合特征（时序数据）

行为数据往往是时序的，需要在不同时间窗口上做聚合：

```python
def build_time_window_features(df, id_col, date_col, value_col, windows=[1, 3, 6]):
    """
    构建多时间窗口聚合特征
    常用于：还款行为、消费行为、征信查询等
    """
    results = []
    observation_date = df[date_col].max()  # 假设单一观察点

    for window in windows:
        cutoff = observation_date - pd.DateOffset(months=window)
        window_df = df[df[date_col] >= cutoff]

        agg = window_df.groupby(id_col)[value_col].agg([
            ('sum', 'sum'),
            ('mean', 'mean'),
            ('max', 'max'),
            ('min', 'min'),
            ('count', 'count'),
            ('std', 'std'),
        ]).add_prefix(f'{value_col}_m{window}_')

        results.append(agg)

    return pd.concat(results, axis=1)
```

### 4.3.2 比率型特征

比率特征往往比绝对值更稳定、更有区分力：

```python
# 常见比率特征示例
features = {
    # 负债率
    'debt_to_income': df['total_debt'] / (df['monthly_income'] + 1e-6),

    # 已用额度比例（Credit Utilization Rate）
    'credit_utilization': df['used_credit'] / (df['total_credit_limit'] + 1e-6),

    # 还款覆盖率
    'payment_coverage': df['monthly_payment'] / (df['monthly_income'] + 1e-6),

    # 近期与历史逾期比（趋势信号）
    'recent_overdue_ratio': df['overdue_m3'] / (df['overdue_m12'] + 1e-6),
}
```

---

## 4.4 类别型特征处理

### 4.4.1 高基数类别特征处理

职业、城市等高基数特征不能直接One-Hot编码：

```python
def target_encode(df, col, target, n_splits=5, smoothing=10):
    """
    Target Encoding（目标编码）
    使用k-fold防止穿越，smoothing防止过拟合稀疏类别
    """
    from sklearn.model_selection import KFold

    df = df.copy()
    global_mean = df[target].mean()
    df[f'{col}_te'] = global_mean  # 初始化

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(df):
        train_fold = df.iloc[train_idx]
        stats = train_fold.groupby(col)[target].agg(['mean', 'count'])
        # 平滑处理
        stats['smoothed'] = (
            (stats['mean'] * stats['count'] + global_mean * smoothing)
            / (stats['count'] + smoothing)
        )
        df.iloc[val_idx, df.columns.get_loc(f'{col}_te')] = (
            df.iloc[val_idx][col].map(stats['smoothed']).fillna(global_mean)
        )

    return df
```

### 4.4.2 职业/行业的风险分层

```python
# 基于业务经验的职业风险分层
OCCUPATION_RISK_MAP = {
    '公务员': 1,        # 低风险
    '教师': 1,
    '医生': 1,
    '工程师': 2,        # 中低风险
    '销售': 3,          # 中等风险
    '个体工商户': 3,
    '自由职业': 4,      # 高风险
    '无业': 5,          # 极高风险
}

df['occupation_risk'] = df['occupation'].map(OCCUPATION_RISK_MAP).fillna(3)
```

---

## 4.5 交叉特征

单一特征无法表达的风险模式，需要通过特征交叉来捕获：

```python
# 示例：收入与负债的交叉
df['income_debt_interaction'] = df['monthly_income'] * df['debt_burden_ratio']

# 示例：年龄与贷款期限的交叉
df['age_tenure_ratio'] = df['age'] / (df['loan_tenure_months'] + 1)

# 示例：设备风险与历史行为的交叉（欺诈信号叠加）
df['device_behavior_risk'] = df['device_risk_score'] * df['abnormal_behavior_flag']
```

**交叉特征的风险**：数量爆炸、过拟合、可解释性下降。需通过IV筛选或SHAP值控制。

---

## 4.6 文本与非结构化特征

### 4.6.1 借款用途文本特征

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def extract_text_features(df, text_col, n_components=10):
    """从借款用途等文本字段提取特征"""
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    text_matrix = tfidf.fit_transform(df[text_col].fillna(''))

    # 降维
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    text_features = svd.fit_transform(text_matrix)

    feat_df = pd.DataFrame(
        text_features,
        columns=[f'{text_col}_svd_{i}' for i in range(n_components)]
    )
    return feat_df
```

### 4.6.2 设备/行为序列特征

```python
# 申请行为序列：用户填写申请表的行为模式
# 例如：修改手机号次数、填写时长、滑动次数等

def extract_behavior_sequence_features(behavior_log):
    """
    从埋点日志中提取申请行为特征
    行为可能是欺诈识别的强信号
    """
    features = {
        'fill_duration_seconds': behavior_log['submit_time'] - behavior_log['start_time'],
        'phone_modify_count': behavior_log['phone_change_events'].apply(len),
        'id_modify_count': behavior_log['id_change_events'].apply(len),
        'copy_paste_count': behavior_log['paste_events'].apply(len),  # 粘贴次数（可能是填单工具）
        'page_back_count': behavior_log['back_events'].apply(len),
    }
    return pd.DataFrame(features)
```

---

## 4.7 特征选择

### 4.7.1 基于IV的初筛

```python
def iv_filter(iv_dict, threshold=0.02):
    """基于IV值筛选特征"""
    return {k: v for k, v in iv_dict.items() if v >= threshold}
```

### 4.7.2 相关性过滤

```python
def correlation_filter(df, features, threshold=0.8):
    """
    删除高度相关的特征（保留IV更高的那个）
    """
    corr_matrix = df[features].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = []
    for col in upper.columns:
        correlated = upper[col][upper[col] > threshold].index.tolist()
        to_drop.extend(correlated)

    return [f for f in features if f not in set(to_drop)]
```

### 4.7.3 基于模型的特征重要性

```python
import lightgbm as lgb
from sklearn.inspection import permutation_importance

def lgbm_feature_importance(X_train, y_train, X_val, y_val):
    """使用LightGBM评估特征重要性"""
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)])

    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance_gain': model.feature_importances_,
    }).sort_values('importance_gain', ascending=False)

    return model, importance_df
```

---

## 4.8 特征工程最佳实践

### 4.8.1 特征文档规范

每个进入模型的特征都应有完整文档：

```markdown
## 特征文档示例

**特征名**：`credit_query_cnt_m3`

**中文名**：近3个月征信查询次数

**计算逻辑**：统计观察点前90天内，人行征信报告中的机构查询记录数

**数据来源**：人行征信报告 → 查询记录表

**缺失含义**：无征信报告（即无贷款记录）→ 填充为0

**分布特征**：集中在0~5次，5次以上视为频繁查询

**风险逻辑**：查询次数越多 → 近期申贷行为频繁 → 资金紧张信号 → 坏率↑

**IV值**：0.15（中等预测力）

**单调性**：递增（查询次数越多，WOE越高）

**上线情况**：已上线，实时查询覆盖率约85%
```

### 4.8.2 特征稳定性检验

新特征上线前必须进行稳定性测试：

```python
def psi(expected, actual, bins=10):
    """
    计算PSI（Population Stability Index）
    expected: 训练集分布（基准）
    actual: 线上/测试集分布
    PSI < 0.1: 稳定
    0.1 <= PSI < 0.2: 轻微变化，需关注
    PSI >= 0.2: 显著漂移，需排查
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # 防止零值
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)

    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi_value
```

---

> **本章小结**：特征工程是风控建模中最需要领域知识的环节。WOE编码提供了模型可解释性，时序聚合特征捕捉行为趋势，而特征文档和稳定性监控保障了特征的工程化质量。好的特征工程体系，是团队建模效率的核心资产。
