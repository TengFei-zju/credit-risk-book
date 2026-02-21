# 第七章 模型评估与验证

## 7.1 风控模型评估框架

模型评估必须在**多个维度**同时进行，单一指标不足以全面衡量模型质量：

```
模型评估维度
├── 排序能力（区分好坏客户的能力）→ AUC、KS、GINI
├── 校准能力（预测概率的准确性）→ Calibration Curve、Brier Score
├── 稳定性（跨时间、跨人群的泛化）→ PSI、KS drift
├── 业务价值（对实际策略的贡献）→ Lift、通过率/坏率曲线
└── 公平性（对特定人群的影响）→ 分群差异分析
```

---

## 7.2 排序能力指标

### 7.2.1 KS统计量

$$KS = \max_t |F_{bad}(t) - F_{good}(t)|$$

KS是评分卡最常用的性能指标，衡量好坏客户评分分布的最大差距。

```python
from sklearn.metrics import roc_curve

def ks_stat(y_true, y_score):
    """计算KS统计量"""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ks = max(tpr - fpr)
    ks_threshold = thresholds[np.argmax(tpr - fpr)]
    return ks, ks_threshold

def ks_by_decile(y_true, y_score, n_deciles=10):
    """按分位数计算KS分档，输出KS表"""
    df = pd.DataFrame({'score': y_score, 'label': y_true})
    df['decile'] = pd.qcut(df['score'], q=n_deciles,
                            labels=False, duplicates='drop')
    df['decile'] = df['decile'].max() - df['decile']  # 翻转：分档1=最高分

    total_bad = df['label'].sum()
    total_good = (df['label'] == 0).sum()

    ks_table = df.groupby('decile').agg(
        total=('label', 'count'),
        bad=('label', 'sum'),
    ).reset_index()

    ks_table['good'] = ks_table['total'] - ks_table['bad']
    ks_table['bad_rate'] = ks_table['bad'] / ks_table['total']
    ks_table['cum_bad_pct'] = ks_table['bad'].cumsum() / total_bad
    ks_table['cum_good_pct'] = ks_table['good'].cumsum() / total_good
    ks_table['ks'] = (ks_table['cum_bad_pct'] - ks_table['cum_good_pct']).abs()

    print(f"模型KS: {ks_table['ks'].max():.4f}")
    return ks_table
```

**KS判断标准**（仅参考，不同业务场景差异大）：

| KS值 | 判断 |
|------|------|
| < 0.2 | 模型效果差 |
| 0.2~0.3 | 一般 |
| 0.3~0.4 | 较好 |
| 0.4~0.5 | 良好 |
| > 0.5 | 优秀（需验证是否穿越） |

### 7.2.2 AUC与GINI

$$AUC = P(score_{bad} > score_{good})$$
$$GINI = 2 \times AUC - 1$$

```python
from sklearn.metrics import roc_auc_score, RocCurveDisplay

def evaluate_model(y_true, y_score, model_name='Model'):
    """完整模型评估报告"""
    auc = roc_auc_score(y_true, y_score)
    ks, _ = ks_stat(y_true, y_score)
    gini = 2 * auc - 1

    print(f"{'='*40}")
    print(f"模型：{model_name}")
    print(f"AUC:  {auc:.4f}")
    print(f"GINI: {gini:.4f}")
    print(f"KS:   {ks:.4f}")
    print(f"{'='*40}")

    return {'auc': auc, 'ks': ks, 'gini': gini}
```

---

## 7.3 模型校准

**校准（Calibration）**：模型输出的概率值是否真实反映违约概率。

对于需要用概率值定价（利率=风险溢价+资金成本）的场景，校准至关重要。

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def plot_calibration(y_true, y_prob_list, model_names, n_bins=10):
    """绘制校准曲线"""
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot([0, 1], [0, 1], 'k--', label='完美校准')

    for y_prob, name in zip(y_prob_list, model_names):
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins
        )
        ax.plot(mean_predicted_value, fraction_of_positives,
                marker='o', label=name)

    ax.set_xlabel('预测概率（均值）')
    ax.set_ylabel('实际坏率')
    ax.set_title('校准曲线（越接近对角线越好）')
    ax.legend()
    return fig

# 校准方法
from sklearn.calibration import CalibratedClassifierCV

# Platt Scaling（适合小数据集）
calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
calibrated_model.fit(X_train, y_train)

# Isotonic Regression（适合大数据集）
calibrated_model_iso = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
```

---

## 7.4 稳定性指标

### 7.4.1 PSI（群体稳定性指标）

PSI用于监控模型输入分布（特征PSI）或模型输出分布（分数PSI）的稳定性：

```python
def calculate_psi(expected_scores, actual_scores, bins=10, eps=1e-6):
    """
    计算PSI
    expected_scores: 基准期分数（通常是训练集）
    actual_scores: 监控期分数（线上实际分数）
    """
    # 使用基准期分位点定义分箱边界
    breakpoints = np.percentile(expected_scores,
                                 np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    exp_counts = np.histogram(expected_scores, bins=breakpoints)[0]
    act_counts = np.histogram(actual_scores, bins=breakpoints)[0]

    exp_pct = exp_counts / len(expected_scores) + eps
    act_pct = act_counts / len(actual_scores) + eps

    psi_values = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    psi_total = psi_values.sum()

    psi_df = pd.DataFrame({
        'bin_lower': breakpoints[:-1],
        'bin_upper': breakpoints[1:],
        'expected_pct': exp_pct - eps,
        'actual_pct': act_pct - eps,
        'psi': psi_values
    })

    return psi_total, psi_df

# PSI阈值
# < 0.1: 稳定，无需干预
# 0.1~0.2: 轻微变化，加强监控
# > 0.2: 显著漂移，触发模型重评
```

### 7.4.2 跨时间验证（OOT）

OOT（Out-of-Time）验证是风控模型验证的黄金标准：

```python
def oot_validation(model, X_oot, y_oot, feature_names, date_col='apply_month'):
    """
    跨时间验证：按月计算模型性能
    """
    results = []
    for month in X_oot[date_col].unique():
        mask = X_oot[date_col] == month
        X_month = X_oot[mask].drop(columns=[date_col])
        y_month = y_oot[mask]

        if y_month.sum() < 5:  # 样本太少跳过
            continue

        y_pred = model.predict_proba(X_month)[:, 1]
        auc = roc_auc_score(y_month, y_pred)
        ks, _ = ks_stat(y_month, y_pred)

        results.append({
            'month': month,
            'n_samples': len(y_month),
            'bad_rate': y_month.mean(),
            'auc': auc,
            'ks': ks,
        })

    return pd.DataFrame(results)
```

---

## 7.5 业务价值评估

### 7.5.1 Lift曲线

```python
def lift_curve(y_true, y_score, n_bins=10):
    """
    Lift曲线：在前X%高风险客户中，坏客户捕获比例
    """
    df = pd.DataFrame({'score': y_score, 'label': y_true})
    df = df.sort_values('score', ascending=False).reset_index(drop=True)

    df['cum_bad'] = df['label'].cumsum()
    df['pct_population'] = (df.index + 1) / len(df)
    df['cum_bad_rate'] = df['cum_bad'] / df['label'].sum()

    # 随机线基准
    df['random_cum_bad_rate'] = df['pct_population']
    df['lift'] = df['cum_bad_rate'] / df['random_cum_bad_rate']

    return df

# 实践应用：前20%拒绝能捕获多少坏客户？
def rejection_analysis(y_true, y_score, reject_pcts=[0.1, 0.2, 0.3, 0.4]):
    """
    拒绝率与坏客户捕获分析
    """
    df = pd.DataFrame({'score': y_score, 'label': y_true})
    df = df.sort_values('score', ascending=False)

    results = []
    for reject_pct in reject_pcts:
        n_reject = int(len(df) * reject_pct)
        rejected = df.head(n_reject)
        approved = df.tail(len(df) - n_reject)

        results.append({
            'reject_rate': reject_pct,
            'reject_bad_capture': rejected['label'].sum() / df['label'].sum(),
            'approve_bad_rate': approved['label'].mean(),
            'reject_bad_rate': rejected['label'].mean(),
        })

    return pd.DataFrame(results)
```

### 7.5.2 通过率-坏率曲线

```python
def pass_rate_bad_rate_curve(y_true, y_score, n_points=20):
    """
    通过率-坏率曲线：核心业务决策工具
    横轴：通过率（提高门槛=降低通过率）
    纵轴：通过客户中的坏率
    """
    thresholds = np.percentile(y_score,
                                np.linspace(0, 100, n_points))
    results = []

    for threshold in thresholds:
        approved = y_score <= threshold  # 低风险分数=通过
        results.append({
            'threshold': threshold,
            'pass_rate': approved.mean(),
            'approve_bad_rate': y_true[approved].mean() if approved.sum() > 0 else 0,
        })

    return pd.DataFrame(results)
```

---

## 7.6 公平性评估

监管对"算法歧视"的关注日益增加，需要确保模型对特定群体无系统性偏差。

```python
def fairness_audit(y_true, y_score, sensitive_attr, groups=None):
    """
    公平性审计：分析模型对不同群体的差异
    sensitive_attr: 敏感属性（性别、年龄段、地区等）
    """
    if groups is None:
        groups = sensitive_attr.unique()

    results = []
    overall_auc = roc_auc_score(y_true, y_score)

    for group in groups:
        mask = sensitive_attr == group
        if mask.sum() < 30:
            continue

        group_auc = roc_auc_score(y_true[mask], y_score[mask])
        group_ks, _ = ks_stat(y_true[mask], y_score[mask])

        # 计算同等分数下的通过率差异
        median_score = np.median(y_score)
        pass_rate = (y_score[mask] <= median_score).mean()

        results.append({
            'group': group,
            'n_samples': mask.sum(),
            'bad_rate': y_true[mask].mean(),
            'auc': group_auc,
            'ks': group_ks,
            'pass_rate_at_median': pass_rate,
            'auc_gap_vs_overall': group_auc - overall_auc,
        })

    return pd.DataFrame(results)
```

---

## 7.7 模型验证报告规范

完整的模型验证报告应包含：

```markdown
## 模型验证报告 - 申请评分卡 v3.2

### 执行摘要
- 模型类型：逻辑回归评分卡
- 验证结论：✅ 通过验证，建议上线
- 关键风险点：OOT最近2个月KS轻微下降（42.1→39.3），需上线后重点监控

### 数据集概况
| 数据集 | 时间范围 | 样本量 | 坏率 |
|--------|---------|--------|------|
| 训练集 | 22Q1~23Q1 | 85,000 | 4.2% |
| 验证集（OOS）| 22Q1~23Q1（随机30%）| 36,000 | 4.1% |
| 时间外验证（OOT）| 23Q2~23Q4 | 28,000 | 4.8% |

### 性能指标
[KS、AUC、GINI表格]

### 稳定性
[PSI表格、OOT按月性能]

### 分群验证
[按产品线、渠道、地区的分群验证结果]

### 公平性审计
[按性别、年龄段的差异分析]

### 上线建议
- 切分点：580分（对应通过率60%，预期坏率3.1%）
- 监控方案：每周计算分数PSI，每月OOT验证
- 触发重建条件：PSI>0.2 或 KS连续2月下降>5个百分点
```

---

> **本章小结**：风控模型的评估不能只看AUC。KS、PSI、OOT验证、业务Lift是同等重要的指标。公平性审计在监管趋严的背景下已不可忽视。完整的验证报告是模型上线决策的依据，也是后续追责的证据链。
