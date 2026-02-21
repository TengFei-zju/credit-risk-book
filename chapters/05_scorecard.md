# 第五章 评分卡建模

## 5.1 评分卡的地位与价值

评分卡（Scorecard）在信贷风控中有着不可替代的地位，尽管机器学习模型的性能通常更优：

| 维度 | 评分卡 | 机器学习模型 |
|------|--------|------------|
| 可解释性 | 极强（每个变量贡献可量化） | 较弱（需SHAP辅助） |
| 监管合规 | 天然满足拒绝原因说明要求 | 需要额外工作 |
| 稳定性 | 强（分箱机制天然防漂移） | 较弱 |
| 性能上限 | 中等（线性假设限制） | 高 |
| 开发周期 | 中等 | 较短（模型部分），较长（解释部分） |
| 维护成本 | 低 | 中等 |

**实践建议**：申请评分卡仍是大多数持牌机构的主流选择，机器学习模型更多作为辅助或行为评分使用。

---

## 5.2 标准评分卡开发流程

```
数据准备 → 变量筛选 → WOE分箱 → 逻辑回归 → 刻度转换 → 验证 → 上线
```

---

## 5.3 逻辑回归建模

评分卡的模型核心是逻辑回归（Logistic Regression），使用WOE编码后的特征。

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def build_scorecard_lr(X_woe_train, y_train, X_woe_val, y_val,
                        C_values=[0.01, 0.05, 0.1, 0.5, 1.0]):
    """
    训练评分卡逻辑回归模型
    X_woe_*: 已经WOE编码的特征矩阵
    """
    from sklearn.model_selection import cross_val_score

    best_C, best_score = None, 0
    for C in C_values:
        lr = LogisticRegression(C=C, max_iter=1000,
                                class_weight='balanced', solver='lbfgs')
        scores = cross_val_score(lr, X_woe_train, y_train,
                                  cv=5, scoring='roc_auc')
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_C = C

    # 用最优C重新训练
    final_model = LogisticRegression(C=best_C, max_iter=1000,
                                      class_weight='balanced', solver='lbfgs')
    final_model.fit(X_woe_train, y_train)

    print(f"最优 C={best_C}, CV AUC={best_score:.4f}")
    return final_model
```

### 5.3.1 评分卡逻辑回归的特殊要求

1. **系数符号检验**：每个变量的系数方向必须与WOE单调性一致
2. **多重共线性检验**：VIF（方差膨胀因子）< 5

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_vif(X_woe):
    """检验多重共线性"""
    vif_data = pd.DataFrame({
        'feature': X_woe.columns,
        'VIF': [variance_inflation_factor(X_woe.values, i)
                for i in range(X_woe.shape[1])]
    })
    return vif_data.sort_values('VIF', ascending=False)

# 逐步删除VIF > 5的变量，每次删一个，重新计算
```

---

## 5.4 评分刻度转换

将逻辑回归输出的log-odds转换为直觉易懂的评分：

**标准公式**：

$$Score = A - B \times \ln(odds)$$

其中 $odds = P(bad)/P(good)$，参数A和B由以下约定确定：
- 基准分（如600分）对应基准odds（如1:50）
- 每翻一倍odds，分数减少PDO分（如20分）

```python
def logodds_to_score(log_odds, base_score=600, base_odds=1/50, pdo=20):
    """
    将log-odds转换为评分
    base_score: 基准分（对应base_odds时的分数）
    base_odds: 基准好坏比（good:bad）
    pdo: Points to Double the Odds（好坏比翻倍对应的分数变化）
    """
    B = pdo / np.log(2)
    A = base_score + B * np.log(base_odds)
    score = A - B * log_odds
    return score

def build_scorecard_table(model, feature_names, woe_tables,
                           base_score=600, base_odds=1/50, pdo=20):
    """
    生成评分卡明细表
    每个变量的每个分箱对应一个分值
    """
    B = pdo / np.log(2)
    A = base_score + B * np.log(base_odds)

    # 截距对应的基准分
    n_features = len(feature_names)
    intercept_score = A - B * model.intercept_[0]
    base_per_feature = intercept_score / n_features  # 每个变量均分截距

    scorecard = []
    for i, feature in enumerate(feature_names):
        coef = model.coef_[0][i]
        woe_table = woe_tables[feature]  # 该特征的WOE分箱表

        for _, row in woe_table.iterrows():
            points = round(base_per_feature - B * coef * row['woe'])
            scorecard.append({
                'feature': feature,
                'bin': row['bin'],
                'woe': row['woe'],
                'points': points,
            })

    return pd.DataFrame(scorecard)
```

### 5.4.1 评分卡明细表示例

| 变量 | 分箱 | WOE | 分值 |
|------|------|-----|------|
| 近3月征信查询次数 | 缺失 | -0.82 | 15 |
| 近3月征信查询次数 | 0次 | -0.65 | 12 |
| 近3月征信查询次数 | 1~2次 | 0.12 | 5 |
| 近3月征信查询次数 | 3~5次 | 0.48 | -2 |
| 近3月征信查询次数 | 6次以上 | 1.25 | -18 |
| 月收入（元） | <3000 | 0.95 | -14 |
| 月收入（元） | 3000~8000 | 0.10 | 4 |
| 月收入（元） | 8000~20000 | -0.35 | 10 |
| 月收入（元） | >20000 | -0.72 | 18 |

---

## 5.5 评分分布与切分点设计

```python
import matplotlib.pyplot as plt

def plot_score_distribution(scores, labels, bins=30):
    """绘制好坏客户评分分布"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：分布叠加
    axes[0].hist(scores[labels == 0], bins=bins, alpha=0.6, label='好客户', color='green')
    axes[0].hist(scores[labels == 1], bins=bins, alpha=0.6, label='坏客户', color='red')
    axes[0].set_xlabel('评分')
    axes[0].set_ylabel('频次')
    axes[0].legend()
    axes[0].set_title('好坏客户评分分布')

    # 右图：各分段坏率
    score_bins = pd.cut(scores, bins=10)
    bad_rate_by_bin = pd.DataFrame({'score_bin': score_bins, 'label': labels})
    bad_rate_by_bin = bad_rate_by_bin.groupby('score_bin', observed=True)['label'].mean()

    axes[1].bar(range(len(bad_rate_by_bin)), bad_rate_by_bin.values, color='steelblue')
    axes[1].set_xticks(range(len(bad_rate_by_bin)))
    axes[1].set_xticklabels([str(x) for x in bad_rate_by_bin.index], rotation=45, ha='right')
    axes[1].set_ylabel('坏率')
    axes[1].set_title('各分段坏率')

    plt.tight_layout()
    return fig

def design_cutoffs(scores, labels, target_pass_rates=[0.4, 0.5, 0.6, 0.7]):
    """
    根据目标通过率设计切分点
    """
    results = []
    for target_rate in target_pass_rates:
        cutoff = np.percentile(scores, (1 - target_rate) * 100)
        passed = scores >= cutoff
        pass_bad_rate = labels[passed].mean()
        reject_bad_rate = labels[~passed].mean()
        results.append({
            'target_pass_rate': target_rate,
            'cutoff_score': round(cutoff),
            'actual_pass_rate': passed.mean(),
            'pass_bad_rate': pass_bad_rate,
            'reject_bad_rate': reject_bad_rate,
        })
    return pd.DataFrame(results)
```

---

## 5.6 评分卡的分层与多模型架构

单一评分卡难以覆盖所有客群，实践中常用**分层建模**：

```
全量客户
├── 有征信记录客户 → 标准申请评分卡（使用征信特征）
├── 无征信记录客户 → 无征信评分卡（使用替代数据）
│   ├── 年轻客群（<25岁） → 专属模型
│   └── 其他新客 → 通用无征信模型
└── 内部历史客户 → 行为评分卡（使用历史还款数据）
```

**分层的收益**：
- 不同客群的风险驱动因素不同
- 避免多数客群的模式淹没少数客群的信号
- 各层模型可独立迭代

---

## 5.7 拒绝推断（Reject Inference）

**问题**：训练数据只有被批准的客户的表现记录，被拒绝的客户缺乏标签，导致样本选择偏差（Selection Bias）。

```
贷款申请人
├── 批准（有Y标签）→ 用于训练
└── 拒绝（无Y标签）→ 被排除，但实际上线时要对所有申请人打分！
```

### 5.7.1 拒绝推断方法

```python
# 方法一：粗暴填充（Hard Cutoff）
# 假设被拒绝的人全是坏客户（过于保守）
reject_df['label'] = 1

# 方法二：概率填充（Fuzzy Augmentation）
# 用已有模型预测被拒绝客户的坏概率，然后按概率加权采样
def fuzzy_augmentation(approved_df, rejected_df, model, bad_multiplier=2.0):
    """
    拒绝推断：概率填充法
    """
    rejected_features = rejected_df.drop(columns=['label'])
    reject_prob = model.predict_proba(rejected_features)[:, 1]

    # 放大拒绝人群的坏概率（因为他们本来就是高风险）
    reject_prob_adjusted = np.clip(reject_prob * bad_multiplier, 0, 1)

    # 按概率随机分配标签
    np.random.seed(42)
    rejected_df['label'] = (np.random.rand(len(rejected_df)) < reject_prob_adjusted).astype(int)

    # 合并
    combined = pd.concat([approved_df, rejected_df], ignore_index=True)
    return combined

# 方法三：半监督学习
# 将拒绝样本作为无标签数据，使用Label Propagation等算法推断标签
```

**实践中的态度**：拒绝推断的效果有限且方法论争议大。更务实的做法是：
1. 在新模型上线时做小比例放量（如随机放5%），获取真实坏样本
2. 定期更新模型，逐渐纠正选择偏差

---

## 5.8 评分卡的文档规范

一张生产级评分卡需要完整的文档，包括：

```markdown
## 申请评分卡 v3.2 模型说明书

### 1. 基本信息
- 适用产品：消费信贷（无抵押）
- 适用人群：有征信记录客户（人行征信查询次数≥1）
- 上线日期：2024-03-01
- 建模人：XXX
- 审批人：XXX

### 2. 标签定义
- 坏：贷款存续期内M3+逾期（任意一期逾期90天以上）
- 好：贷款结清且无逾期记录
- 不确定：不纳入样本

### 3. 样本信息
- 建模样本：2022Q1~2023Q1 放款，表现期12个月
- 训练集：70%（时间序列随机切分）
- 验证集：30%（OOT，2023Q2~2023Q4）
- 训练坏率：4.2%
- 验证坏率：4.8%（合理漂移）

### 4. 模型性能
| 数据集 | KS | AUC | PSI |
|--------|----|----|-----|
| 训练集 | 42.3 | 0.78 | - |
| 验证集 | 39.7 | 0.75 | 0.03 |
| OOT | 38.1 | 0.74 | 0.05 |

### 5. 最终入模变量（共12个）
略...
```

---

> **本章小结**：评分卡不是过时的技术，而是风控行业经过数十年验证的工程化范式。掌握WOE分箱、刻度转换、拒绝推断，理解评分卡背后的统计学原理，是成为一名合格风控算法工程师的基础。
