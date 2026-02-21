# 第十章 策略设计与决策引擎

## 10.1 策略与模型的关系

模型输出的是**风险概率**，而业务决策需要的是**是/否通过**、**额度是多少**、**利率是多少**。

策略（Policy）是将模型分值转化为业务行动的规则体系。

```
模型层 → 策略层 → 决策层 → 执行层
风险评分  分档/切分点  综合规则  放款/拒绝/人工审核
```

**重要原则**：模型工程师需要理解策略设计，但策略的最终拍板权属于策略团队（风控经理）。算法工程师的职责是：
1. 准确传达模型性能边界
2. 量化不同策略对风险和业务的影响
3. 提供数据支撑帮助策略决策

---

## 10.2 评分分档设计

### 10.2.1 分档方法

```python
def design_score_bands(scores, labels, method='business_driven',
                         n_bands=5, target_bad_rates=None):
    """
    评分分档设计
    method:
        'equal_frequency': 等频分箱（每档人数相同）
        'equal_width': 等距分箱
        'business_driven': 业务驱动（指定目标坏率）
        'tree_based': 基于决策树的最优分箱
    """
    df = pd.DataFrame({'score': scores, 'label': labels})

    if method == 'equal_frequency':
        df['band'] = pd.qcut(df['score'], q=n_bands, labels=False)

    elif method == 'business_driven' and target_bad_rates:
        # 根据目标坏率确定切分点
        cutoffs = []
        for target_rate in target_bad_rates:
            # 二分查找满足目标坏率的评分切分点
            threshold = find_threshold_for_bad_rate(scores, labels, target_rate)
            cutoffs.append(threshold)
        df['band'] = pd.cut(df['score'], bins=[-np.inf] + sorted(cutoffs) + [np.inf],
                             labels=False)

    # 统计各档结果
    band_stats = df.groupby('band').agg(
        count=('label', 'count'),
        bad=('label', 'sum'),
    ).reset_index()
    band_stats['bad_rate'] = band_stats['bad'] / band_stats['count']
    band_stats['pct_of_total'] = band_stats['count'] / len(df)

    return band_stats

def find_threshold_for_bad_rate(scores, labels, target_bad_rate, tolerance=0.001):
    """二分查找：找到使通过客户坏率不超过target的评分阈值"""
    lo, hi = scores.min(), scores.max()
    for _ in range(50):
        mid = (lo + hi) / 2
        approved = labels[scores <= mid]
        if len(approved) == 0:
            break
        current_rate = approved.mean()
        if abs(current_rate - target_bad_rate) < tolerance:
            return mid
        elif current_rate > target_bad_rate:
            hi = mid
        else:
            lo = mid
    return mid
```

### 10.2.2 分档可视化与决策支撑

```python
def band_analysis_table(band_stats, total_bad_rate):
    """
    生成用于策略讨论的分档分析表
    """
    band_stats = band_stats.copy()
    band_stats['cum_approval_rate'] = band_stats['pct_of_total'].cumsum()
    band_stats['bad_rate_vs_avg'] = band_stats['bad_rate'] / total_bad_rate
    band_stats['lift'] = band_stats['bad_rate'] / total_bad_rate

    # 标注策略建议
    def suggest_action(row):
        if row['bad_rate'] < total_bad_rate * 0.5:
            return '✅ 建议通过'
        elif row['bad_rate'] < total_bad_rate * 1.5:
            return '🔶 可配合其他条件'
        else:
            return '❌ 建议拒绝'

    band_stats['suggested_action'] = band_stats.apply(suggest_action, axis=1)
    return band_stats
```

---

## 10.3 多维度策略矩阵

单一评分不够，风控策略通常是**多维度交叉**的：

```python
def strategy_matrix_analysis(df, score_col, segment_col,
                               score_bins=5, target='label'):
    """
    策略矩阵分析：评分 × 业务分层
    例如：申请评分 × 渠道/产品/客群
    """
    df['score_band'] = pd.qcut(df[score_col], q=score_bins,
                                labels=[f'S{i+1}' for i in range(score_bins)])

    matrix = df.pivot_table(
        values=target,
        index='score_band',
        columns=segment_col,
        aggfunc='mean'
    ).round(4)

    # 数量矩阵
    count_matrix = df.pivot_table(
        values=target,
        index='score_band',
        columns=segment_col,
        aggfunc='count'
    )

    print("坏率矩阵：")
    print(matrix)
    print("\n样本量矩阵：")
    print(count_matrix)

    return matrix, count_matrix
```

### 10.3.1 双评分卡矩阵

结合两个模型（如申请评分+反欺诈评分）做联合策略：

```python
def dual_score_strategy(apply_score, fraud_score,
                          credit_cutoffs=(500, 550, 600),
                          fraud_cutoffs=(0.1, 0.3)):
    """
    双评分矩阵策略
    返回：APPROVE / MANUAL_REVIEW / REJECT
    """
    decisions = []
    for cs, fs in zip(apply_score, fraud_score):
        if fs > fraud_cutoffs[1]:        # 高欺诈风险，直接拒绝
            decisions.append('REJECT_FRAUD')
        elif fs > fraud_cutoffs[0]:      # 中欺诈风险
            if cs < credit_cutoffs[0]:
                decisions.append('REJECT')
            else:
                decisions.append('MANUAL_REVIEW')
        else:                            # 低欺诈风险
            if cs >= credit_cutoffs[2]:
                decisions.append('APPROVE')
            elif cs >= credit_cutoffs[1]:
                decisions.append('APPROVE_CONDITIONAL')  # 有条件通过（低额度）
            elif cs >= credit_cutoffs[0]:
                decisions.append('MANUAL_REVIEW')
            else:
                decisions.append('REJECT')

    return decisions
```

---

## 10.4 额度策略

### 10.4.1 基于风险的差异化定价

```python
def risk_based_pricing(probability_of_default, lgd=0.6, cost_of_fund=0.05,
                         target_roi=0.02):
    """
    基于风险的利率定价（简化版）
    利率 = 资金成本 + 违约损失 + 目标回报

    probability_of_default: 违约概率（PD）
    lgd: 违约损失率（Loss Given Default）
    cost_of_fund: 资金成本
    target_roi: 目标收益率
    """
    expected_loss = probability_of_default * lgd
    required_rate = cost_of_fund + expected_loss + target_roi

    # 受监管利率上限约束
    MAX_RATE = 0.24  # 年化24%（参考监管红线）
    actual_rate = min(required_rate, MAX_RATE)

    return {
        'pd': probability_of_default,
        'expected_loss_rate': expected_loss,
        'required_rate': required_rate,
        'actual_rate': actual_rate,
        'is_viable': required_rate <= MAX_RATE,  # 若要求利率超过上限，则该客户不可做
    }

def credit_limit_strategy(base_limit, risk_score, income,
                            debt_burden_ratio, policy_matrix):
    """
    额度策略：综合考虑风险评分、收入、负债情况
    """
    # 基础额度 = 收入倍数 × 风险系数
    income_multiple = policy_matrix.get_income_multiple(risk_score)
    base = income * income_multiple

    # 负债调整
    if debt_burden_ratio > 0.5:
        base *= 0.7  # 高负债打折

    # 风险分档调整
    if risk_score < 500:
        base = min(base, 5000)     # 高风险额度上限
    elif risk_score < 600:
        base = min(base, 20000)

    return round(base / 1000) * 1000  # 取整到千元
```

---

## 10.5 决策引擎基础

### 10.5.1 规则引擎的本质

决策引擎是将规则（Rule）+ 模型（Model）+ 策略（Policy）整合为自动化决策的系统：

```
决策引擎核心组件：
├── 硬性规则（Hard Rules）：准入/拒绝的绝对条件
│   ├── 黑名单核查
│   ├── 反洗钱规则
│   └── 监管硬性要求（年龄、资质等）
├── 软性规则（Soft Rules）：基于模型分数的条件规则
│   ├── 评分切分点
│   ├── 多条件组合规则
│   └── 人工审核触发条件
└── 策略矩阵（Strategy Matrix）
    ├── 额度策略
    ├── 利率定价策略
    └── 期限策略
```

```python
class DecisionEngine:
    """简化版决策引擎"""

    def __init__(self, hard_rules, model_scores, strategy_matrix):
        self.hard_rules = hard_rules
        self.model_scores = model_scores
        self.strategy_matrix = strategy_matrix

    def decide(self, applicant_id, applicant_features):
        decision = {'loan_id': applicant_id, 'steps': []}

        # 第一步：硬性规则
        hard_result = self._check_hard_rules(applicant_features)
        decision['steps'].append(hard_result)
        if hard_result['result'] == 'REJECT':
            decision.update({'final': 'REJECT', 'reason': hard_result['reason']})
            return decision

        # 第二步：模型评分
        credit_score = self.model_scores['credit'](applicant_features)
        fraud_score = self.model_scores['fraud'](applicant_features)
        decision['steps'].append({'credit_score': credit_score, 'fraud_score': fraud_score})

        # 第三步：策略决策
        policy_result = self.strategy_matrix.lookup(credit_score, fraud_score,
                                                      applicant_features.get('channel'))
        decision['steps'].append(policy_result)

        # 第四步：额度与利率
        if policy_result['result'] == 'APPROVE':
            credit_limit = self._calculate_limit(credit_score, applicant_features)
            rate = self._calculate_rate(credit_score)
            decision.update({
                'final': 'APPROVE',
                'credit_limit': credit_limit,
                'interest_rate': rate,
            })
        elif policy_result['result'] == 'MANUAL':
            decision.update({'final': 'MANUAL_REVIEW', 'priority': policy_result.get('priority', 'normal')})
        else:
            decision.update({'final': 'REJECT', 'reason': policy_result.get('reason', 'policy_reject')})

        return decision

    def _check_hard_rules(self, features):
        """硬性规则检查"""
        if features.get('is_blacklist', False):
            return {'result': 'REJECT', 'reason': 'blacklist'}
        if features.get('age', 25) < 18 or features.get('age', 25) > 70:
            return {'result': 'REJECT', 'reason': 'age_out_of_range'}
        if features.get('is_sanctions', False):
            return {'result': 'REJECT', 'reason': 'sanctions'}
        return {'result': 'PASS'}

    def _calculate_limit(self, score, features):
        income = features.get('monthly_income', 5000)
        if score > 700:
            limit = income * 6
        elif score > 600:
            limit = income * 4
        else:
            limit = income * 2
        return min(limit, 200000)  # 上限20万

    def _calculate_rate(self, score):
        if score > 700: return 0.10
        elif score > 600: return 0.15
        else: return 0.20
```

---

## 10.6 策略效果评估

策略上线后，需要持续评估策略效果（区别于模型效果评估）：

```python
def strategy_effectiveness_report(pre_strategy_df, post_strategy_df):
    """
    策略调整前后对比分析
    """
    metrics = {}

    for period, df in [('调整前', pre_strategy_df), ('调整后', post_strategy_df)]:
        metrics[period] = {
            'pass_rate': df['is_approved'].mean(),
            'auto_approve_rate': (df['decision'] == 'APPROVE').mean(),
            'manual_rate': (df['decision'] == 'MANUAL_REVIEW').mean(),
            'reject_rate': (df['decision'] == 'REJECT').mean(),
            'avg_credit_limit': df[df['is_approved']]['credit_limit'].mean(),
            'avg_interest_rate': df[df['is_approved']]['interest_rate'].mean(),
        }

    comparison = pd.DataFrame(metrics)
    print(comparison)
    return comparison
```

---

> **本章小结**：策略设计是模型价值落地的关键环节。评分分档、多维度矩阵、基于风险的定价，将模型的抽象分数转化为具体的业务决策。决策引擎是策略的工程化载体，算法工程师需要理解其运作机制，才能设计出真正可落地的模型方案。
