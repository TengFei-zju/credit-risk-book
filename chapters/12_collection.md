# 第十二章 催收评分与贷后管理

## 12.1 贷后管理的业务目标

贷后管理的核心是在**有限资源下最大化回收**：

```
贷后管理目标：
1. 早期预警：提前识别风险客户，在逾期前干预
2. 催收分级：对逾期客户按回收可能性分级，优先分配资源
3. 策略优化：为不同客户选择最有效的催收策略（短信/电话/外访/法诉）
4. 损失最小化：在催收成本与回收金额之间找到最优平衡点
```

---

## 12.2 贷后模型体系

```
贷后模型谱系：
├── 早期预警模型（Early Warning Score）
│   目标：提前1~3个月预测将要逾期的客户
│   样本：当前正常还款的在贷客户
│
├── 催收评分（Collection Score）
│   目标：预测逾期客户在催收后的回款概率
│   样本：已逾期客户
│
├── 回收率预测（Recovery Rate）
│   目标：预测最终能回收多少（LGD建模的一部分）
│   样本：进入催收/核销的账户
│
└── 策略推荐模型（Treatment Assignment）
    目标：为每个客户推荐最优催收策略
    样本：历史催收结果数据
```

---

## 12.3 早期预警模型

### 12.3.1 标签定义与样本设计

```python
def define_early_warning_sample(loan_df, performance_window_months=3):
    """
    早期预警模型样本定义：
    对象：当前正常（无逾期）的在贷客户
    标签：未来3个月内发生M2+逾期
    """
    # 观察点：每月月末截面
    observation_dates = loan_df['snapshot_date'].unique()

    samples = []
    for obs_date in observation_dates:
        snapshot = loan_df[loan_df['snapshot_date'] == obs_date].copy()

        # 只取当前正常还款的客户
        normal_loans = snapshot[snapshot['current_dpd'] == 0]

        # 计算表现期结束日期
        perf_end = obs_date + pd.DateOffset(months=performance_window_months)

        # 关联未来的逾期表现
        future_perf = get_future_performance(normal_loans['loan_id'],
                                              obs_date, perf_end)

        # 标签：未来表现期内最大逾期期数 >= 2
        normal_loans['label'] = normal_loans['loan_id'].map(
            lambda lid: 1 if future_perf.get(lid, {}).get('max_dpd', 0) >= 2 else 0
        )

        samples.append(normal_loans)

    return pd.concat(samples, ignore_index=True)
```

### 12.3.2 早期预警特征

```python
early_warning_features = {
    # 当前账户状态
    'current_dpd': '当前逾期天数',
    'remaining_tenure': '剩余期数',
    'outstanding_balance': '未偿余额',
    'utilization_rate': '已用额度比例',

    # 历史还款行为（趋势信号最重要）
    'overdue_count_m3': '近3月逾期次数',
    'overdue_count_m6': '近6月逾期次数',
    'max_dpd_m12': '近12月最大逾期天数',
    'late_payment_trend': '还款延迟趋势（加速/减速）',
    'avg_days_to_pay': '平均还款延迟天数',

    # 外部信号
    'credit_query_delta': '近期征信查询增量',
    'new_credit_accounts': '近3月新增信贷账户数',
    'total_debt_change': '总负债变化率',

    # APP行为（如有）
    'app_active_days_m1': '近1月APP活跃天数',
    'login_frequency_change': '登录频率变化',
}
```

---

## 12.4 催收评分

### 12.4.1 催收评分的特殊性

```
催收评分 vs 申请评分的区别：

申请评分：预测客户是否会违约
催收评分：预测已违约客户是否会还款

两个完全不同的群体！
```

### 12.4.2 催收标签定义

```python
# 催收标签的多种定义方式
COLLECTION_LABELS = {
    # 定义1：是否有回款（二分类）
    'has_payment': lambda df: (df['collected_amount'] > 0).astype(int),

    # 定义2：是否全额还清（严格定义）
    'fully_paid': lambda df: (df['collected_amount'] >= df['overdue_amount']).astype(int),

    # 定义3：回收率（回归问题）
    'recovery_rate': lambda df: df['collected_amount'] / df['overdue_amount'].replace(0, 1),

    # 定义4：30天内是否还款（更强的时效要求）
    'paid_within_30d': lambda df: (
        (df['payment_date'] - df['due_date']).dt.days <= 30
    ).astype(int),
}
```

### 12.4.3 催收特征设计

```python
collection_features = {
    # 账户当前状态
    'dpd': '当前逾期天数',
    'overdue_amount': '逾期金额',
    'total_outstanding': '贷款总余额',
    'overdue_installments': '逾期期数',

    # 历史还款能力与意愿（关键区分维度）
    'ever_paid_late': '历史是否有过还款',  # 意愿信号
    'voluntary_payment_history': '主动还款比例',  # 意愿
    'partial_payment_count': '部分还款次数',  # 能力有限但有意愿

    # 催收接触记录
    'contact_count': '催收接触次数',
    'response_rate': '接听率',
    'promise_to_pay_flag': '是否承诺还款',
    'broken_promise_count': '承诺违约次数',

    # 客户背景
    'employment_status': '就业状态',
    'income_change_flag': '收入是否发生变化',
    'has_guarantor': '是否有担保人',

    # 外部信号
    'new_overdue_accounts': '近期新增逾期账户',  # 资金紧张信号
    'social_security_active': '社保是否仍在缴纳',  # 在职信号
}
```

---

## 12.5 催收策略优化

### 12.5.1 催收资源分配

```python
def collection_priority_assignment(loans_df, score_col, overdue_amount_col,
                                    cost_per_contact):
    """
    基于催收评分的优先级分配
    目标：在预算约束下最大化回收金额

    优先级 = 回收概率 × 逾期金额 - 催收成本
    """
    df = loans_df.copy()

    # 预期回收金额
    df['expected_recovery'] = df[score_col] * df[overdue_amount_col]

    # 净回收价值（扣除催收成本）
    df['net_value'] = df['expected_recovery'] - cost_per_contact

    # 按净价值排序，优先催收高价值账户
    df = df.sort_values('net_value', ascending=False)

    # 分档
    df['collection_tier'] = pd.qcut(df[score_col], q=5,
                                     labels=['T5-极低', 'T4-低', 'T3-中', 'T2-高', 'T1-极高'])

    # 策略映射
    strategy_map = {
        'T1-极高': 'sms_only',          # 高意愿，自动短信即可
        'T2-高': 'outbound_call',       # 主动电话催收
        'T3-中': 'intensive_call',      # 密集电话催收
        'T4-低': 'third_party',         # 转外部催收机构
        'T5-极低': 'legal_action',      # 法诉或核销
    }
    df['collection_strategy'] = df['collection_tier'].map(strategy_map)

    return df
```

### 12.5.2 最优催收策略（Treatment Effect）

不同客户对不同催收策略的响应不同，需要个性化匹配：

```python
# 因果推断：评估不同催收策略的增量效果
from sklearn.ensemble import GradientBoostingClassifier

class UpliftModel:
    """
    Uplift建模（增量效果建模）
    目标：找到"催了有用"的客户（而非"不催也会还"或"催了也没用"）
    """
    def __init__(self):
        self.model_treatment = GradientBoostingClassifier()
        self.model_control = GradientBoostingClassifier()

    def fit(self, X, y, treatment):
        """
        X: 客户特征
        y: 是否回款（1/0）
        treatment: 是否施加催收（1=催收组，0=对照组）
        """
        self.model_treatment.fit(X[treatment == 1], y[treatment == 1])
        self.model_control.fit(X[treatment == 0], y[treatment == 0])

    def predict_uplift(self, X):
        """
        预测Uplift（增量效果）
        Uplift = P(回款|催收) - P(回款|不催收)
        Uplift > 0: 催收有效（"说服型客户"）
        Uplift < 0: 催收适得其反（"报复型客户"）
        """
        prob_treatment = self.model_treatment.predict_proba(X)[:, 1]
        prob_control = self.model_control.predict_proba(X)[:, 1]
        return prob_treatment - prob_control
```

---

## 12.6 LGD建模

LGD（Loss Given Default，违约损失率）= 1 - 回收率

$$LGD = 1 - \frac{\text{回收金额}}{\text{违约时风险敞口（EAD）}}$$

```python
def build_lgd_model(recovery_data):
    """
    LGD建模
    目标变量：回收率（0~1之间的连续值）
    特殊性：分布在0和1处有大量堆积（两点质量分布）
    """
    # 回收率分布分析
    recovery_rate = recovery_data['collected_amount'] / recovery_data['ead']

    # 使用两步模型处理边界堆积：
    # Step 1: 预测是否有任何回收（二分类）
    # Step 2: 在有回收的样本上预测回收率（回归）

    has_recovery = (recovery_rate > 0).astype(int)

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingRegressor

    # Step 1: 分类模型
    clf = LogisticRegression()
    clf.fit(recovery_data[FEATURES], has_recovery)

    # Step 2: 回归模型（只用有回收的样本）
    has_recovery_data = recovery_data[recovery_rate > 0]
    reg = GradientBoostingRegressor()
    reg.fit(has_recovery_data[FEATURES], recovery_rate[recovery_rate > 0])

    # 最终预测：E(LGD) = P(无回收) × 1 + P(有回收) × (1 - E(回收率|有回收))
    def predict_lgd(X):
        p_recovery = clf.predict_proba(X)[:, 1]
        expected_recovery_rate = reg.predict(X)
        expected_lgd = (1 - p_recovery) * 1.0 + p_recovery * (1 - expected_recovery_rate)
        return np.clip(expected_lgd, 0, 1)

    return predict_lgd
```

---

## 12.7 贷后数据回流体系

贷后数据是持续改进贷前模型的宝贵资产：

```
贷后数据回流：

催收结果  → 更新训练标签 → 优化申请评分
还款模式  → 丰富行为特征 → 改进行为评分
欺诈暴露  → 黑名单更新  → 强化反欺诈规则
损失数据  → 更新LGD估计 → 改进定价模型
```

---

> **本章小结**：贷后管理是风险闭环的最后环节，也是数据最丰富的阶段。早期预警模型是资产质量管控的主动武器，催收评分提升资源利用效率，Uplift建模让催收策略个性化，LGD建模支撑风险定价。贷后数据的系统性回流，是持续改进整个风控体系的核心飞轮。
