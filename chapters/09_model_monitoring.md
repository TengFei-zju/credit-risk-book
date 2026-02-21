# 第九章 模型监控与迭代

## 9.1 为什么模型会衰退

模型上线后性能下降（Model Decay）是普遍现象，根本原因是**数据分布的动态变化**：

| 变化来源 | 典型案例 | 影响速度 |
|---------|---------|---------|
| 宏观经济变化 | 疫情、就业率变化 → 客群还款能力改变 | 缓慢（月级） |
| 业务策略变化 | 准入放松 → 新客群涌入 | 中等（周级） |
| 渠道结构变化 | 新增某渠道合作 → 客群结构改变 | 快（日级） |
| 数据源变化 | 某三方数据接口升级 → 特征含义改变 | 突发 |
| 欺诈对抗 | 欺诈团伙学会规避模型 | 快（天级） |

---

## 9.2 监控体系设计

完整的模型监控需要覆盖三层：

```
监控层次：
┌─────────────────────────────┐
│  L3: 业务结果监控            │  ← 慢（需要足够账龄）
│  逾期率、FPD、坏账率         │
├─────────────────────────────┤
│  L2: 模型输出监控            │  ← 中（即时可观测）
│  分数分布PSI、通过率、拒绝率  │
├─────────────────────────────┤
│  L1: 数据/特征监控           │  ← 快（数据问题第一层体现）
│  特征PSI、缺失率、覆盖率      │
└─────────────────────────────┘
```

**监控原则**：L1报警通常先于L3出现，是模型问题的早期预警。

---

## 9.3 特征监控

```python
class FeatureMonitor:
    """特征监控器：检测特征分布漂移"""

    def __init__(self, baseline_df, features, psi_threshold=0.1):
        """
        baseline_df: 训练期特征分布（基准）
        features: 待监控特征列表
        """
        self.baseline = baseline_df
        self.features = features
        self.psi_threshold = psi_threshold
        self._compute_baseline_stats()

    def _compute_baseline_stats(self):
        """计算基准分布"""
        self.baseline_stats = {}
        for feat in self.features:
            if self.baseline[feat].dtype in ['float64', 'int64']:
                self.baseline_stats[feat] = {
                    'type': 'numerical',
                    'percentiles': np.percentile(
                        self.baseline[feat].dropna(),
                        np.linspace(0, 100, 11)
                    ),
                    'missing_rate': self.baseline[feat].isnull().mean(),
                }
            else:
                self.baseline_stats[feat] = {
                    'type': 'categorical',
                    'value_counts': self.baseline[feat].value_counts(normalize=True).to_dict(),
                    'missing_rate': self.baseline[feat].isnull().mean(),
                }

    def monitor(self, current_df, report_date=None):
        """对当前期数据进行监控"""
        report = []

        for feat in self.features:
            baseline_stat = self.baseline_stats[feat]
            current_series = current_df[feat]

            if baseline_stat['type'] == 'numerical':
                psi = self._psi_numerical(
                    baseline_stat['percentiles'], current_series.dropna()
                )
            else:
                psi = self._psi_categorical(
                    baseline_stat['value_counts'], current_series
                )

            current_missing_rate = current_series.isnull().mean()
            missing_change = current_missing_rate - baseline_stat['missing_rate']

            status = 'ALERT' if psi > self.psi_threshold else 'OK'
            if abs(missing_change) > 0.1:
                status = 'ALERT'

            report.append({
                'feature': feat,
                'psi': round(psi, 4),
                'current_missing_rate': round(current_missing_rate, 4),
                'missing_rate_change': round(missing_change, 4),
                'status': status,
                'report_date': report_date or pd.Timestamp.today().date(),
            })

        report_df = pd.DataFrame(report)
        alert_features = report_df[report_df['status'] == 'ALERT']

        if len(alert_features) > 0:
            print(f"⚠️  {len(alert_features)} 个特征触发告警：")
            print(alert_features[['feature', 'psi', 'status']].to_string())

        return report_df

    def _psi_numerical(self, baseline_percentiles, current_values, eps=1e-6):
        breakpoints = np.unique(baseline_percentiles)
        exp_counts = np.diff(np.linspace(0, 1, len(breakpoints)))
        act_counts = np.histogram(current_values, bins=breakpoints)[0]
        act_pct = act_counts / len(current_values) + eps
        exp_pct = exp_counts + eps
        return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

    def _psi_categorical(self, baseline_dist, current_series, eps=1e-6):
        current_dist = current_series.value_counts(normalize=True).to_dict()
        categories = set(baseline_dist.keys()) | set(current_dist.keys())
        psi = 0
        for cat in categories:
            exp = baseline_dist.get(cat, eps)
            act = current_dist.get(cat, eps)
            psi += (act - exp) * np.log((act + eps) / (exp + eps))
        return psi
```

---

## 9.4 模型输出监控

```python
def monitor_score_distribution(baseline_scores, current_scores,
                                 model_name, report_date):
    """
    监控评分分布变化
    """
    psi, psi_detail = calculate_psi(baseline_scores, current_scores)

    # 关键分位数变化
    pctiles = [10, 25, 50, 75, 90]
    baseline_pcts = np.percentile(baseline_scores, pctiles)
    current_pcts = np.percentile(current_scores, pctiles)

    print(f"\n{'='*50}")
    print(f"模型：{model_name} | 日期：{report_date}")
    print(f"分数PSI：{psi:.4f}  {'⚠️ ALERT' if psi > 0.1 else '✅ OK'}")
    print(f"\n分位数对比：")
    for p, b, c in zip(pctiles, baseline_pcts, current_pcts):
        change = c - b
        print(f"  P{p:2d}: {b:.4f} → {c:.4f} ({change:+.4f})")

    # 通过率变化（假设固定切分点）
    CUTOFF = 0.10  # 10%违约概率为切分点
    baseline_pass = (baseline_scores <= CUTOFF).mean()
    current_pass = (current_scores <= CUTOFF).mean()
    print(f"\n通过率变化（阈值={CUTOFF}）：{baseline_pass:.2%} → {current_pass:.2%}")

    return {'psi': psi, 'status': 'ALERT' if psi > 0.1 else 'OK'}
```

---

## 9.5 业务指标回流监控

```python
def build_vintage_monitor(scoring_data, performance_data,
                           score_col='risk_score', label_col='is_bad',
                           date_col='score_date', age_col='loan_age'):
    """
    Vintage监控：实时跟踪各放款月份的坏率演变
    随着账龄增加，将实际坏率与历史同期对比
    """
    # 合并评分与表现数据
    merged = scoring_data.merge(performance_data, on='loan_id')

    # 按分档和账龄统计
    merged['score_band'] = pd.qcut(merged[score_col], q=5,
                                    labels=['低风险', '中低', '中等', '中高', '高风险'])

    vintage = merged.pivot_table(
        values=label_col,
        index=[date_col, 'score_band'],
        columns=age_col,
        aggfunc='mean'
    )

    return vintage

def early_warning_monitor(fpd_data, target_fpd=0.02, window_days=7):
    """
    FPD早期预警：首期逾期率异常升高
    FPD是最快速的风险信号，通常放款后30天即可观测
    """
    recent = fpd_data[fpd_data['loan_date'] >= pd.Timestamp.today() - pd.Timedelta(days=window_days)]
    current_fpd = recent['is_fpd'].mean()

    alert = current_fpd > target_fpd * 1.2  # 超过目标值20%触发告警
    print(f"近{window_days}天FPD：{current_fpd:.2%}  目标：{target_fpd:.2%}  "
          f"{'⚠️ ALERT' if alert else '✅ OK'}")

    return {'fpd': current_fpd, 'alert': alert}
```

---

## 9.6 模型迭代策略

### 9.6.1 触发迭代的条件

```markdown
自动触发迭代检查：
- 分数PSI > 0.2（显著漂移）
- KS 连续3个月下降，且累计降幅 > 5个百分点
- FPD 连续4周超过目标值1.5倍
- 主要特征 PSI > 0.25（数据源变化）

人工触发迭代：
- 业务策略发生重大调整（新渠道、新产品）
- 监管要求（如需增加公平性评估）
- 模型使用超过12个月（定期重训）
```

### 9.6.2 迭代方案选择

```
迭代方案决策树：

观测到模型性能下降
    ↓
特征PSI > 0.2?
    是 → 数据问题 → 查数据源变化 → 修复特征/替换特征 → 小规模重训
    否 ↓
分数PSI > 0.2?
    是 → 客群漂移 → 拒绝推断 + 新样本补充 → 完整重建
    否 ↓
仅KS下降，分布稳定?
    → 局部优化：新增特征/调整分箱 → 增量更新
```

### 9.6.3 在线增量学习（Online Learning）

对于需要快速响应变化的反欺诈场景：

```python
import lightgbm as lgb

class IncrementalModel:
    """
    LightGBM增量学习：在已有模型基础上用新数据继续训练
    适用于：欺诈模式快速迭代、客群快速变化
    """
    def __init__(self, base_model_path):
        self.model = lgb.Booster(model_file=base_model_path)

    def update(self, X_new, y_new, n_iter=50):
        """
        用新数据在已有模型基础上继续boosting
        注意：需要控制迭代次数，防止过拟合新数据
        """
        new_data = lgb.Dataset(X_new, label=y_new)

        params = {
            'learning_rate': 0.01,  # 增量学习用更小的学习率
            'n_estimators': n_iter,
        }

        self.model = lgb.train(
            params=params,
            train_set=new_data,
            init_model=self.model,    # 关键：从已有模型继续
            keep_training_booster=True,
        )
        return self.model
```

---

## 9.7 监控报表自动化

```python
# 每日自动监控报告模板
def generate_daily_report(model_config, feature_monitor, score_monitor,
                            business_monitor):
    """生成每日模型监控报告"""
    report_date = pd.Timestamp.today().date()

    report = {
        'report_date': str(report_date),
        'model_name': model_config['name'],
        'model_version': model_config['version'],

        # L1: 特征监控
        'feature_monitor': {
            'n_features_alert': len(feature_monitor[feature_monitor['status']=='ALERT']),
            'top_psi_features': feature_monitor.nlargest(5, 'psi')[['feature','psi']].to_dict('records'),
        },

        # L2: 分数监控
        'score_monitor': score_monitor,

        # L3: 业务指标（有延迟）
        'business_monitor': business_monitor,

        # 综合建议
        'recommendation': _generate_recommendation(
            feature_monitor, score_monitor, business_monitor
        ),
    }

    return report

def _generate_recommendation(feature_mon, score_mon, biz_mon):
    """基于监控结果自动生成处置建议"""
    alerts = []

    if score_mon.get('psi', 0) > 0.2:
        alerts.append('❗ 分数分布显著漂移，建议立即排查数据源和客群变化')
    elif score_mon.get('psi', 0) > 0.1:
        alerts.append('⚠️ 分数分布轻微漂移，加强监控')

    n_feat_alert = feature_mon.get('n_features_alert', 0) if isinstance(feature_mon, dict) else 0
    if n_feat_alert > 5:
        alerts.append(f'❗ {n_feat_alert}个特征告警，疑似数据源问题，建议暂停模型更新')

    if biz_mon.get('fpd_alert', False):
        alerts.append('❗ FPD超标，建议收紧策略并启动模型重评')

    if not alerts:
        alerts.append('✅ 所有指标正常，无需干预')

    return alerts
```

---

> **本章小结**：模型监控是保障风控系统长期有效性的最后一道防线。特征PSI是最早的预警信号，分数PSI是模型整体健康的快照，业务指标是最终的验金石。建立完整的三层监控体系，配合自动化报表，是规模化风控运营的必要能力。
