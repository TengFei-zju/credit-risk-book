# 第六章 机器学习建模

## 6.1 机器学习在风控中的定位

ML模型在风控中的角色已从"尝鲜"进化为"标配"，但需要明确其适用边界：

**适合ML的场景**：
- 特征维度高（>50个），交互关系复杂
- 行为评分（存量客户，可解释性要求相对低）
- 反欺诈（需要捕捉非线性、交互关系）
- 辅助模型/拒绝辅助决策

**仍需评分卡的场景**：
- 监管要求强可解释性（需提供拒绝原因）
- 特征维度少且业务逻辑清晰
- 机构内部稳健性优先于性能

---

## 6.2 梯度提升树（GBDT系列）

XGBoost/LightGBM/CatBoost 是信贷风控ML建模的主力模型。

### 6.2.1 LightGBM调参实践

```python
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import optuna

def lgbm_objective(trial, X, y):
    """Optuna超参数搜索目标函数"""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',

        # 树结构参数
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 300),

        # 学习率
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),

        # 正则化
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),

        # 采样（防过拟合）
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': 1,

        # 类别不平衡
        'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),
    }

    cv_results = lgb.cv(
        params, lgb.Dataset(X, label=y),
        nfold=5, stratified=True,
        num_boost_round=params['n_estimators'],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(-1)],
    )

    return max(cv_results['valid auc-mean'])

def tune_lgbm(X, y, n_trials=50):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: lgbm_objective(trial, X, y),
                   n_trials=n_trials, show_progress_bar=True)
    return study.best_params
```

### 6.2.2 风控建模的关键参数设置

```python
# 生产级LightGBM配置（风控场景）
lgbm_params = {
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],

    # 防止过拟合（风控数据样本量通常不大）
    'num_leaves': 31,           # 不宜过大
    'min_child_samples': 100,   # 每个叶节点至少100个样本
    'feature_fraction': 0.8,    # 列采样
    'bagging_fraction': 0.8,    # 行采样
    'bagging_freq': 5,

    # 学习率要小，配合早停
    'learning_rate': 0.05,
    'n_estimators': 2000,       # 配合early stopping使用

    # 类别不平衡
    'scale_pos_weight': 20,     # 根据坏率调整，约为 good/bad 比例

    # 稳定性优先
    'max_depth': 6,             # 控制树深度
    'lambda_l1': 0.1,           # L1正则
    'lambda_l2': 0.1,           # L2正则
}

# 训练
model = lgb.LGBMClassifier(**lgbm_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(100),
    ]
)
```

---

## 6.3 模型解释性工具

### 6.3.1 SHAP值分析

SHAP（SHapley Additive exPlanations）是当前最主流的模型解释方法：

```python
import shap

def shap_analysis(model, X_train, X_test, feature_names):
    """SHAP值分析"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # 如果是二分类，取坏客户的SHAP值
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # 1. 全局特征重要性（beeswarm图）
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                       plot_type='dot', max_display=20)

    # 2. 单样本解释（瀑布图）
    def explain_single(idx):
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[idx],
                base_values=explainer.expected_value,
                data=X_test.iloc[idx],
                feature_names=feature_names
            )
        )

    # 3. 特征依赖图
    def dependence_plot(feature):
        shap.dependence_plot(feature, shap_values, X_test,
                             feature_names=feature_names)

    return shap_values, explainer, explain_single

# SHAP值用于生成拒绝理由码
def generate_reject_reasons(shap_values_single, feature_names, top_n=3):
    """生成前N个拒绝原因"""
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values_single
    })
    # 正SHAP值=增加违约概率=拒绝原因
    top_reasons = shap_df[shap_df['shap_value'] > 0].nlargest(top_n, 'shap_value')
    return top_reasons
```

### 6.3.2 理由码体系

将SHAP值映射为业务可理解的拒绝原因：

```python
REASON_CODE_MAP = {
    'credit_query_cnt_m3': ('R01', '近期征信查询次数过多'),
    'max_overdue_months':  ('R02', '历史最高逾期期数较高'),
    'debt_to_income':      ('R03', '当前负债收入比过高'),
    'age':                 ('R04', '年龄不在申请范围内'),
    'loan_tenure':         ('R05', '申请期限不符合要求'),
}

def get_top_reject_reasons(shap_values, feature_names, reason_map, top_n=3):
    reasons = []
    sorted_idx = np.argsort(shap_values)[::-1]  # 按SHAP值降序
    for idx in sorted_idx:
        feat = feature_names[idx]
        if feat in reason_map and shap_values[idx] > 0:
            code, desc = reason_map[feat]
            reasons.append({'code': code, 'description': desc,
                            'shap': shap_values[idx]})
        if len(reasons) == top_n:
            break
    return reasons
```

---

## 6.4 模型融合

### 6.4.1 Stacking

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

def stacking_blend(base_models, X_train, y_train, X_test, n_folds=5):
    """
    Stacking融合：基模型预测结果作为元模型的输入
    """
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    train_meta = np.zeros((len(X_train), len(base_models)))
    test_meta = np.zeros((len(X_test), len(base_models)))

    for j, model in enumerate(base_models):
        test_preds = np.zeros((len(X_test), n_folds))
        for i, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            train_meta[val_idx, j] = model.predict_proba(
                X_train.iloc[val_idx])[:, 1]
            test_preds[:, i] = model.predict_proba(X_test)[:, 1]
        test_meta[:, j] = test_preds.mean(axis=1)

    # 元模型
    meta_model = LogisticRegression(C=1.0)
    meta_model.fit(train_meta, y_train)
    final_pred = meta_model.predict_proba(test_meta)[:, 1]

    return final_pred, meta_model

# 实践中更常用简单平均或加权平均
def weighted_blend(model_probs, weights=None):
    """加权平均融合"""
    probs = np.stack(model_probs, axis=1)
    if weights is None:
        weights = np.ones(len(model_probs)) / len(model_probs)
    return probs @ np.array(weights)
```

### 6.4.2 评分卡与ML模型融合

```python
# 常见方案：评分卡做准入门控，ML做精细化排序
def two_stage_decision(scorecard_score, ml_score,
                        sc_cutoff=550, ml_cutoff=0.15):
    """
    两阶段决策：
    1. 评分卡低于阈值直接拒绝（可解释）
    2. 评分卡通过的用ML精细排序
    """
    decisions = []
    for sc, ml in zip(scorecard_score, ml_score):
        if sc < sc_cutoff:
            decisions.append({'result': 'REJECT', 'reason': 'scorecard_reject'})
        elif ml > ml_cutoff:
            decisions.append({'result': 'REJECT', 'reason': 'ml_high_risk'})
        else:
            decisions.append({'result': 'APPROVE', 'reason': None})
    return decisions
```

---

## 6.5 神经网络在风控中的应用

深度学习在结构化数据上的表现并不总优于GBDT，但在以下场景有优势：

### 6.5.1 宽深模型（Wide & Deep）

```python
import torch
import torch.nn as nn

class WideDeep(nn.Module):
    """
    Wide & Deep 模型
    Wide部分：线性模型（可解释），处理稀疏特征
    Deep部分：MLP（非线性），处理稠密特征
    """
    def __init__(self, n_sparse_features, n_dense_features,
                 hidden_dims=[128, 64, 32]):
        super().__init__()

        # Wide部分（线性）
        self.wide = nn.Linear(n_sparse_features, 1)

        # Deep部分（MLP）
        layers = []
        in_dim = n_dense_features
        for out_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            ])
            in_dim = out_dim
        self.deep = nn.Sequential(*layers)
        self.deep_output = nn.Linear(in_dim, 1)

    def forward(self, x_sparse, x_dense):
        wide_out = self.wide(x_sparse)
        deep_out = self.deep_output(self.deep(x_dense))
        output = torch.sigmoid(wide_out + deep_out)
        return output.squeeze()
```

### 6.5.2 序列模型（行为数据）

```python
class BehaviorLSTM(nn.Module):
    """
    LSTM处理用户行为序列
    适用于：还款序列、APP使用序列
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])  # 取最后一层的hidden state
        return torch.sigmoid(out).squeeze()
```

---

## 6.6 建模中的时间序列问题

### 6.6.1 时间序列交叉验证

**不能**用随机交叉验证，必须用时序划分：

```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_cv(X, y, dates, n_splits=5):
    """
    时间序列交叉验证
    训练集始终在验证集之前（模拟真实上线场景）
    """
    # 按日期排序
    sort_idx = dates.argsort()
    X_sorted = X.iloc[sort_idx]
    y_sorted = y.iloc[sort_idx]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_sorted)):
        X_tr, X_val = X_sorted.iloc[train_idx], X_sorted.iloc[val_idx]
        y_tr, y_val = y_sorted.iloc[train_idx], y_sorted.iloc[val_idx]

        model = lgb.LGBMClassifier(**lgbm_params)
        model.fit(X_tr, y_tr)

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        scores.append(auc)
        print(f"Fold {fold+1}: AUC={auc:.4f}")

    print(f"\n平均 AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return scores
```

---

> **本章小结**：LightGBM是风控ML建模的首选，SHAP提供了向业务解释ML模型的语言，理由码体系使ML模型满足监管要求。神经网络在行为序列建模上有独特价值。时间序列交叉验证是评估风控模型的必要规范。
