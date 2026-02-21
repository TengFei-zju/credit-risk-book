# 第八章 模型上线与工程化

## 8.1 从实验到生产的鸿沟

模型工程化是算法工程师最容易忽视却最影响实际价值的环节。

```
模型生命周期：
实验（Notebook）→ 工程化（Service）→ 上线（Production）→ 监控（Operations）
```

**常见的工程化陷阱**：
- 本地跑通，线上报错（依赖版本不一致）
- 离线特征与线上特征计算逻辑不同（Training-Serving Skew）
- 没有降级方案，服务超时直接影响通过率
- 模型版本没有追踪，无法回滚

---

## 8.2 模型序列化与版本管理

### 8.2.1 模型保存规范

```python
import pickle
import joblib
import mlflow
import json
from datetime import datetime

def save_model_artifact(model, feature_names, preprocessing_pipeline,
                         model_meta, save_dir):
    """
    标准化保存模型产物
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    artifact_path = f"{save_dir}/model_{timestamp}"

    import os
    os.makedirs(artifact_path, exist_ok=True)

    # 1. 保存模型
    joblib.dump(model, f"{artifact_path}/model.pkl")

    # 2. 保存预处理管道（必须与模型一起保存！）
    joblib.dump(preprocessing_pipeline, f"{artifact_path}/pipeline.pkl")

    # 3. 保存特征列表（线上取数的依据）
    with open(f"{artifact_path}/features.json", 'w') as f:
        json.dump({'features': feature_names}, f, ensure_ascii=False, indent=2)

    # 4. 保存模型元信息
    model_meta.update({
        'saved_at': timestamp,
        'model_type': type(model).__name__,
        'n_features': len(feature_names),
    })
    with open(f"{artifact_path}/meta.json", 'w') as f:
        json.dump(model_meta, f, ensure_ascii=False, indent=2)

    print(f"模型已保存至: {artifact_path}")
    return artifact_path

# MLflow跟踪（推荐用于团队协作）
def mlflow_log_model(model, params, metrics, feature_names, model_name):
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict({'features': feature_names}, 'features.json')
        mlflow.sklearn.log_model(model, model_name)
```

### 8.2.2 模型版本策略

```
模型版本命名规范：{产品线}_{模型类型}_{主版本}.{子版本}
例如：consumer_apply_scorecard_v3.2
      consumer_behavior_lgbm_v1.0

版本升级规则：
- 子版本（.x）：特征微调、参数调整，无结构性变化
- 主版本（vX）：重大重建（新标签定义/样本重构/模型架构变更）
```

---

## 8.3 模型服务化

### 8.3.1 FastAPI部署

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import json
import time
import logging

logger = logging.getLogger(__name__)

# 加载模型（启动时加载，避免每次请求重新加载）
MODEL = joblib.load('model.pkl')
PIPELINE = joblib.load('pipeline.pkl')
with open('features.json') as f:
    FEATURES = json.load(f)['features']

app = FastAPI(title="Credit Scoring Service", version="1.0")

class ScoringRequest(BaseModel):
    loan_id: str
    features: dict  # 原始特征字典

class ScoringResponse(BaseModel):
    loan_id: str
    score: float         # 违约概率
    risk_level: str      # HIGH / MEDIUM / LOW
    model_version: str
    reject_reasons: list  # 拒绝理由码
    latency_ms: float

@app.post("/score", response_model=ScoringResponse)
async def score(request: ScoringRequest):
    start_time = time.time()

    try:
        # 特征提取与校验
        features = request.features
        missing_features = [f for f in FEATURES if f not in features]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # 缺失特征使用默认值，不直接报错（容错处理）
            for f in missing_features:
                features[f] = None

        # 构建特征向量
        import pandas as pd
        X = pd.DataFrame([features])[FEATURES]

        # 预处理（与训练时相同的pipeline）
        X_processed = PIPELINE.transform(X)

        # 预测
        prob = float(MODEL.predict_proba(X_processed)[0][1])
        score = prob

        # 分档
        if prob > 0.15:
            risk_level = "HIGH"
        elif prob > 0.05:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # 生成拒绝理由（使用SHAP，可按需开启）
        reject_reasons = []

    except Exception as e:
        logger.error(f"Scoring failed for loan_id={request.loan_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency = (time.time() - start_time) * 1000
    return ScoringResponse(
        loan_id=request.loan_id,
        score=round(score, 6),
        risk_level=risk_level,
        model_version="consumer_apply_lgbm_v2.1",
        reject_reasons=reject_reasons,
        latency_ms=round(latency, 2),
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": MODEL is not None}
```

### 8.3.2 批量评分（离线场景）

```python
def batch_scoring(loan_ids, feature_df, model, pipeline, batch_size=10000):
    """
    大规模批量评分（行为评分、存量客户定期跑分）
    """
    results = []
    n_batches = (len(feature_df) + batch_size - 1) // batch_size

    for i in range(n_batches):
        batch = feature_df.iloc[i*batch_size:(i+1)*batch_size]
        X_processed = pipeline.transform(batch)
        probs = model.predict_proba(X_processed)[:, 1]

        batch_results = pd.DataFrame({
            'loan_id': loan_ids[i*batch_size:(i+1)*batch_size],
            'risk_score': probs,
            'score_date': pd.Timestamp.today().date(),
            'model_version': 'behavior_lgbm_v1.2',
        })
        results.append(batch_results)

        if (i + 1) % 10 == 0:
            print(f"Processed {(i+1)*batch_size:,} / {len(feature_df):,}")

    return pd.concat(results, ignore_index=True)
```

---

## 8.4 灰度上线与A/B测试

### 8.4.1 灰度策略

**新模型不应直接全量上线，需要逐步验证：**

```
灰度流程：
1. 影子模式（Shadow Mode）：新模型并行运行，不影响决策，仅记录分数
   └── 目的：验证线上特征取数逻辑、服务稳定性
   └── 时间：1~2周

2. 小比例切量：5%流量使用新模型决策
   └── 目的：验证业务指标（通过率、坏率方向是否符合预期）
   └── 时间：2~4周（等待足够的表现数据）

3. 扩大切量：20%→50%→100%
   └── 每次扩量前验证前一阶段数据无异常
```

### 8.4.2 A/B测试设计

```python
class ABTestMonitor:
    """A/B测试监控"""

    def __init__(self, control_model='v3.1', treatment_model='v3.2'):
        self.control = control_model
        self.treatment = treatment_model

    def assign_group(self, user_id, treatment_pct=0.1):
        """基于用户ID哈希稳定分组"""
        import hashlib
        hash_val = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
        return 'treatment' if (hash_val % 100) < (treatment_pct * 100) else 'control'

    def compare_metrics(self, df, groupby='ab_group'):
        """比较两组关键指标"""
        metrics = df.groupby(groupby).agg(
            n_applications=('loan_id', 'count'),
            pass_rate=('is_approved', 'mean'),
            avg_score=('risk_score', 'mean'),
            fpd_rate=('is_fpd', 'mean'),  # 等足够账龄后才能统计
        )

        # 统计显著性检验（通过率差异）
        from scipy import stats
        control = df[df[groupby] == 'control']['is_approved']
        treatment = df[df[groupby] == 'treatment']['is_approved']
        t_stat, p_value = stats.ttest_ind(control, treatment)

        print(f"通过率差异 p-value: {p_value:.4f}")
        print("显著性：", "显著(p<0.05)" if p_value < 0.05 else "不显著")

        return metrics
```

---

## 8.5 特征一致性保障

Training-Serving Skew 是工程化中最常见的隐患：

```python
# 特征一致性检验工具
def check_feature_consistency(offline_features, online_features, sample_ids):
    """
    比较离线特征计算与线上特征计算的一致性
    通过抽样比对发现skew
    """
    results = []
    for feature in offline_features.columns:
        offline_vals = offline_features.loc[sample_ids, feature]
        online_vals = online_features.loc[sample_ids, feature]

        # 数值型：计算绝对误差
        if offline_vals.dtype in ['float64', 'int64']:
            abs_error = (offline_vals - online_vals).abs()
            results.append({
                'feature': feature,
                'max_abs_error': abs_error.max(),
                'mean_abs_error': abs_error.mean(),
                'mismatch_rate': (abs_error > 1e-6).mean(),
                'status': 'OK' if abs_error.max() < 1e-6 else 'MISMATCH'
            })
        else:
            mismatch = (offline_vals != online_vals).mean()
            results.append({
                'feature': feature,
                'mismatch_rate': mismatch,
                'status': 'OK' if mismatch == 0 else 'MISMATCH'
            })

    report = pd.DataFrame(results)
    mismatches = report[report['status'] == 'MISMATCH']
    if len(mismatches) > 0:
        print(f"⚠️  发现 {len(mismatches)} 个特征存在不一致：")
        print(mismatches[['feature', 'mismatch_rate']].to_string())
    else:
        print("✅ 所有特征一致性检验通过")

    return report
```

---

## 8.6 模型降级与容灾

线上服务必须考虑模型不可用时的降级方案：

```python
class RobustScoringService:
    """带降级机制的评分服务"""

    def __init__(self, primary_model, fallback_rules):
        self.primary_model = primary_model
        self.fallback_rules = fallback_rules  # 简单规则兜底
        self.timeout_ms = 200  # 模型超时阈值

    def score(self, features):
        try:
            # 主模型：超时则抛异常
            import signal
            # ... 实现超时逻辑 ...
            return self._score_with_model(features)

        except TimeoutError:
            logger.warning("主模型超时，切换到规则降级")
            return self._score_with_rules(features)

        except Exception as e:
            logger.error(f"主模型异常: {e}，切换到规则降级")
            return self._score_with_rules(features)

    def _score_with_rules(self, features):
        """规则降级：基于简单特征的保守策略"""
        # 降级时通常采用保守策略（宁可多拒少批）
        if features.get('credit_query_cnt_m3', 0) > 10:
            return {'score': 0.8, 'source': 'fallback_rules', 'risk': 'HIGH'}
        if features.get('max_overdue_months', 0) >= 3:
            return {'score': 0.7, 'source': 'fallback_rules', 'risk': 'HIGH'}
        # 其他情况给中等风险
        return {'score': 0.1, 'source': 'fallback_rules', 'risk': 'MEDIUM'}
```

---

> **本章小结**：模型工程化的核心是**可靠性**和**可追踪性**。版本管理、特征一致性检验、灰度上线、降级方案，每一个环节的缺失都可能导致生产事故。算法工程师需要在模型质量和工程质量上同等投入。
