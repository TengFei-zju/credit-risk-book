# 第十一章 反欺诈建模

## 11.1 欺诈与信用风险的本质区别

| 维度 | 信用风险 | 欺诈风险 |
|------|---------|---------|
| 本质 | 客户**不能**还款 | 客户**不想**还款（主观恶意） |
| 时间特征 | 随时间逐渐暴露 | 早期爆发（FPD高） |
| 群体特征 | 通常个体行为 | 常有团伙组织性 |
| 数据信号 | 财务行为、征信 | 身份、设备、行为序列 |
| 模型目标 | 准确排序 | 精准捕获（高Recall） |
| 时效性 | 相对稳定 | 快速对抗，模型衰退快 |

---

## 11.2 欺诈类型图谱

```
欺诈类型
├── 身份欺诈
│   ├── 伪冒身份（使用他人身份证）
│   ├── 合成身份（拼凑虚假身份）
│   └── 账户盗用（合法用户账号被盗）
├── 申请欺诈
│   ├── 收入/工作造假
│   ├── 资产造假
│   └── 联系方式造假（虚假紧急联系人）
├── 行为欺诈
│   ├── 薅羊毛（套取优惠）
│   └── 骗贷（一旦放款立即失联）
└── 团伙欺诈（最复杂）
    ├── 中介代办（批量申请）
    ├── 欺诈工厂（有组织批量骗贷）
    └── 设备农场（模拟器批量注册）
```

---

## 11.3 反欺诈特征体系

### 11.3.1 身份核验特征

```python
# 身份一致性特征
identity_features = {
    # 手机号与身份证的关联强度
    'phone_id_match_score': ...,         # 三方实名认证分

    # 身份证信息的内部一致性
    'id_age_match': ...,                  # 身份证年龄 vs 填写年龄是否一致
    'id_hometown_match': ...,             # 身份证归属地 vs 填写住址是否一致

    # 手机号特征
    'phone_tenure_months': ...,           # 手机号使用年限（<6月为风险信号）
    'phone_carrier': ...,                 # 运营商
    'phone_is_virtual': ...,             # 是否虚拟号段

    # 银行卡特征
    'bank_card_verified': ...,           # 银行卡四要素验证是否通过
    'bank_card_multi_user': ...,         # 该银行卡是否被多人使用（共债/团伙信号）
}
```

### 11.3.2 设备与网络特征

```python
device_features = {
    # 设备指纹
    'device_id': ...,                    # 设备唯一标识
    'device_type': ...,                  # 手机型号
    'is_emulator': ...,                  # 是否模拟器（强欺诈信号）
    'is_rooted': ...,                    # 是否root/越狱
    'is_vpn': ...,                       # 是否使用VPN

    # 设备-账号关联
    'device_multi_account': ...,         # 同设备注册多账号数量
    'device_risk_history': ...,          # 该设备历史欺诈记录

    # 网络特征
    'ip_risk_score': ...,               # IP风险评分（黑IP库）
    'ip_location_mismatch': ...,         # IP归属地 vs 申请地是否一致
    'wifi_ssid_match': ...,             # WiFi SSID是否异常
}
```

### 11.3.3 申请行为特征（埋点数据）

```python
def extract_application_behavior(behavior_log):
    """
    从申请表填写行为中提取欺诈信号
    正常用户的填写行为有固定的时间、顺序模式
    """
    features = {}

    # 时间特征
    features['fill_total_seconds'] = (
        behavior_log['submit_time'] - behavior_log['start_time']
    ).total_seconds()

    # 异常快速填写（可能使用填单工具）
    features['is_suspiciously_fast'] = features['fill_total_seconds'] < 30

    # 粘贴行为
    features['paste_phone_count'] = behavior_log['phone_paste_events']
    features['paste_id_count'] = behavior_log['id_paste_events']
    features['paste_bank_count'] = behavior_log['bank_paste_events']
    features['total_paste_count'] = sum([
        features['paste_phone_count'],
        features['paste_id_count'],
        features['paste_bank_count'],
    ])

    # 修改行为（修改次数多可能是在尝试不同身份）
    features['phone_modify_count'] = behavior_log['phone_changes']
    features['id_modify_count'] = behavior_log['id_changes']

    # 页面浏览时间（欺诈者往往直接跳过隐私协议）
    features['privacy_page_stay_seconds'] = behavior_log.get('privacy_stay_time', 0)
    features['skim_privacy_policy'] = features['privacy_page_stay_seconds'] < 2

    return features
```

---

## 11.4 关系图谱与团伙识别

团伙欺诈识别是反欺诈的核心难点，图神经网络（GNN）是当前主流方向。

### 11.4.1 图构建

```python
import networkx as nx

def build_fraud_graph(applications_df):
    """
    构建欺诈关系图
    节点：申请人
    边：共享同一属性（手机号、设备、IP、紧急联系人等）
    """
    G = nx.Graph()

    # 添加节点
    for _, row in applications_df.iterrows():
        G.add_node(row['loan_id'], **{
            'is_fraud': row.get('is_fraud', None),
            'apply_date': row['apply_date'],
        })

    # 基于共享属性添加边
    shared_attrs = ['phone', 'device_id', 'ip_address', 'emergency_phone',
                    'bank_card', 'company_name', 'home_address']

    for attr in shared_attrs:
        groups = applications_df.groupby(attr)['loan_id'].apply(list)
        for loan_ids in groups:
            if len(loan_ids) > 1:
                for i in range(len(loan_ids)):
                    for j in range(i + 1, len(loan_ids)):
                        G.add_edge(loan_ids[i], loan_ids[j],
                                   relation=attr,
                                   weight=G[loan_ids[i]][loan_ids[j]].get('weight', 0) + 1
                                   if G.has_edge(loan_ids[i], loan_ids[j])
                                   else 1)
    return G

def extract_graph_features(G, node_id):
    """从图中提取节点特征"""
    if node_id not in G:
        return {}

    neighbors = list(G.neighbors(node_id))
    fraud_neighbors = [n for n in neighbors if G.nodes[n].get('is_fraud') == 1]

    return {
        'degree': G.degree(node_id),                    # 关联人数
        'fraud_neighbor_count': len(fraud_neighbors),    # 欺诈邻居数量
        'fraud_neighbor_rate': len(fraud_neighbors) / max(len(neighbors), 1),
        'max_edge_weight': max([G[node_id][n].get('weight', 1) for n in neighbors], default=0),
        'clustering_coef': nx.clustering(G, node_id),   # 聚类系数
    }
```

### 11.4.2 图神经网络（GNN）简介

```python
# 使用 PyTorch Geometric 构建简单的 GraphSAGE 模型
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class FraudDetectionGNN(torch.nn.Module):
    """
    GraphSAGE 用于欺诈检测
    通过聚合邻居特征来识别团伙
    """
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels // 2)
        self.classifier = torch.nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index):
        # 第一层：聚合1跳邻居
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # 第二层：聚合2跳邻居
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # 分类输出
        x = self.classifier(x)
        return torch.sigmoid(x)
```

---

## 11.5 实时反欺诈架构

反欺诈对时效性要求极高，需要毫秒级响应：

```
实时反欺诈架构：

申请请求
    ↓
API网关
    ↓
规则引擎（<10ms）          ← 黑名单、设备指纹、IP黑名单
    ↓
特征实时计算（<50ms）      ← 行为序列、关系图特征
    ↓
模型推理（<20ms）          ← 欺诈评分
    ↓
决策（<5ms）               ← 通过/审核/拒绝
    ↓
异步处理（>5s）             ← 图谱更新、团伙标记、人工审核队列
```

```python
# 实时特征计算：Redis缓存热数据
import redis
import json

class RealtimeFeatureStore:
    """
    基于Redis的实时特征存储
    用于反欺诈的实时特征查询
    """
    def __init__(self, redis_client):
        self.r = redis_client
        self.expire_seconds = 3600  # 1小时过期

    def get_device_features(self, device_id):
        """获取设备风险特征"""
        key = f"device:{device_id}"
        cached = self.r.get(key)
        if cached:
            return json.loads(cached)

        # 缓存未命中，从数据库查询
        features = self._query_device_history(device_id)
        self.r.setex(key, self.expire_seconds, json.dumps(features))
        return features

    def increment_apply_count(self, entity_key, window_seconds=3600):
        """
        统计滑动窗口内的申请次数
        用于检测：同一设备/IP在1小时内的申请频次
        """
        current_time = int(pd.Timestamp.now().timestamp())
        window_start = current_time - window_seconds

        pipe = self.r.pipeline()
        pipe.zadd(entity_key, {str(current_time): current_time})
        pipe.zremrangebyscore(entity_key, '-inf', window_start)
        pipe.zcard(entity_key)
        pipe.expire(entity_key, window_seconds)
        results = pipe.execute()

        return results[2]  # 返回窗口内的申请次数
```

---

## 11.6 反欺诈模型的特殊评估

欺诈检测中，**Recall比Precision更重要**（漏掉一个欺诈的代价 >> 多拒一个好客户）：

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

def fraud_model_evaluation(y_true, y_score):
    """反欺诈模型评估（以Recall为核心）"""

    # Precision-Recall曲线
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    # 在固定Recall下的Precision（业务常见需求）
    target_recalls = [0.7, 0.8, 0.9]
    for target_recall in target_recalls:
        idx = np.argmin(np.abs(recall - target_recall))
        print(f"Recall={target_recall:.0%} 时：Precision={precision[idx]:.2%}, "
              f"阈值={thresholds[min(idx, len(thresholds)-1)]:.4f}")

    print(f"\nAverage Precision (AP): {ap:.4f}")

    # 欺诈金额捕获率（比笔数更有业务价值）
    df = pd.DataFrame({'score': y_score, 'label': y_true,
                        'amount': np.random.lognormal(8, 1, len(y_true))})  # 示意
    df_sorted = df.sort_values('score', ascending=False)

    for reject_pct in [0.05, 0.10, 0.20]:
        n_reject = int(len(df) * reject_pct)
        rejected = df_sorted.head(n_reject)
        fraud_capture_rate = rejected['label'].sum() / df['label'].sum()
        fraud_amount_rate = rejected[rejected['label']==1]['amount'].sum() / df[df['label']==1]['amount'].sum()
        print(f"拒绝前{reject_pct:.0%}：欺诈笔数捕获率={fraud_capture_rate:.2%}, "
              f"欺诈金额捕获率={fraud_amount_rate:.2%}")
```

---

## 11.7 欺诈对抗

欺诈是动态对抗，模型会被欺诈团伙逐步破解：

```
对抗演进：
欺诈团伙攻击 → 模型识别 → 团伙学习规避 → 模型迭代
      ↓                              ↓
  新型欺诈手法                   规则更新/模型重建
```

**对抗策略**：
- **特征多样化**：使用欺诈团伙难以批量伪造的特征（生物特征、行为序列）
- **模型集成**：多个模型联合决策，单一规则规避无效
- **在线学习**：快速响应新型欺诈模式
- **蜜罐（Honeypot）**：故意留出可利用的漏洞，引诱欺诈团伙暴露更多信息

---

> **本章小结**：反欺诈建模的核心挑战是对抗性和时效性。图神经网络使团伙欺诈识别成为可能，实时特征存储（Redis）支撑毫秒级响应，而正确的评估指标（以Recall为核心）确保业务决策不偏离目标。欺诈对抗是一场永无止境的军备竞赛，持续迭代是核心能力。
