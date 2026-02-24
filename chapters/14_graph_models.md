# 第十四章 图模型在风控中的应用

## 14.1 图模型在风控中的价值

### 14.1.1 为什么需要图模型

传统机器学习模型假设样本独立同分布（i.i.d.），但信贷风控中存在大量**关联关系**：

```
传统模型局限：
- 只利用用户自身特征（年龄、收入、征信...）
- 忽略用户之间的关系（担保人、共同借款、设备共享...）
- 无法识别团伙欺诈

图模型优势：
- 显式建模用户间关系
- 捕捉风险传导路径
- 识别异常子图（欺诈团伙）
```

### 14.1.2 风控场景中的图结构

```
节点（Nodes）：
├── 借款人（核心节点）
├── 担保人/共同借款人
├── 设备（手机、IP 地址）
├── 联系方式（手机号、邮箱）
└── 地址（家庭地址、公司地址）

边（Edges）：
├── 担保关系（强连接）
├── 共同借款（强连接）
├── 设备共享（中强连接）
├── 联系方式共享（中强连接）
└── 地址共享（弱连接）
```

---

### 14.1.3 图结构可视化

#### 客户关系图谱

![客户关系图谱](diagrams/ch14_customer_relationship_graph.drawio)

**图例说明**：
- 🔴 红色节点：违约客户
- 🟢 绿色节点：正常客户
- 🔵 蓝色节点：设备节点
- 🟡 黄色节点：地址节点
- 实线边：担保关系（强连接）
- 虚线边：共享关系（中弱连接）

**风险洞察**：
- 用户 A、C 均违约，且共用设备 D1 → 可能存在团伙欺诈
- 用户 B 与违约用户 A 有担保关系，且共用同一设备 → 风险传导信号
- 用户 D、E 共用地址但无其他风险信号 → 需结合其他特征判断

---

### 14.1.4 GCN 消息传递机制

![GCN 消息传递示意图](diagrams/ch14_gcn_message_passing.drawio)

上图展示了图卷积网络（GCN）的核心操作：
1. 中心节点 A 聚合邻居 B、C、D 的特征
2. 通过聚合函数（⊕）生成新的节点嵌入
3. 输出层将嵌入映射为违约概率

---

## 14.2 图特征工程

### 14.2.1 基础图特征

```python
import networkx as nx
import pandas as pd

def build_customer_graph(loans_df, relations_df):
    """
    构建客户关系图
    loans_df: 借款记录表（customer_id, loan_id, ...）
    relations_df: 关系表（customer_id_1, customer_id_2, relation_type）
    """
    G = nx.Graph()

    # 添加节点（客户）
    customers = loans_df['customer_id'].unique()
    for c in customers:
        G.add_node(c, node_type='customer')

    # 添加边（关系）
    for _, row in relations_df.iterrows():
        G.add_edge(
            row['customer_id_1'],
            row['customer_id_2'],
            relation_type=row['relation_type'],
            weight={'guarantor': 1.0, 'co_borrower': 0.8, 'device': 0.5}.get(
                row['relation_type'], 0.3
            )
        )

    return G


def extract_graph_features(G, customer_id):
    """
    提取客户的图特征
    """
    if customer_id not in G:
        return {}

    # 1. 一度邻居特征
    neighbors = list(G.neighbors(customer_id))
    n_neighbors = len(neighbors)

    # 2. 二度邻居特征（朋友的朋友）
    two_hop_neighbors = set()
    for n in neighbors:
        two_hop_neighbors.update(G.neighbors(n))
    two_hop_neighbors.discard(customer_id)
    n_two_hop = len(two_hop_neighbors)

    # 3. 节点中心性
    degree centrality = nx.degree_centrality(G).get(customer_id, 0)
    betweenness = nx.betweenness_centrality(G).get(customer_id, 0)

    # 4. 所在连通分量
    component_id = -1
    component_size = 0
    for i, component in enumerate(nx.connected_components(G)):
        if customer_id in component:
            component_id = i
            component_size = len(component)
            break

    # 5. 聚类系数（衡量邻居间的连接紧密程度）
    clustering_coef = nx.clustering(G).get(customer_id, 0)

    return {
        'n_neighbors': n_neighbors,
        'n_two_hop_neighbors': n_two_hop,
        'degree_centrality': degree_centrality,
        'betweenness_centrality': betweenness,
        'component_size': component_size,
        'clustering_coefficient': clustering_coef,
    }
```

### 14.2.2 风险传导特征

```python
def risk_propagation_features(G, loans_df, target_customer):
    """
    计算风险传导特征
    基于邻居的违约情况
    """
    # 构建客户违约映射
    default_map = loans_df.set_index('customer_id')['isDefault'].to_dict()

    # 一度邻居违约统计
    neighbors = list(G.neighbors(target_customer))
    if not neighbors:
        return {'neighbor_default_rate': 0, 'neighbor_default_count': 0}

    neighbor_defaults = sum(default_map.get(n, 0) for n in neighbors)
    neighbor_default_rate = neighbor_defaults / len(neighbors)

    # 加权违约率（考虑关系强度）
    weighted_defaults = 0
    total_weight = 0
    for n in neighbors:
        weight = G[target_customer][n].get('weight', 0.5)
        weighted_defaults += default_map.get(n, 0) * weight
        total_weight += weight

    weighted_default_rate = weighted_defaults / (total_weight + 1e-6)

    return {
        'neighbor_default_rate': neighbor_default_rate,
        'neighbor_default_count': neighbor_defaults,
        'weighted_neighbor_default_rate': weighted_default_rate,
        'n_good_neighbors': len(neighbors) - neighbor_defaults,
    }
```

---

## 14.3 图神经网络（GNN）基础

### 14.3.1 图卷积网络（GCN）原理

GCN 的核心思想：**节点的特征通过邻居聚合进行更新**。

```
数学形式（简化版）:
h_v^(l+1) = σ(Σ_{u∈N(v)} W^(l) · h_u^(l) / |N(v)|)

其中：
- h_v^(l): 节点 v 在第 l 层的特征
- N(v): v 的邻居节点集合
- W^(l): 可学习权重矩阵
- σ: 激活函数（如 ReLU）
```

**消息传递流程**：
1. **输入**：每个节点有初始特征向量（如用户属性）
2. **聚合**：中心节点 A 聚合邻居 B、C、D 的特征
3. **更新**：通过可学习权重 W 和激活函数σ，生成新的节点嵌入
4. **输出**：最终的节点嵌入用于预测（如违约概率）

**多层 GCN 的表达能力**：
- 1 层 GCN：聚合 1 阶邻居信息
- 2 层 GCN：聚合 2 阶邻居信息（朋友的朋友）
- 3 层 GCN：聚合 3 阶邻居信息（通常 2-3 层已足够）

### 14.3.2 使用 PyTorch Geometric 实现 GCN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GCNForRisk(nn.Module):
    """
    用于风控的图卷积网络
    """
    def __init__(self, num_node_features, hidden_dim=64, num_layers=2):
        super().__init__()

        layers = []
        # 输入层
        layers.append(GCNConv(num_node_features, hidden_dim))
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)

        # 隐藏层
        for i in range(num_layers - 1):
            layers.append(GCNConv(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))

        self.gcn_layers = nn.ModuleList(layers)

        # 输出层
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, edge_weight=None):
        """
        x: 节点特征矩阵 (num_nodes, num_features)
        edge_index: 边索引 (2, num_edges)
        edge_weight: 边权重 (num_edges,)
        """
        h = x
        for i, layer in enumerate(self.gcn_layers):
            if isinstance(layer, GCNConv):
                h = layer(h, edge_index, edge_weight=edge_weight)
                if i < len(self.gcn_layers) - 1:  # 最后一层不加激活
                    h = self.batch_norm1(h)
                    h = F.relu(h)
                    h = self.dropout(h)
            else:
                h = layer(h)
                h = F.relu(h)
                h = self.dropout(h)

        # 输出违约概率
        out = torch.sigmoid(self.classifier(h))
        return out.squeeze()


# 数据准备示例
def prepare_graph_data(loans_df, relations_df, node_features_df):
    """
    准备图神经网络输入数据
    """
    # 创建 customer_id 到索引的映射
    customers = loans_df['customer_id'].unique()
    id_to_idx = {c: i for i, c in enumerate(customers)}

    # 节点特征
    node_features = torch.FloatTensor(
        node_features_df.loc[customers].fillna(0).values
    )

    # 边索引
    edge_list = []
    edge_weights = []
    for _, row in relations_df.iterrows():
        if row['customer_id_1'] in id_to_idx and row['customer_id_2'] in id_to_idx:
            edge_list.append([
                id_to_idx[row['customer_id_1']],
                id_to_idx[row['customer_id_2']]
            ])
            edge_weights.append(0.8)  # 无向图，双向添加
            edge_list.append([
                id_to_idx[row['customer_id_2']],
                id_to_idx[row['customer_id_1']]
            ])
            edge_weights.append(0.8)

    edge_index = torch.LongTensor(edge_list).t()  # (2, num_edges)
    edge_weight = torch.FloatTensor(edge_weights)

    # 标签
    y = torch.FloatTensor(loans_df.set_index('customer_id').loc[customers]['isDefault'].values)

    # 构建 PyG Data
    data = Data(x=node_features, edge_index=edge_index,
                edge_weight=edge_weight, y=y)

    return data, customers, id_to_idx
```

### 14.3.3 GraphSAGE：适用于大图的采样方法

```python
from torch_geometric.nn import SAGEConv

class GraphSAGEForRisk(nn.Module):
    """
    GraphSAGE：通过邻居采样进行归纳式学习
    适合大规模图（无法一次性加载到内存）
    """
    def __init__(self, num_node_features, hidden_dim=128, num_layers=3):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(num_node_features, hidden_dim))

        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)

        out = torch.sigmoid(self.classifier(x))
        return out.squeeze()


# 使用邻居采样进行小批量训练
from torch_geometric.loader import NeighborLoader

def create_neighbor_loader(data, batch_size=256, num_neighbors=[10, 5]):
    """
    邻居采样 DataLoader
    num_neighbors: 每层采样的邻居数
    """
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,  # 一阶采样 10 个，二阶采样 5 个
        batch_size=batch_size,
        shuffle=True,
        input_nodes=torch.arange(data.num_nodes),
    )
    return loader
```

---

## 14.4 图模型在反欺诈中的应用

### 14.4.1 欺诈团伙检测

```python
import networkx as nx
from community import community_louvain

def detect_fraud_rings(G, resolution=1.0):
    """
    使用 Louvain 算法检测欺诈团伙
    resolution: 分辨率参数，越大社区越小
    """
    # Louvain 社区检测
    partition = community_louvain.best_partition(G, resolution=resolution)

    # 分析每个社区的风险
    community_risk = {}
    for node, comm_id in partition.items():
        if comm_id not in community_risk:
            community_risk[comm_id] = {'nodes': [], 'default_count': 0}
        community_risk[comm_id]['nodes'].append(node)

    # 计算每个社区的违约率
    for comm_id, info in community_risk.items():
        # 这里需要实际的违约标签
        # default_count = sum(1 for n in info['nodes'] if has_default_label(n))
        # info['default_rate'] = default_count / len(info['nodes'])
        pass

    # 识别高风险社区（违约率高且节点数适中）
    high_risk_communities = [
        comm_id for comm_id, info in community_risk.items()
        if len(info['nodes']) >= 3  # 至少 3 个节点
        # and info['default_rate'] > 0.3  # 违约率超过 30%
    ]

    return partition, high_risk_communities
```

### 14.4.2 异常子图检测

```python
def detect_anomalous_subgraphs(G, min_size=3, max_diameter=3):
    """
    检测异常子图（可能是欺诈团伙）
    特征：
    - 完全子图（clique）：所有节点两两相连
    - 高密度子图：边数接近节点数的完全图
    """
    anomalous_subgraphs = []

    # 查找所有团（clique）
    cliques = list(nx.find_cliques(G))

    for clique in cliques:
        if len(clique) >= min_size:
            anomalous_subgraphs.append({
                'nodes': clique,
                'type': 'clique',
                'size': len(clique),
                'density': 1.0,  # 完全图密度为 1
            })

    # 查找高密度子图
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        if len(component) >= min_size:
            density = nx.density(subgraph)
            diameter = nx.diameter(subgraph) if nx.is_connected(subgraph) else -1

            if density > 0.5 and diameter <= max_diameter:
                anomalous_subgraphs.append({
                    'nodes': list(component),
                    'type': 'dense_subgraph',
                    'size': len(component),
                    'density': density,
                    'diameter': diameter,
                })

    return anomalous_subgraphs
```

---

## 14.5 实战：设备关联图谱

### 14.5.1 构建设备 - 用户二分图

```python
import networkx as nx

def build_device_user_graph(loans_df, device_log_df):
    """
    构建设备 - 用户二分图
    loans_df: 借款记录（user_id, loan_id）
    device_log_df: 设备日志（user_id, device_id, device_type）
    """
    G = nx.Graph()

    # 添加用户节点
    users = loans_df['user_id'].unique()
    for u in users:
        G.add_node(f'U_{u}', node_type='user', is_borrower=True)

    # 添加设备节点
    devices = device_log_df['device_id'].unique()
    for d in devices:
        device_info = device_log_df[device_log_df['device_id'] == d].iloc[0]
        G.add_node(
            f'D_{d}',
            node_type='device',
            device_type=device_info['device_type']
        )

    # 添加用户 - 设备边
    for _, row in device_log_df.iterrows():
        G.add_edge(
            f'U_{row["user_id"]}',
            f'D_{row["device_id"]}',
            weight=0.8
        )

    return G


def analyze_device_sharing(G):
    """
    分析设备共享情况
    """
    device_sharing_stats = {}

    for node, data in G.nodes(data=True):
        if data['node_type'] == 'user':
            neighbors = list(G.neighbors(node))
            device_neighbors = [
                n for n in neighbors
                if G.nodes[n]['node_type'] == 'device'
            ]

            # 共享设备数（被多个用户使用的设备）
            shared_devices = 0
            for d in device_neighbors:
                device_users = [
                    n for n in G.neighbors(d)
                    if G.nodes[n]['node_type'] == 'user'
                ]
                if len(device_users) > 1:
                    shared_devices += 1

            device_sharing_stats[node] = {
                'n_devices': len(device_neighbors),
                'n_shared_devices': shared_devices,
                'shared_device_ratio': shared_devices / (len(device_neighbors) + 1e-6),
            }

    return device_sharing_stats
```

### 14.5.2 图特征加入机器学习模型

```python
def integrate_graph_features_into_ml(X_train, G, customer_ids):
    """
    将图特征整合到传统机器学习流程
    """
    graph_features = []

    for customer_id in customer_ids:
        # 提取基础图特征
        feat = extract_graph_features(G, customer_id)

        # 提取风险传导特征
        risk_feat = risk_propagation_features(G, X_train, customer_id)

        # 合并
        feat.update(risk_feat)
        graph_features.append(feat)

    # 转换为 DataFrame
    graph_df = pd.DataFrame(graph_features, index=customer_ids)

    # 与原始特征合并
    X_enhanced = pd.concat([X_train, graph_df], axis=1)

    return X_enhanced
```

---

## 14.6 图模型的挑战与注意事项

### 14.6.1 数据隐私与合规

```markdown
⚠️ 图模型使用的合规注意事项：

1. 数据来源合法性
   - 用户关系数据需经用户授权
   - 不得非法获取通讯录等隐私数据

2. 关联关系的使用边界
   - 不得因"关联人违约"直接拒绝客户
   - 图特征只能作为风险参考信号

3. 监管要求
   - 需向监管说明图模型的使用逻辑
   - 保留人工复核通道
```

### 14.6.2 图数据的时间一致性

```python
def build_temporal_graph(loans_df, relations_df, observation_date):
    """
    构建时序一致的图
    只能使用观察日之前的关系数据
    """
    # 过滤：关系必须在观察日之前存在
    relations_before = relations_df[
        relations_df['relation_start_date'] <= observation_date
    ].copy()

    # 过滤：只考虑在观察日之前的借款
    loans_before = loans_df[
        loans_df['loan_date'] <= observation_date
    ].copy()

    G = build_customer_graph(loans_before, relations_before)
    return G
```

---

> **本章小结**：图模型为风控提供了关系视角的风险识别能力，特别适合反欺诈场景。GCN 和 GraphSAGE 是主流的图神经网络方法，能够端到端学习节点表示。使用时需注意数据隐私合规和时序一致性问题。
