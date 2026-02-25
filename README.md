# 信贷风控建模：打工人手册

> 信贷从业者的实战建模指南
> 附完整可运行代码库，基于阿里云天池「贷款违约预测」数据集

作者：汪叽意且
版本：v0.2
最后更新：2026-02

---

## 项目结构

```
credit_pd_book/
├── creditrisk/               ← Python 工具包（核心代码库）
│   ├── data/                 数据加载、天池数据清洗
│   ├── features/             WOE编码、Target Encoding、特征构造
│   ├── models/               评分卡、LightGBM+OOF
│   ├── evaluation/           KS、AUC、PSI、Lift
│   ├── ensemble/             Stacking、Rank Blend、Pseudo-label
│   ├── selection/            Null Importance、对抗验证
│   └── utils/                可视化、计时器
│
├── notebooks/                ← 按章节组织的 Jupyter Notebooks（可逐步运行）
│   ├── ch00_data_setup.ipynb
│   ├── ch04_feature_engineering.ipynb
│   ├── ch05_scorecard.ipynb
│   └── ch06_machine_learning.ipynb
│
├── chapters/                 ← Markdown 书稿
├── data/                     ← 数据目录（见 data/README.md）
├── run_tests.py              ← 快速验证（无需真实数据）
└── pyproject.toml
```

## 快速开始

```bash
# 1. 安装依赖
pip install -e .

# 2. 验证代码库（无需数据）
python run_tests.py

# 3. 下载天池数据放入 data/ 目录，然后顺序运行 notebooks：
#    ch00 → ch04 → ch05 → ch06
```

## Kaggle 核心技巧索引

| 技巧 | 所在模块 | Notebook |
|------|---------|---------|
| Null Importance 特征选择 | `selection.NullImportanceSelector` | ch04 |
| 对抗验证 Adversarial Validation | `selection.AdversarialValidator` | ch04 |
| Target Encoding（CV防穿越） | `features.TargetEncoder` | ch04 |
| OOF 交叉验证预测 | `models.LGBMWithOOF` | ch06 |
| Optuna 超参数搜索 | ch06 notebook | ch06 |
| Rank-based Blending | `ensemble.rank_blend` | ch06 |
| OOF Stacking | `ensemble.StackingEnsemble` | ch06 |
| Pseudo-labeling（拒绝推断） | `ensemble.pseudo_label` | ch06 |
| SHAP 理由码生成 | ch06 notebook | ch06 |
| 匿名特征统计聚合 | `features.anonymized_stats` | ch04 |

---

## 章节目录

| 章节 | 标题 | Notebook |
|------|------|---------|
| 第一章 | [信贷风控行业全景](chapters/01_industry_overview.md) | — |
| 第二章 | [业务理解与需求转化](chapters/02_business_understanding.md) | — |
| 第三章 | [数据体系与数据治理](chapters/03_data_system.md) | ch00 |
| 第四章 | [特征工程](chapters/04_feature_engineering.md) | ch04 |
| 第五章 | [评分卡建模](chapters/05_scorecard.md) | ch05 |
| 第六章 | [机器学习建模](chapters/06_machine_learning.md) | ch06 |
| 第七章 | [模型评估与验证](chapters/07_model_evaluation.md) | ch05/06 |
| 第八章 | [模型上线与工程化](chapters/08_model_deployment.md) | — |
| 第九章 | [模型监控与迭代](chapters/09_model_monitoring.md) | — |
| 第十章 | [策略设计与决策引擎](chapters/10_strategy_decision.md) | — |
| 第十一章 | [反欺诈建模](chapters/11_anti_fraud.md) | — |
| 第十二章 | [催收评分与贷后管理](chapters/12_collection.md) | — |
| 第十三章 | [图模型在风控中的应用](chapters/13_graph_models.md) | — |
| 第十四章 | [序列模型在风控中的应用](chapters/14_sequence_models.md) | — |
| 第十五章 | [风控中的大模型应用](chapters/15_llm_in_risk.md) | — |
| 附录 | [常用工具与资源](chapters/appendix.md) | — |
