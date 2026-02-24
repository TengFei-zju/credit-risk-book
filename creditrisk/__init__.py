"""
creditrisk: 信贷风控建模工具包
配套《信贷风控建模：打工人手册》

模块结构：
  creditrisk.data        数据加载与预处理
  creditrisk.features    特征工程（WOE、聚合、编码）
  creditrisk.models      模型（评分卡、LightGBM、集成）
  creditrisk.evaluation  模型评估（KS、AUC、PSI、Lift）
  creditrisk.ensemble    模型融合（Stacking、Blending）
  creditrisk.selection   特征选择（Null Importance、对抗验证）
  creditrisk.utils       通用工具
"""

__version__ = "0.1.0"
__author__  = "信贷风控建模：打工人手册"

from creditrisk import data, features, models, evaluation, ensemble, selection, utils
