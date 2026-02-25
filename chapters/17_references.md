# 第十七章 参考文献与延伸阅读

本章列出本书参考的经典书籍、学术论文、技术文档和行业报告，供读者深入学习。

---

## 一、经典书籍

### 信贷风控与评分卡

1. **Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring**
   - 作者：Naeem Siddiqi
   - 出版社：Wiley (2005)
   - ISBN: 978-0471753384
   - **本书地位**：评分卡领域的"圣经"，系统讲解了从数据准备到模型部署的完整流程
   - **本书引用**：第 5 章评分卡建模的 WOE 变换、分数刻度设计主要参考此书

2. **Credit Risk Analytics: Measurement Techniques, Applications, and Examples in SAS**
   - 作者：Bart Baesens
   - 出版社：Wiley (2016)
   - ISBN: 978-1118977125
   - **本书地位**：风控量化分析的百科全书，涵盖统计学、机器学习方法
   - **本书引用**：第 7 章模型评估的 AUC/KS 理论、第 6 章机器学习对比实验

3. **Handbook of Credit Data Science: From Data Collection to Credit Scoring and Risk Management**
   - 作者：Stefan Jaschke 等
   - 出版社：Springer (2022)
   - ISBN: 978-3030927837
   - **本书地位**：最新的数据科学视角下的信贷风控手册
   - **本书引用**：第 3 章数据体系、第 4 章特征工程的替代数据应用

4. **《消费信贷风险管理》**
   - 作者：陈建
   - 出版社：中国金融出版社 (2019)
   - ISBN: 978-7-5049-9876-5
   - **本书地位**：国内消费信贷风控的权威著作，结合中国监管环境
   - **本书引用**：第 1 章行业全景、第 8 章模型部署的合规要求

5. **《风控要略：互联网业务反欺诈之路》**
   - 作者：马强
   - 出版社：电子工业出版社 (2018)
   - ISBN: 978-7121345678
   - **本书地位**：互联网反欺诈实战指南
   - **本书引用**：第 11 章反欺诈建模的图模型应用

### 机器学习与数据科学

6. **The Elements of Statistical Learning: Data Mining, Inference, and Prediction (2nd Edition)**
   - 作者：Trevor Hastie, Robert Tibshirani, Jerome Friedman
   - 出版社：Springer (2009)
   - ISBN: 978-0387848570
   - **本书地位**：统计学习理论的奠基之作
   - **本书引用**：第 6 章机器学习的偏差 - 方差权衡、交叉验证理论

7. **Pattern Recognition and Machine Learning**
   - 作者：Christopher M. Bishop
   - 出版社：Springer (2006)
   - ISBN: 978-0387310732
   - **本书地位**：贝叶斯机器学习的经典教材
   - **本书引用**：第 6 章 SHAP 值的贝叶斯解释

8. **Deep Learning**
   - 作者：Ian Goodfellow, Yoshua Bengio, Aaron Courville
   - 出版社：MIT Press (2016)
   - ISBN: 978-0262035613
   - **本书地位**：深度学习领域的标准教材
   - **本书引用**：第 14 章序列模型的 LSTM/Transformer 架构、第 13 章图神经网络

9. **《机器学习》（西瓜书）**
   - 作者：周志华
   - 出版社：清华大学出版社 (2016)
   - ISBN: 978-7302407997
   - **本书地位**：中文机器学习入门经典
   - **本书引用**：第 6 章机器学习的集成学习理论

10. **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (3rd Edition)**
    - 作者：Aurélien Géron
    - 出版社：O'Reilly (2022)
    - ISBN: 978-1098125974
    - **本书地位**：实战机器学习的最佳入门书
    - **本书引用**：第 6 章 LightGBM 调参实践、Pipeline 设计

### 图神经网络与序列模型

11. **Graph Representation Learning**
    - 作者：William L. Hamilton
    - 出版社：Morgan & Claypool (2020)
    - ISBN: 978-1681736822
    - **本书地位**：图表示学习的奠基教材
    - **本书引用**：第 13 章 GCN 消息传递机制、GraphSAGE 算法

12. **Graph Neural Networks: Foundations, Frontiers, and Applications**
    - 作者：Lingfei Wu 等
    - 出版社：Springer (2022)
    - ISBN: 978-9811900440
    - **本书地位**：GNN 应用领域的最新综述
    - **本书引用**：第 13 章图模型在欺诈检测中的应用案例

13. **Deep Learning for Sequence Modeling**
    - 作者：Maxime Saintigny
    - 出版社：Packt (2021)
    - ISBN: 978-1800207264
    - **本书地位**：序列建模实战指南
    - **本书引用**：第 14 章 LSTM/注意力机制的实现

---

## 二、经典论文

### 评分卡与逻辑回归

14. **The Weight of Evidence: A Foundation for Credit Scoring**
    - 作者：W. E. Henery
    - 期刊：Journal of the Operational Research Society (1981)
    - **贡献**：WOE 变换的统计学基础

15. **Regression Shrinkage and Selection via the Lasso**
    - 作者：Robert Tibshirani
    - 期刊：Journal of the Royal Statistical Society (1996)
    - **贡献**：L1 正则化，特征选择的理论基础

### 梯度提升树

16. **Greedy Function Approximation: A Gradient Boosting Machine**
    - 作者：Jerome H. Friedman
    - 期刊：Annals of Statistics (2001)
    - **贡献**：GBM 的开创性论文

17. **XGBoost: A Scalable Tree Boosting System**
    - 作者：Tianqi Chen, Carlos Guestrin
    - 期刊：KDD (2016)
    - **贡献**：XGBoost 算法，Kaggle 竞赛神器

18. **LightGBM: A Highly Efficient Gradient Boosting Decision Tree**
    - 作者：Guolin Ke 等（微软亚洲研究院）
    - 期刊：NeurIPS (2017)
    - **贡献**：LightGBM 算法，大规模数据训练

### 模型解释性

19. **A Unified Approach to Interpreting Model Predictions (SHAP)**
    - 作者：Scott M. Lundberg, Su-In Lee
    - 期刊：NeurIPS (2017)
    - **贡献**：SHAP 值理论，模型解释的统一框架

### 图神经网络

20. **Semi-Supervised Classification with Graph Convolutional Networks**
    - 作者：Thomas N. Kipf, Max Welling
    - 期刊：ICLR (2017)
    - **贡献**：GCN 算法的简化与普及

21. **Inductive Representation Learning on Large Graphs (GraphSAGE)**
    - 作者：William L. Hamilton 等
    - 期刊：NeurIPS (2017)
    - **贡献**：GraphSAGE，适用于大规模图的归纳学习

### 序列模型

22. **Long Short-Term Memory**
    - 作者：Sepp Hochreiter, Jürgen Schmidhuber
    - 期刊：Neural Computation (1997)
    - **贡献**：LSTM 的开创性论文

23. **Attention Is All You Need (Transformer)**
    - 作者：Ashish Vaswani 等
    - 期刊：NeurIPS (2017)
    - **贡献**：Transformer 架构，颠覆序列建模

24. **Neural Machine Translation by Jointly Learning to Align and Translate (Attention Mechanism)**
    - 作者：Dzmitry Bahdanau 等
    - 期刊：ICLR (2015)
    - **贡献**：注意力机制在序列模型中的应用

### 不平衡学习

25. **SMOTE: Synthetic Minority Over-sampling Technique**
    - 作者：Nitesh V. Chawla 等
    - 期刊：Journal of Artificial Intelligence Research (2002)
    - **贡献**：SMOTE 过采样算法

26. **Learning from Imbalanced Data**
    - 作者：Haibo He, Edwardo A. Garcia
    - 期刊：IEEE Transactions on Knowledge and Data Engineering (2009)
    - **贡献**：不平衡学习的综述论文

---

## 三、行业报告与监管文件

### 中国监管框架

27. **《商业银行互联网贷款管理暂行办法》**
    - 发布机构：中国银保监会
    - 发布时间：2020 年
    - **核心要求**：风控模型可解释性、数据合规性、第三方合作管理

28. **《个人金融信息保护技术规范》**
    - 发布机构：中国人民银行
    - 标准号：JR/T 0171-2020
    - **核心要求**：个人金融信息分类、加密存储、访问控制

29. **《征信业务管理办法》**
    - 发布机构：中国人民银行
    - 发布时间：2021 年
    - **核心要求**：征信数据采集、使用、共享的合规要求

### 国际监管框架

30. **Basel II/III: International Framework for Liquidity Risk Measurement, Standards and Monitoring**
    - 发布机构：巴塞尔银行监管委员会
    - **核心内容**：内部评级法（IRB）、资本充足率要求

31. **SR 11-7: Guidance on Model Risk Management**
    - 发布机构：美联储 (Federal Reserve)
    - **核心内容**：模型验证、模型治理、模型风险管理框架

### 行业白皮书

32. **《中国消费金融行业发展报告》**
    - 发布机构：中国消费金融公司协会
    - 年度：2023
    - **核心数据**：行业规模、资产质量、风险趋势

33. **《人工智能在金融风控中的应用白皮书》**
    - 发布机构：中国人工智能学会金融科技专委会
    - 年度：2022
    - **核心内容**：AI 风控应用场景、技术路线、合规边界

---

## 四、技术文档与开源项目

### Python 数据科学生态

34. **scikit-learn Documentation**
    - 网址：https://scikit-learn.org/
    - **本书引用**：第 6 章模型训练、评估的 API 设计

35. **LightGBM Documentation**
    - 网址：https://lightgbm.readthedocs.io/
    - **本书引用**：第 6 章 LightGBM 参数详解

36. **SHAP Documentation**
    - 网址：https://shap.readthedocs.io/
    - **本书引用**：第 6 章 SHAP 值分析与可视化

37. **PyTorch Geometric Documentation**
    - 网址：https://pytorch-geometric.readthedocs.io/
    - **本书引用**：第 13 章图神经网络实现

### Kaggle 竞赛资源

38. **Home Credit Default Risk Competition**
    - 网址：https://www.kaggle.com/c/home-credit-default-risk
    - **本书引用**：第 4 章特征工程、第 6 章集成学习技巧

39. **Otto Group Product Classification Competition**
    - 网址：https://www.kaggle.com/c/otto-group-product-classification-challenge
    - **本书引用**：多分类问题、类别不平衡处理

### 开源代码库

40. **optbinning: Optimal Binning for Python**
    - 网址：https://github.com/guillermo-navas-palencia/optbinning
    - **本书引用**：第 4 章最优分箱算法

41. **Credit Scoring Toolkit**
    - 网址：https://github.com/credit-scoring-toolkit
    - **本书引用**：评分卡开发的开源实现参考

---

## 五、在线学习资源

### 课程与教程

42. **Machine Learning Specialization (Coursera)**
    - 讲师：Andrew Ng
    - 网址：https://www.coursera.org/specializations/machine-learning-introduction
    - **推荐章节**：监督学习、正则化、集成方法

43. **Deep Learning Specialization (Coursera)**
    - 讲师：Andrew Ng
    - 网址：https://www.coursera.org/specializations/deep-learning
    - **推荐章节**：序列模型、注意力机制

44. **Graph Neural Networks Tutorial**
    - 讲师：William L. Hamilton
    - 网址：http://www.cs.mcgill.ca/~wlh/grl_book/
    - **推荐章节**：GCN、GraphSAGE、图注意力网络

### 博客与社区

45. **Towards Data Science**
    - 网址：https://towardsdatascience.com/
    - **推荐主题**：Credit Scoring、Feature Engineering、Imbalanced Learning

46. **Kaggle Learn**
    - 网址：https://www.kaggle.com/learn
    - **推荐课程**：Feature Engineering、XGBoost、Deep Learning

47. **知乎 - 风控话题**
    - 网址：https://www.zhihu.com/topic/19552841
    - **推荐关注**：风控从业者分享实战经验

---

## 六、推荐阅读路径

### 入门级（0-1 年经验）
1. 《机器学习》（西瓜书）第 1-6 章
2. 本书第 1-7 章
3. scikit-learn 官方教程

### 进阶级（1-3 年经验）
1. Credit Risk Scorecards (Siddiqi)
2. 本书第 13-15 章
3. Kaggle 竞赛实战

### 专家级（3 年以上经验）
1. The Elements of Statistical Learning
2. Graph Neural Networks 相关论文
3. 监管文件与行业白皮书

---

> **注**：以上文献按引用优先级排序，建议读者根据实际需求选择性阅读。本书代码实现主要参考官方文档和开源项目，理论推导主要参考经典教材和论文。
