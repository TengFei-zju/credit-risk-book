## 数据目录

请将天池比赛数据集文件放入此目录：

```
data/
├── train.csv       ← 天池训练集（约 800,000 行，47列）
├── testA.csv       ← 天池测试集A（约 200,000 行，46列）
└── processed/      ← 自动生成（运行 notebooks 后产生）
    ├── train_clean.parquet
    ├── test_clean.parquet
    ├── X_train_features.parquet
    ├── X_test_features.parquet
    ├── y_train.parquet
    ├── selected_features.json
    └── submission.csv
```

### 数据下载

1. 访问：https://tianchi.aliyun.com/competition/entrance/531830
2. 注册/登录天池账号
3. 点击「数据」页签，下载 `train.csv` 和 `testA.csv`
4. 将文件放入本目录

### 数据字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| id | int | 贷款唯一标识 |
| loanAmnt | float | 贷款金额 |
| term | int | 贷款期限（36 或 60 个月） |
| interestRate | float | 贷款利率（%） |
| installment | float | 月还款金额 |
| grade | str | 贷款等级（A~G） |
| subGrade | str | 贷款子等级（A1~G5） |
| employmentTitle | float | 就业职称（编码） |
| employmentLength | str | 就业年限（'< 1 year' ~ '10+ years'） |
| homeOwnership | str | 房屋所有权状态 |
| annualIncome | float | 年收入 |
| verificationStatus | str | 收入核验状态 |
| issueDate | str | 放款日期（格式：YYYY-MM-DD） |
| **isDefault** | **int** | **目标变量：1=违约，0=正常** |
| purpose | str | 借款目的 |
| postCode | float | 邮政编码（编码） |
| regionCode | float | 地区代码 |
| dti | float | 负债收入比（%） |
| delinquency_2years | float | 近2年逾期次数 |
| ficoRangeLow | float | FICO信用分下限 |
| ficoRangeHigh | float | FICO信用分上限 |
| openAcc | float | 开放信用账户数 |
| pubRec | float | 负面公开记录数 |
| pubRecBankruptcies | float | 公开破产记录数 |
| revolBal | float | 循环信用余额 |
| revolUtil | float | 循环信用利用率（%） |
| totalAcc | float | 信用账户总数 |
| initialListStatus | str | 初始上市状态（w/f） |
| applicationType | str | 申请类型（Individual / Joint App） |
| earliesCreditLine | str | 最早信用账户开立日期 |
| title | float | 贷款标题（编码） |
| policyCode | float | 政策代码（全为1，无信息） |
| n0~n14 | float | 匿名特征（共15个） |
