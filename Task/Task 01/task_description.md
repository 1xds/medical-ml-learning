# 数据集介绍

## 中国人群心脏病发作风险与健康因素数据集

本数据集是一份面向**心血管疾病（CVD）流行病学与预测建模研究**的多维度微观数据集，聚焦于中国成年人群，系统记录了与**心肌梗死（心脏病发作）**相关的个人健康特征、生活方式、环境暴露和区域医疗资源信息。

所有变量可归为六大类，覆盖**生物医学、行为环境、社会经济和医疗系统**层面。

---

# （一）个体标识与基本信息

| 变量名 | 类型 | 说明 |
|---------|------|------|
| Patient_ID | 唯一标识 | 匿名化患者编号 |
| Age | 数值（岁） | 年龄范围 25~95 |
| Gender | 分类（Male/Female） | 性别 |

---

# （二）经典心血管风险因子（临床与生活方式）

| 变量名 | 类型 | 说明 |
|---------|------|------|
| Smoking_Status | 二分类（Smoker/Non-Smoker） | 当前吸烟状态 |
| Hypertension | 二分类（Yes/No） | 是否有高血压诊断 |
| Diabetes | 二分类（Yes/No） | 是否患糖尿病 |
| Obesity | 二分类（Yes/No） | 是否肥胖（基于BMI或腰围） |
| Cholesterol_Level | 三分类（High/Normal/Low） | 总胆固醇水平 |
| Blood_Pressure | 数值（mmHg） | 收缩压或舒张压（可明确为收缩压） |
| Chronic_Kidney_Disease | 二分类（Yes/No） | 是否合并慢性肾病 |
| Family_History_CVD | 二分类（Yes/No） | 直系亲属心血管病史 |
| Previous_Heart_Attack | 二分类（Yes/No） | 既往心脏病发作史 |

---

# （三）行为与环境暴露

| 变量名 | 类型 | 说明 |
|---------|------|------|
| Air_Pollution_Exposure | 三分类（Low/Medium/High） | 日常空气污染暴露等级（基于PM2.5等） |
| Physical_Activity | 三分类（Low/Medium/High） | 体力活动水平 |
| Diet_Score | 三分类（Healthy/Moderate/Poor） | 饮食健康程度（如蔬果、盐摄入） |
| Stress_Level | 三分类（Low/Medium/High） | 自评或心理量表压力水平 |
| Alcohol_Consumption | 二分类（Yes/No） | 是否定期饮酒 |

---

# （四）医疗资源可及性与利用

| 变量名 | 类型 | 说明 |
|---------|------|------|
| Healthcare_Access | 三分类（Good/Moderate/Poor） | 就医便利性与医保覆盖综合评分 |
| Hospital_Availability | 三分类（High/Medium/Low） | 所在区域医院密度/床位资源 |
| TCM_Use | 二分类（Yes/No） | 是否定期使用中医药服务（体现中国医疗特色） |

---

# （五）社会经济与教育

| 变量名 | 类型 | 说明 |
|---------|------|------|
| Employment_Status | 三分类（Employed/Unemployed/Retired） | 就业状态 |
| Education_Level | 四分类（None/Primary/Secondary/Higher） | 教育程度 |
| Income_Level | 三分类（Low/Middle/High） | 家庭收入水平 |

---

# （六）地理与区域特征

| 变量名 | 类型 | 说明 |
|---------|------|------|
| Rural_or_Urban | 二分类（Rural/Urban） | 城乡属性 |
| Region | 五分类（Eastern/Western/Northern/Southern/Central） | 中国五大地理大区 |
| Province | 字符串（如 Beijing、Gansu） | 具体省份（可进一步细粒度分析） |

---

# （七）综合性评估与目标变量

| 变量名 | 类型 | 说明 |
|---------|------|------|
| CVD_Risk_Score | 数值（0~100） | 综合心血管风险评分（可由传统公式或模型生成） |
| Heart_Attack | 二分类（Yes/No） | 目标变量：研究期内是否发生急性心肌梗死 |
