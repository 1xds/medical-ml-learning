# Module 04 笔记: 特征工程

> 本模块的核心命题：**特征质量决定模型上限——通过标准化加速模型收敛、通过特征构造注入领域知识，从而提升模型性能。**

---

## 核心概念梳理

### 特征工程是什么？

特征工程是从原始数据中提取和构造特征的过程，是机器学习流程中投入产出比最高的环节。在医学预后预测中，调整超参数可能带来 1-2% 的性能提升，而一个精心设计的构造特征（如基于年龄分组的交互特征）可以带来 5-10% 甚至更高的提升。特征工程的核心公式为：

**好特征 = 领域知识 + 数据形态 + 模型特性**

本模块分为两个部分：标准化方法比较和特征构造。标准化统一特征的量纲以加速模型收敛；特征构造则通过将医学领域知识编码为新变量来为模型提供额外的信息。

### 标准化方法对比

| 方法 | 公式 | 中心化基准 | 缩放基准 | 离群值敏感性 |
|------|------|-----------|---------|------------|
| Raw（原始） | x | — | — | — |
| StandardScaler | (x - μ) / σ | 均值 μ | 标准差 σ | **敏感**（离群值拉偏 μ 和 σ） |
| RobustScaler | (x - median) / IQR | 中位数 | IQR（Q3-Q1）| **不敏感**（中位数和 IQR 不受离群值影响） |

### 构造特征一览

| 新特征 | 基于 | 领域知识动机 |
|--------|------|-------------|
| **Age_Group** | Age | 不同年龄段的癌症预后差异显著 |
| **Age_Sq** | Age | 年龄与死亡率可能存在 U 型关系（婴儿和老年人风险高） |
| **Year_Decade** | year | 医疗技术进步使诊断年代影响存活率 |
| **Year_From_2000** | year | 存活率随时间的线性改善趋势 |
| **Gender_x_AgeGroup** | Gender × Age_Group | 性别的效应在不同年龄段表现不同 |
| **Is_Child** | Age | 儿童癌症在病理和治疗上具有特殊性 |
| **Age_Centered** | Age - 60 | 提高截距项的可解释性 |

---

## 代码精读

### Block 1: 环境配置与数据准备

```python
import pandas as pd                            # 数据处理
import numpy as np                             # 数值计算
from sklearn.preprocessing import StandardScaler, RobustScaler  # 标准化方法
from sklearn.impute import SimpleImputer        # 均值插补
from sklearn.linear_model import LogisticRegression  # 评估模型
from sklearn.metrics import roc_auc_score, recall_score, brier_score_loss  # 多指标评估

df = pd.read_csv(DATA_PATH, low_memory=False)   # 读取癌症数据
df['target'] = df['Status.Vital'].map({'VIVO': 1, 'MORTO': 0})  # 标签编码
```

要点：
1. 同时引入 `StandardScaler` 和 `RobustScaler` 用于对比
2. 采用标准化→训练→评估的流水线，核心是比较收敛速度和系数可解释性

### Block 2: 特征编码（Label Encoding）

```python
base_features = ['Age', 'year', 'Gender', 'Code.Profession', 'Diagnostic.means', 'Raca.Color']
cat_cols = ['Gender', 'Diagnostic.means', 'Raca.Color']  # 分类变量列表
for col in cat_cols:
    le = LabelEncoder()
    le.fit(non_null.astype(str))            # 拟合编码器
    most_common = non_null.value_counts().index[0]  # 最常见类别（未见类别回退）
    def encode(x):
        if pd.isna(x): return np.nan        # 保留缺失值(后续插补处理)
        xs = str(x)
        return le.transform([xs])[0] if xs in le.classes_ else le.transform([most_common])[0]
    df_feat[col] = df_feat[col].apply(encode)
df_feat = df_feat.astype(float)             # 统一转为浮点数
```

要点：
1. 编码器只在训练集上 `fit`，但此处是在采样后全数据上 `fit`（编码不同于标准化，不涉及目标变量）
2. 保留缺失值为 NaN，后续由 `SimpleImputer` 统一处理

### Block 3: 标准化方法对比实验

```python
scalers = {
    'Raw (No Scaling)': None,                 # 不做任何缩放
    'StandardScaler': StandardScaler(),       # Z-score标准化
    'RobustScaler': RobustScaler(),           # 中位数-IQR标准化
}

for name, scaler in scalers.items():
    if scaler is not None:
        X_train = scaler.fit_transform(X_train_raw)   # 训练集fit
        X_test = scaler.transform(X_test_raw)          # 测试集transform（防泄漏）

    lr = LogisticRegression(class_weight='balanced', max_iter=5000, solver='lbfgs')
    lr.fit(X_train, y_train)
    n_iter = lr.n_iter_[0]                   # 实际迭代次数
    coef_mean_abs = np.abs(lr.coef_[0]).mean()  # 系数绝对值的均值
```

要点：
1. **关键发现**：迭代次数差异巨大——Raw 需 174 次，StandardScaler 仅需 10 次
2. 系数绝对值均值差异显著——标准化使不同量纲的系数变得可比较
3. `lbfgs` 求解器对特征尺度敏感，标准化后梯度方向不再振荡

### Block 4: 标准化性能对比可视化

```python
# AUC/Recall/Brier 柱状图
for i, metric in enumerate(['AUC', 'Recall', 'Brier']):
    bars = ax.bar(names, vals, color=['#7f8c8d', '#3498db', '#e67e22'])
    # 三方法性能几乎一致

# 系数对比图
for idx, (name, scaler) in enumerate(scalers.items()):
    x_pos = np.arange(len(feature_names)) + idx * 0.25
    ax_coef.bar(x_pos, np.abs(lr.coef_[0]), width=0.2, label=name)
```

要点：
- 标准化后 AUC/Recall/Brier 几乎完全一致——标准化不改变模型预测能力
- 系数可比较性得到体现：标准化后可直接从系数大小判断特征重要性

### Block 5: 特征构造代码

```python
# 年龄分组（医学重要意义）
def age_group(a):
    if pd.isna(a): return np.nan
    a = float(a)
    if a < 18: return 0      # 儿童/青少年
    elif a < 40: return 1    # 中青年
    elif a < 60: return 2    # 中年
    elif a < 75: return 3    # 老年
    else: return 4           # 高龄

df_eng['Age_Group'] = df_eng['Age'].apply(age_group)
df_eng['Age_Sq'] = df_eng['Age'] ** 2            # 平方项捕捉非线性效应
df_eng['Year_Decade'] = df_eng['year'].apply(year_decade)  # 诊断年代分组
df_eng['Year_From_2000'] = df_eng['year'] - 2000  # 连续趋势
df_eng['Gender_x_AgeGroup'] = df_eng['Gender'] * df_eng['Age_Group']  # 交互项
df_eng['Is_Child'] = (df_eng['Age'] < 18).astype(float)  # 儿童标志
df_eng['Age_Centered'] = df_eng['Age'] - 60              # 中心化年龄
```

要点：
1. 每个特征都有对应的医学领域知识支撑，非盲目构造
2. `Age_Group` 的分类阈值基于医学共识（儿童/青壮年/中年/老年/高龄）
3. `Gender_x_AgeGroup` 捕捉的是交互效应——性别对生存率的影响在不同年龄段可能不同

### Block 6: 特征构造后模型对比

```python
fe_sets = {
    'Base (6 features)': (X_base_train_s, X_base_test_s, 6),
    'Engineered (13 features)': (X_all_train_s, X_all_test_s, 13),
}

for set_name, (X_tr, X_te, n_feat) in fe_sets.items():
    lr_fe = LogisticRegression(class_weight='balanced', max_iter=5000)
    lr_fe.fit(X_tr, y_eng_train)
    auc_fe = roc_auc_score(y_eng_test, lr_fe.predict_proba(X_te)[:, 1])
    recall_fe = recall_score(y_eng_test, lr_fe.predict(X_te), pos_label=1)
    brier_fe = brier_score_loss(y_eng_test, lr_fe.predict_proba(X_te)[:, 1])
    ap_fe = average_precision_score(y_eng_test, lr_fe.predict_proba(X_te)[:, 1])
```

**对比结果**：

| 指标 | Base（6特征） | Engineered（13特征） | 变化 |
|------|-------------|-------------------|------|
| AUC | 0.8698 | **0.8737** | +0.0039 |
| Recall | **0.9026** | 0.8886 | -0.0140 |
| Brier | 0.1437 | **0.1409** | -0.0028 |
| PR-AUC | 0.7412 | **0.7530** | +0.0118 |

要点：
1. 特征构造提升了 AUC 和校准度（Brier 降低），但 Recall 下降
2. 这是"反直觉"结果：更多信息不一定在所有指标上都更好
3. 原因：13 个特征的决策边界更复杂，在轻度不平衡数据中可能降低少数类召回

---

## 关键收获

1. **标准化不改变性能，但改变效率**：174 次迭代降至 10 次，速度提升约 17 倍。从数学角度，标准化将损失函数的等高线从"椭圆"变为"圆"，优化路径从之字形变为直线。

2. **标准化后系数可比较**：未标准化时，`Code.Profession` 系数仅 -0.001（看似不重要），实则是量纲假象（该特征取值范围约 0-10）。标准化后系数直观反映特征对 log-odds 的贡献大小。

3. **StandardScaler vs RobustScaler 的选择**：数据有离群值或严重偏态时选 RobustScaler（如 `Code.Profession` 偏度 0.94）；近似正态时选 StandardScaler。两者的区分与梯度下降的数值稳定性直接相关。

4. **特征构造等于编码领域知识**：`Age_Group` 捕捉年龄的非线性效应，`Gender_x_AgeGroup` 捕捉交互效应——这些构造的意义来自医学认知，而非数据本身的统计学模式。

5. **特征质量决定模型上限**：即使使用简单的逻辑回归，优质构造特征也能显著提升性能。但性能提升并非均匀地分布给所有指标。

6. **多指标评估的必要性**：AUC 上升 + Brier 下降 + Recall 下降的组合表明，特征构造增加了模型的整体信息量，但非线性特征的加入使决策边界更复杂，可能牺牲少数类的敏感度。

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 标准化后 AUC 完全不变 | 逻辑回归是线性模型，标准化只是重新参数化 | 正常现象；标准化不改变模型预测能力 |
| 标准化后系数解释困难 | RobustScaler 改变了量纲 | 统一为"每 IQR 变化对 log-odds 的影响" |
| 构造特征后 Recall 下降 | 决策边界更复杂，少数类拟合更难 | 调整分类阈值或使用 PR-AUC 替代 Recall |
| Age 和 Age_Sq 高度相关（r≈0.98） | 平方项天然与原始值相关 | 后续 VIF/相关性分析处理共线性 |
| 交互特征数量爆炸 | n 个特征两两交互产生 n(n-1)/2 个新特征 | 基于领域知识筛选有意义的交互项 |

---

## 与其他模块的联系

- **前置模块**：Module 03（数据预处理）的均值插补为本模块提供了完整的数据集，标准化实验也展示了"先插补后标准化"的标准流程
- **后续模块**：Module 05（特征选择）将处理本模块引入的共线性问题（Age 与 Age_Sq 相关性 0.98），Module 06（降维）将从另一个角度（PCA 特征变换）处理特征冗余


---

## 参考资料

- 教程原文：`ml4health-main/jupyter/04_feature_engineering.ipynb`
- 讲义：`ml4health-main/lectures/04_feature_engineering_teaching_doc.md`
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
- Kuhn, M. & Johnson, K. (2019). *Feature Engineering and Selection*. CRC Press.
