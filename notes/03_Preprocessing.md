# Module 03 笔记: 数据预处理与缺失值插补

> 本模块的核心命题：**不同的缺失值处理方法如何通过"样本量效应"和"分布失真效应"两条路径影响模型的区分能力、召回率和校准度。**

---

## 核心概念梳理

### 数据预处理与缺失值插补是什么？

数据预处理是机器学习流程中决定输入质量的关键环节。在医学数据中，缺失值几乎无处不在——该教程所使用的巴西癌症登记数据中，30/38 列存在缺失，18 列缺失率超过 50%。缺失值插补（Imputation）是指用合理的估计值替代缺失数据的过程，其本质是对数据分布的重构。

在医学场景中，缺失值插补的选择具有特殊的严谨性要求。临床预测模型的输出概率直接影响医生的决策，因此不仅要关注模型的区分能力（AUC），更需关注预测概率的校准度（Calibration）——即模型输出的 0.8 是否真的对应 80% 的生存概率。不同的插补策略通过改变特征的数值分布，间接影响逻辑回归的决策边界，最终体现为 AUC、Recall 和 Brier Score 三个维度的变化。

### 关键方法

| 方法 | 核心思想 | 优点 | 缺点 |
|------|---------|------|------|
| **Complete Case** | 删除任何含缺失的行 | 简单、无偏（MCAR下） | 损失样本量，可能损失信息 |
| **均值插补** | 用均值（数值）/众数（分类）填充 | 计算快速，易实现 | 低估方差，破坏协方差结构 |
| **KNN 插补** | 用 k 个最近邻的特征均值填充 | 保留局部结构 | 时间复杂度 O(n^2×d)，大数据下代价高 |
| **MICE 插补** | 用其他特征迭代预测缺失值 | 保留变量间关系，医学黄金标准 | 需多轮迭代，实现较复杂 |

### 评价指标体系

| 维度 | 指标 | 所衡量内容 |
|------|------|-----------|
| 区分能力（Discrimination） | AUC (ROC) | 模型能否区分存活与死亡患者 |
| 少数类捕捉（Sensitivity） | Recall | 模型能否找到真正的存活患者 |
| 概率校准（Calibration） | Brier Score | 模型预测的概率是否可靠 |

---

## 代码精读

### Block 1: 环境配置与数据加载

```python
import pandas as pd                   # 数据处理
import numpy as np                    # 数值计算
import matplotlib.pyplot as plt       # 可视化
from sklearn.impute import SimpleImputer, KNNImputer  # 均值/KNN插补
from sklearn.impute import IterativeImputer            # MICE插补
from sklearn.linear_model import LogisticRegression    # 评估模型
from sklearn.metrics import roc_auc_score, recall_score, brier_score_loss  # 评估指标
from sklearn.calibration import calibration_curve      # 校准曲线
```

要点：
1. 使用 `IterativeImputer`（sklearn 的 MICE 实现）作为多重插补工具
2. 同时使用 AUC、Recall、Brier Score 三个指标，避免单一指标的局限
3. 校准曲线用于可视化概率输出的可靠性

### Block 2: 数据加载与特征选择

```python
df = pd.read_csv(DATA_PATH, low_memory=False, encoding='latin-1')  # 读取CSV
df['target'] = df['Status.Vital'].map({'VIVO': 1, 'MORTO': 0})     # 二分类标签编码
df = df.dropna(subset=['target'])                                    # 删除无标签样本

# 选择特征：兼顾缺失率、数据类型、临床意义
features_config = {
    'Age': 'numerical',              # 0.15% 缺失，连续数值
    'year': 'numerical',             # 0.00% 缺失，连续数值
    'Gender': 'categorical',         # 二分类
    'Diagnostic.means': 'categorical',  # 诊断方式（7类）
    'Raca.Color': 'categorical',     # 人种（5类），15.31% 缺失
}

# 随机采样 N_SAMPLES 个样本（平衡计算开销与统计可靠性）
np.random.seed(RANDOM_STATE)
sample_idx = np.random.choice(len(df), N_SAMPLES, replace=False)
df_sample = df.iloc[sample_idx].copy()
```

要点：
1. 特征选择遵循"低缺失率、高临床意义"原则
2. 使用 `stratify=y` 保持训练/测试集的标签分布一致
3. 采样 80,000 条以平衡计算效率

### Block 3: 数据划分与编码

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)  # 30%测试集，保持标签比例

# 对分类特征做 Label Encoding
for col in categorical_features:
    le = LabelEncoder()
    non_null_train = X_train[col].dropna()              # 用非缺失值拟合编码器
    le.fit(non_null_train.astype(str))
    most_common = non_null_train.value_counts().index[0] # 最常见类别（用于未知类别回退）
    # 处理测试集中的新类别：替换为最常见类别
    def transform_with_unknown(x):
        if pd.isna(x): return np.nan
        x_str = str(x)
        if x_str in le.classes_: return le.transform([x_str])[0]
        else: return le.transform([most_common])[0]
    X_train[col] = X_train[col].apply(transform_with_unknown)
    X_test[col] = X_test[col].apply(transform_with_unknown)
```

要点：
1. **关键防泄漏操作**：编码器只在训练集上 `fit`，测试集只 `transform`
2. 处理测试集中的未见类别（Unknown Category）用最常见类别回退
3. 编码后转换为 `float` 以便后续插补

### Block 4: 四种插补策略执行

```python
methods = {
    'Complete Case': {
        'imputer': None,                      # 不插补，直接删除缺失行
        'color': '#7f8c8d'},
    'Mean Imputation': {
        'imputer': SimpleImputer(strategy='mean'),  # 均值（数值）/众数（分类）
        'color': '#3498db'},
    'KNN Imputation': {
        'imputer': KNNImputer(n_neighbors=5, weights='distance'),  # k=5，距离加权
        'color': '#e67e22'},
    'MICE Imputation': {
        'imputer': IterativeImputer(max_iter=10, random_state=RANDOM_STATE),  # 10轮迭代
        'color': '#9b59b6'}
}

for method_name, config in methods.items():
    if method_name == 'Complete Case':
        # 删除任何含缺失的行
        train_mask = X_train.isnull().any(axis=1)   # 标记含缺失的行
        X_train_imp = X_train[~train_mask].copy()    # 保留完整行
        y_train_imp = y_train[~train_mask]
    else:
        imp = config['imputer']
        X_train_imp = pd.DataFrame(
            imp.fit_transform(X_train),              # fit_transform 训练集
            columns=feature_names)
        X_test_imp = pd.DataFrame(
            imp.transform(X_test),                   # transform 测试集（防泄漏）
            columns=feature_names)

    # 标准化 + 逻辑回归 + 训练评估
    scaler = StandardScaler()
    X_train_imp[num_cols] = scaler.fit_transform(X_train_imp[num_cols])  # 训练集fit
    X_test_imp[num_cols] = scaler.transform(X_test_imp[num_cols])        # 测试集transform

    lr = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE)
    lr.fit(X_train_imp, y_train_imp)
    y_pred_proba = lr.predict_proba(X_test_imp)[:, 1]
    auc = roc_auc_score(y_test_imp, y_pred_proba)
    recall = recall_score(y_test_imp, lr.predict(X_test_imp), pos_label=1)
    brier = brier_score_loss(y_test_imp, y_pred_proba)
```

要点：
1. **防泄漏规范**：插补器在训练集上 `fit_transform`，在测试集上仅 `transform`
2. 使用 `class_weight='balanced'` 处理 VIVO/MORTO 的轻度不平衡
3. `IterativeImputer(max_iter=10)` 表示 MICE 执行 10 轮迭代直到收敛

### Block 5: 结果对比与关键发现

| 方法 | AUC | Recall | Brier | 训练样本 | 耗时 |
|------|-----|--------|-------|---------|------|
| Complete Case | **0.8644** | **0.9240** | 0.1493 | 47,270 | 0.1s |
| Mean Imputation | 0.8632 | 0.9236 | 0.1466 | 56,000 | 0.1s |
| KNN Imputation | 0.8628 | 0.9237 | 0.1467 | 56,000 | **20.5s** |
| MICE Imputation | 0.8637 | 0.9223 | **0.1462** | 56,000 | 0.1s |

**核心发现**：
- 四种方法性能差异极小（ΔAUC < 0.002），因为最高缺失率仅 15%
- Complete Case 在 AUC/Recall 上反而最优：保留了 84% 样本，信息量足够
- MICE 的 Brier Score 最低（校准度最优）：利用变量间联合分布保留概率关系
- KNN 耗时是其他方法的 200+ 倍，性能无优势

### Block 6: 分布影响可视化（Age 列为例）

```python
# 直方图对比：原始 vs 均值插补 / KNN / MICE
ax.hist(age_original, bins=60, alpha=0.5, density=True,
        color='#7f8c8d', label=f'Original (n={len(age_original):,})')
ax.hist(age_data['Mean Imputation'], bins=60, alpha=0.4, density=True,
        color='#3498db', label=f'Mean Imp.')

# 方差对比条形图
variances = [age_original.var(), mean_var, knn_var, mice_var]
ax.barh(var_labels, variances, color=colors_variance)
```

要点：
1. 均值插补在均值点（~64 岁）产生异常的"尖峰"——所有缺失 Age 都被设为同一值
2. KNN 和 MICE 的分布更接近原始数据的自然形态
3. 均值插补后方差最小，验证了"低估方差"的理论预测

### Block 7: ROC 曲线与校准曲线

```python
# ROC 曲线
for r in results:
    fpr, tpr, _ = roc_curve(y_true_roc, y_prob_roc)
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f'{method} (AUC = {auc_val:.4f})')

# 校准曲线
for r in results:
    ax.plot(r['Calibration_Pred'], r['Calibration_True'],
            marker='o', color=r['Color'], linewidth=2, markersize=8)
ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
```

要点：
1. 四条 ROC 曲线几乎重叠，反映各方法区分能力差异极小
2. 校准曲线中 MICE 最接近对角线（Brier 最小），验证其校准度优势
3. 校准度对插补方法的敏感度高于 AUC——因为校准关注概率绝对值，分布失真直接影响它

---

## 关键收获

1. **缺失率决定方法差异大小**：本教程缺失率最高 15%，四种方法性能几乎无差异。当缺失率超过 30% 时，Complete Case 的样本损失加剧，插补方法的优势将逐渐显现。

2. **均值插补的隐形危害**：均值的数学期望不变，但方差被压缩，破坏了特征间的协方差结构。对于回归系数，这会导致向零的衰减偏倚（attenuation bias），使系数的标准误被低估，进而影响统计推断的可靠性。

3. **MICE 在医学场景的适用性**：医学数据缺失多为 MAR（Missing at Random）机制，MICE 充分利用变量间的联合分布进行多元插补，在校准度上具有系统优势。该方法生成的插补值保留了原始分布的自然形态。

4. **KNN 插补的计算瓶颈**：时间复杂度为 O(n^2 × d)，本实验 56,000 样本即需 20.5s。当样本量增长至 50 万时预计需 27 分钟，在百万级医学数据中间接近不可行。KD-Tree/Ball-Tree 搜索加速可作为优化方向。

5. **三指标评价的必要性**：AUC 高不代表 Recall 高（可能通过降低阈值获得），Recall 高不代表校准好（可能过度预测正类），校准好不代表区分好（可能所有样本输出相同概率）。

6. **插补影响模型的因果链**：缺失值→插补方式→特征分布改变（均值压缩/方差缩小）→模型决策边界偏移→AUC/Recall/校准度变化。

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 均值插补后方差缩小 | 所有缺失值被替换为同一数值（均值） | 使用 MICE 或 KNN 保留自然方差 |
| KNN 插补在大数据上极慢 | O(n²) 距离计算复杂度 | 使用 KD-Tree 加速或先降采样再插补 |
| Complete Case 删除过多样本 | 任何列有缺失即删除整行 | 仅删除高缺失率特征的对应行；或使用插补 |
| 分类变量如何插补 | 均值插补不适用于分类变量 | 使用众数插补或 MICE 的决策树回归变体 |
| Brier Score 有改善但 AUC 无变化 | 校准度改善不影响排序能力 | 需根据业务需求选择：临床决策关注校准，筛查关注区分 |

---

## 与其他模块的联系

- **前置模块**：Module 02（EDA 与统计检验）发现 30/38 列缺失、所有特征均显著但效应量差异大，为本模块提供了"需要插补，且需关注效应量"的动机
- **后续模块**：Module 04（特征工程）中构造的新特征（如 Age_Sq、Is_Child）也面临缺失值问题，插补策略的选择直接影响构造特征的质量
- **与研究工作的联系**：在 EcMurJ 虚拟筛选研究中，分子描述符（如 LogP、分子量）存在计算失败导致的缺失值。当 236K 化合物库中某描述符因 RDKit 计算异常而缺失时，需根据缺失机制选择策略：若因分子过大导致（MNAR），简单删除可能产生偏倚；若为随机缺失（MAR），MICE 或 KNN 插补可保留更多候选化合物。宏基因组研究中，物种丰度矩阵的高缺失率（大量零值代表真实不表达 + 技术性缺失）需要区分"零膨胀"与"真正缺失"，这与本模块的 MCAR/MAR/MNAR 判断框架直接相关。

---

## 参考资料

- 教程原文：`ml4health-main/jupyter/03_preprocessing_imputation.ipynb`
- 讲义：`ml4health-main/lectures/03_preprocessing_imputation_teaching_doc.md`
- Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*. Wiley.
- van Buuren, S. (2018). *Flexible Imputation of Missing Data*. CRC Press.
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825-2830.
