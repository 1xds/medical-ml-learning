# Module 08 笔记: 数据划分与交叉验证

> 本模块的核心命题：**单次数据划分的评估结果具有高度不稳定性，交叉验证通过多次划分取平均的方式提供更可靠的泛化能力估计。**

---

## 核心概念梳理

### 交叉验证是什么？

交叉验证（Cross-Validation, CV）是一种模型评估方法，其核心思想是将数据集划分为若干子集，轮流将其中一个子集作为验证集、其余作为训练集，最终取所有轮次评估指标的均值作为模型泛化能力的估计。该方法解决了单次 Train-Test Split 中因随机种子不同而导致评估结果波动的问题。

在医学机器学习场景中，交叉验证的意义尤为突出。医学数据集通常存在类别不平衡（如阳性样本占比低于 10%）、样本量有限等问题，单次划分可能使验证集中少数类样本过少，导致 AUC 等指标产生剧烈波动。交叉验证通过确保每个样本恰好参与一次验证，最大限度地利用有限数据，同时通过多次评估降低随机性影响。

### 关键方法概览

| 方法 | 核心特征 | 模型数 | 推荐场景 |
|------|---------|--------|---------|
| Train-Test Split | 单次划分 | 1 | 快速探索 |
| K-Fold CV | K 折轮流验证 | K | 通用默认选择 |
| Stratified K-Fold | 保持每折类别比例 | K | 医学数据标准做法 |
| Repeated K-Fold | 多次重复取平均 | K×R | 更精确的均值估计 |
| Repeated Stratified K-Fold | 分层+重复 | K×R | 不平衡+高可靠性 |
| LOOCV | 每次留一个样本 | n | n < 100 |
| Nested CV | 内外双层 CV | 外×内 | 高水平论文标配 |

### 三大约束框架

评估方法的选择本质上是在偏差（Bias）、方差（Variance）和计算成本（Cost）三者之间进行权衡：

- **偏差**：评估值与真实泛化能力的差距。K 越大，训练集越接近全数据集，偏差越小。
- **方差**：评估值自身的稳定性。K 越大，各折训练集越相似，模型间相关性越高，方差可能增大。
- **计算成本**：需要训练的模型数量，随 K 和重复次数线性增长。

---

## 代码精读

### Block 1: 数据加载与预处理 Pipeline

```python
# 数据加载与目标变量编码
df = pd.read_csv(DATA_PATH, low_memory=False, encoding='latin-1')  # 读取癌症数据
df['target'] = df['Status.Vital'].map({'VIVO': 1, 'MORTO': 0})    # 存活=1, 死亡=0
df = df.dropna(subset=['target'])                                   # 剔除目标缺失行

# 随机采样至 20,000 样本
np.random.seed(RANDOM_STATE)                                        # 固定随机种子
if len(df) > N_SAMPLES:                                             # 超过上限则采样
    idx = np.random.choice(len(df), N_SAMPLES, replace=False)
    df = df.iloc[idx].copy()

# 类别特征编码
for col in cat_cols:                                                # 遍历分类列
    le = LabelEncoder()                                             # 初始化编码器
    non_null = df_feat[col].dropna().astype(str)                    # 非空值转字符串
    le.fit(non_null)                                                # 拟合编码器
    df_feat[col] = df_feat[col].apply(encode)                       # 应用编码

# 构建预处理+建模 Pipeline
def create_pipeline(C=1.0):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),              # 中位数插补
        ('scaler', StandardScaler()),                               # 标准化
        ('lr', LogisticRegression(C=C, class_weight='balanced',     # 逻辑回归
                                   max_iter=5000, random_state=RANDOM_STATE))
    ])
```

**要点说明：**

1. **Pipeline 封装**：将插补、标准化、建模封装为一个 Pipeline，确保交叉验证的每折独立执行预处理，防止数据泄漏。
2. **class_weight='balanced'**：自动调整类别权重，应对 VIVO（41.15%）与 MORTO（58.85%）的比例差异。
3. **VIVO 占比 41.15%**：属于轻度不平衡，但分层抽样仍能进一步稳定评估。

### Block 2: Train-Test Split 不稳定性演示

```python
# 20 次不同随机种子的 80/20 划分
split_results = []
n_trials = 20                                                      # 重复20次

for seed in range(n_trials):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)        # 不同种子划分
    pipe = create_pipeline()
    pipe.fit(X_tr, y_tr)                                           # 训练模型
    y_prob = pipe.predict_proba(X_te)[:, 1]                        # 预测概率
    auc = roc_auc_score(y_te, y_prob)                              # 计算 AUC
    split_results.append({'Seed': seed, 'AUC': auc})

# 统计结果
# Mean AUC = 0.8920, Std = 0.0040
# Min = 0.8839, Max = 0.8990, 极差 = 0.0151
```

**实验结果：**

| 统计量 | 值 | 解读 |
|--------|------|------|
| Mean AUC | 0.8920 | 中心趋势 |
| Std (σ) | 0.0040 | 单次划分的标准误 |
| Min AUC | 0.8839 | "最差"的一次划分 |
| Max AUC | 0.8990 | "最好"的一次划分 |
| 极差 | 0.0151 | 运气好坏的差距 |

**核心教学点**：0.0151 的极差意味着同样的模型和数据，仅因随机种子不同就可能改变"是否达到审稿人要求"的结论。

### Block 3: K-Fold 与 Stratified K-Fold

```python
# 定义四种 CV 策略
cv_methods = {
    '5-Fold': KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),          # 5折
    '10-Fold': KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE),        # 10折
    'Stratified 5-Fold': StratifiedKFold(n_splits=5, shuffle=True,                 # 分层5折
                                          random_state=RANDOM_STATE),
    'Stratified 10-Fold': StratifiedKFold(n_splits=10, shuffle=True,               # 分层10折
                                           random_state=RANDOM_STATE),
}

# 交叉验证评估
for name, cv in cv_methods.items():
    scores = cross_val_score(create_pipeline(), X, y, cv=cv,                       # CV评估
                             scoring='roc_auc', n_jobs=-1)                          # 并行计算
    kfold_results[name] = {'scores': scores, 'mean': np.mean(scores), 'std': np.std(scores)}
```

**实验对比：**

| 方法 | Mean AUC | σ | 极端折差异 |
|------|----------|------|-----------|
| 5-Fold | 0.8914 | 0.0042 | 0.0052 |
| 10-Fold | 0.8914 | 0.0048 | — |
| Stratified 5-Fold | 0.8913 | 0.0027 | 0.0082 |
| Stratified 10-Fold | — | — | — |

**关键发现：**

1. **5-Fold 比 10-Fold 方差更低**（σ=0.0042 vs 0.0048）：5-Fold 每折测试集更大（4,000 vs 2,000 样本），AUC 估计更稳定。并非折数越多越好。
2. **Stratified 5-Fold 方差最低**（σ=0.0027）：分层抽样保持了每折类别比例一致，使评估更稳定。
3. **Stratified 是医学数据的安全做法**：不会损害结果，但在小样本或不平衡场景下是唯一可靠的选择。

### Block 4: Repeated K-Fold

```python
# Repeated 5-Fold (10 次重复 = 50 个模型)
repeated_methods = {
    'Repeated 5-Fold (10x)': RepeatedKFold(
        n_splits=5, n_repeats=10, random_state=RANDOM_STATE),                      # 重复5折
    'Repeated Stratified 5-Fold (10x)': RepeatedStratifiedKFold(
        n_splits=5, n_repeats=10, random_state=RANDOM_STATE),                      # 重复分层5折
}

for name, cv in repeated_methods.items():
    scores_all = []
    for tr_idx, te_idx in cv.split(X, y):                                           # 50次划分
        pipe = create_pipeline()
        pipe.fit(X[tr_idx], y[tr_idx])                                              # 训练
        auc = roc_auc_score(y[te_idx], pipe.predict_proba(X[te_idx])[:, 1])        # 评估
        scores_all.append(auc)
```

**与普通 K-Fold 对比：**

| 方法 | Mean AUC | σ | 模型数 |
|------|----------|------|--------|
| 5-Fold | 0.8914 | 0.0042 | 5 |
| Repeated 5-Fold | 0.8915 | 0.0046 | 50 |
| Repeated Stratified 5-Fold | 0.8914 | 0.0040 | 50 |

**教学结论：**

- Repeated CV 的均值更可靠（基于更多数据点），但标准差可能更大（包含了不同随机洗牌的变异性）。
- 建议同时报告 Mean ± Std，不应只看其中一个指标。

### Block 5: LOOCV（留一法）

```python
# 取 500 样本做 LOOCV
N_LOOCV = 500
X_loocv = X[loocv_idx]                                                              # 子集
y_loocv = y[loocv_idx]

loocv = LeaveOneOut()
loocv_preds = []
for tr_idx, te_idx in loocv.split(X_loocv, y_loocv):                               # n次循环
    pipe = create_pipeline()
    pipe.fit(X_loocv[tr_idx], y_loocv[tr_idx])                                     # n-1个样本训练
    loocv_preds.append(pipe.predict_proba(X_loocv[te_idx])[:, 1][0])              # 1个样本验证

loocv_auc = roc_auc_score(y_loocv, loocv_preds)                                    # 汇总AUC
```

**优缺点分析：**

| 维度 | 说明 |
|------|------|
| 优点 | 几乎无偏（训练集用了 n-1 个样本）；确定性结果（无随机划分）；每个样本都参与评估 |
| 缺点 | 计算成本 O(n)；方差大（每次只验证 1 个样本）；模型之间高度相似，评估不独立 |

**计算成本对比：**

| 样本量 | LOOCV 耗时 | K-Fold 耗时 |
|--------|-----------|------------|
| 500 | 0.6s | < 0.1s |
| 20,000 | ~24s | ~0.6s |
| 1,000,000 | ~20min | ~30s |

**适用场景：** 仅当样本量 < 100 且 K-Fold 每折样本太少时考虑使用。

### Block 6: Nested Cross Validation

```python
# 外层 5-Fold, 内层 3-Fold (共 15 个模型)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)    # 外层CV
param_grid = {'lr__C': [0.01, 0.1, 1, 10]}                                         # 参数网格

# Nested CV: 每个外折独立做内层 GridSearch
nested_scores = []
for tr_idx, te_idx in outer_cv.split(X_nest, y_nest):
    X_tr_n, X_te_n = X_nest[tr_idx], X_nest[te_idx]                               # 外层划分
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    gs_inner = GridSearchCV(create_pipeline(), param_grid, cv=inner_cv,            # 内层选参
                            scoring='roc_auc')
    gs_inner.fit(X_tr_n, y_tr_n)                                                   # 仅在训练折上选参
    best_C = gs_inner.best_params_['lr__C']                                        # 最佳参数

    best_pipe = create_pipeline(C=best_C)
    best_pipe.fit(X_tr_n, y_tr_n)                                                  # 用最佳参数训练
    auc_fold = roc_auc_score(y_te_n, best_pipe.predict_proba(X_te_n)[:, 1])       # 外折评估
    nested_scores.append(auc_fold)
```

**Nested CV 的双层结构：**

```
外层 CV (5-Fold) → 评估泛化能力
│
├── 外折 1:
│   ├── 训练集 (80%) → 内层 CV (3-Fold) → 选最佳参数 C
│   └── 测试集 (20%) ← 用最佳 C 训练模型 → AUC₁
│
├── 外折 2: ... → AUC₂
│ ...
└── 最终: AUC = mean(AUC₁, ..., AUC₅)
```

**实验结果：**

| 方法 | AUC | 解读 |
|------|-----|------|
| 非嵌套 CV | 0.8952 | 调参偏倚（本实验中偏倚很小） |
| Nested CV | 0.8952 | 无偏的泛化能力估计 |
| Fixed C=1 | 0.8941 | 不调参的基线 |
| 外折选择的 C | [0.01, 0.01, 0.01, 0.01, 0.01] | 所有外折选相同参数 |

**教学要点**：本实验中 Nested 与 Non-Nested 差异很小，因为参数网格仅有 4 个值。在复杂调参场景（如 XGBoost 的 3125 种参数组合），非嵌套 CV 的乐观偏倚会显著增大。

### Block 7: 全部方法汇总对比

```python
# 汇总所有方法的结果
summary_rows = []
summary_rows.append({'Method': 'Train-Test Split', 'AUC_Mean': 0.8920, 'AUC_Std': 0.0040})
summary_rows.append({'Method': '5-Fold', 'AUC_Mean': 0.8914, 'AUC_Std': 0.0042})
summary_rows.append({'Method': 'Stratified 5-Fold', 'AUC_Mean': 0.8913, 'AUC_Std': 0.0027})
# ... 其他方法

# 按标准差排序 → 可信度排名
sorted_df = summary_df.dropna(subset=['AUC_Std']).sort_values('AUC_Std')
```

**最终对比表：**

| 方法 | Mean AUC | σ | 模型数 | 推荐场景 |
|------|----------|------|--------|---------|
| Train-Test Split | 0.8920 | 0.0040 | 1 | 快速探索 |
| 5-Fold | 0.8914 | 0.0042 | 5 | 默认选择 |
| 10-Fold | 0.8914 | 0.0048 | 10 | 样本量 < 2,000 |
| **Stratified 5-Fold** | **0.8913** | **0.0027** | **5** | **首选 (σ最小)** |
| Repeated 5-Fold | 0.8915 | 0.0046 | 50 | 更可靠均值 |
| Repeated Stratified 5-Fold | 0.8914 | 0.0040 | 50 | 不平衡+高可靠性 |
| LOOCV | 0.8870 | N/A | n | n < 100 |
| **Nested CV** | **0.8952** | **0.0031** | **15** | **高水平论文标配** |

---

## 关键收获

1. **单次 Train-Test Split 不可靠**：极差 0.0151，模型好坏的声明可能仅取决于随机种子的选择。在医学研究中，单次划分的评估结果不应作为最终结论。
2. **Stratified K-Fold 是医学数据的首选**：通过保持每折类别比例一致，将评估方差从 0.0042 降至 0.0027，且在所有场景中不会损害结果，是小样本或不平衡数据下唯一可靠的选择。
3. **折数并非越多越好**：5-Fold 的方差（0.0042）低于 10-Fold（0.0048），因为每折测试集更大（4,000 vs 2,000 样本），AUC 估计更稳定。K=5 是大多数场景的良好默认值。
4. **Nested CV 是高水平论文的标配**：通过内外双层 CV 将超参数选择与性能评估隔离，避免调参偏倚（optimism bias）。虽然本实验中偏倚很小，但在复杂调参场景下差异会显著增大。
5. **Pipeline 是防止数据泄漏的关键**：交叉验证本身不防止泄漏，必须将预处理和建模封装在 Pipeline 中，确保每折独立执行 fit_transform。

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| CV 各折 AUC 波动大 | 类别比例在各折间不一致 | 使用 StratifiedKFold |
| CV 后 AUC 低于单次划分 | 单次划分可能恰好"运气好" | 以 CV 均值为准，报告 Mean ± Std |
| LOOCV 耗时过长 | 需训练 n 个模型 | 仅在 n < 100 时使用，否则用 K-Fold |
| Nested CV 与非嵌套结果相同 | 参数网格太窄，所有外折选同一参数 | 扩大参数搜索空间或使用更复杂模型 |
| Repeated CV 标准差反而更大 | 包含了不同随机洗牌的变异性 | 同时报告 Mean 和 Std，不要只看一个 |
| 预处理后做 CV 导致泄漏 | 全数据上做了 fit_transform | 将预处理放入 Pipeline，每折独立执行 |

---

## 与其他模块的联系

- **前置模块**：Module 07（数据泄漏）— 交叉验证本身不防止泄漏，预处理必须在每折的训练折上独立完成，与 Pipeline 的使用直接衔接。
- **后续模块**：Module 09（建模对比）— 不同模型的对比需要基于可靠的 CV 评估；Module 10（类别不平衡处理）— Stratified K-Fold 是处理不平衡数据评估的基础；Module 11（校准分析）— CV 得到的预测概率可用于校准曲线绘制。
- **与研究工作的联系**：在 EcMurJ 虚拟筛选中，化合物活性/非活性的二分类评估需要使用 Stratified K-Fold 确保评估可靠性；在宏基因组研究中，样本量有限时 Nested CV 可避免模型选择的调参偏倚。对于需要向审稿人证明模型泛化能力的场景，Repeated Stratified K-Fold + Nested CV 是最稳健的评估策略。

---

## 参考资料

- 教程原文：`ml4health-main/jupyter/08_cross_validation.ipynb`
- 讲义：`ml4health-main/lectures/08_cross_validation_teaching_doc.md`
- Kohavi, R. (1995). *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection.* IJCAI.
- Varma, S., & Simon, R. (2006). *Bias in error estimation when using cross-validation for model selection.* BMC Bioinformatics.
- scikit-learn 官方文档：[Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
