# Module 07 笔记: 数据泄漏分析

> 本模块的核心命题：**通过对比实验直观展示数据泄漏如何导致模型性能虚高，建立"测试集隔离"的实验规范，并理解泄漏的"剂量效应"。**

---

## 核心概念梳理

### 数据泄漏是什么？

数据泄漏（Data Leakage）是指任何在模型训练过程中使用了测试集信息的操作。这不是"技术错误"，而是"逻辑错误"——将本应不可见的数据用于决策。其本质危害在于：**夸大了模型的泛化能力**，即模型在未见过数据上的真实表现远差于实验报告值。

在医学 AI 中，数据泄漏是后果最严重的错误之一。如果模型声称 AUC = 0.95 但其中 0.10 来自泄漏，那么临床部署后性能断崖式下跌——这直接关系到患者的生命安全。审稿人和临床医生对数据泄漏的警惕性远高于其他方法学问题，因为它意味着"实验结果从根本上不可信"。

### 泄漏的"剂量效应"

数据泄漏不是二元的（泄漏/不泄漏），而是存在剂量效应——泄漏的信息量越大，性能虚高越严重：

| 泄漏的信息 | 泄漏量 | 典型案例 | 危害程度 |
|-----------|--------|---------|---------|
| 均值/标准差 | 2 个参数/特征 | 全数据标准化 | 低 |
| 缺失值填充值 | 1 个值/特征 | 全数据均值插补 | 低 |
| 特征排名 | K 个特征 | 全数据特征选择 | **中-高** |
| 目标编码值 | 1 个值/类别 | 全数据 Target Encoding | **高** |
| 类别平衡信息 | K 类 | 全数据 SMOTE | **高** |
| 同一患者的重复记录 | 多行 | 训练集和测试集有相同患者 | **极高** |

### 三条核心实验

| 实验 | 泄漏类型 | 泄漏内容 | 本数据效应 |
|------|---------|---------|-----------|
| 实验 1 & 2 | 标准化泄漏 | 测试集的 μ 和 σ 泄露到训练集 | ΔAUC ≈ 0.001（极小） |
| 实验 3 | 特征选择泄漏 | 测试集的特征-目标关系影响特征选择 | ΔAUC ≈ 0.000（本数据较小） |
| 扩展实验 | 综合 + CV 泄漏 | 多种泄漏叠加 + 伪交叉验证选参 | ΔAUC ≈ 0.001 |

---

## 代码精读

### Block 1: 环境配置与数据准备

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline                  # 防泄漏的标准工具
import time

df = pd.read_csv(DATA_PATH, low_memory=False)
df['target'] = df['Status.Vital'].map({'VIVO': 1, 'MORTO': 0})

# 8 个特征：含数值型和分类型
feature_cols = ['Age', 'year', 'Gender', 'Code.Profession',
                'Diagnostic.means', 'Extension', 'Raca.Color', 'State.Civil']
```

要点：
1. Pipeline 是本模块的核心防泄漏工具——自动确保每折独立做预处理
2. 故意设计 8 个特征的"低维安全"场景，为后续"高维危险"讨论埋下伏笔

### Block 2: 实验 1——标准化泄漏（错误流程）

```python
# ❌ 错误：在全数据上标准化，再划分
X_leak = X_raw.copy()
imputer_full = SimpleImputer(strategy='median')
scaler_full_leak = StandardScaler()
X_leak_scaled = scaler_full_leak.fit_transform(imputer_full.fit_transform(X_leak))
#                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                      fit_transform 使用了测试集的 μ 和 σ！

X_tr_leak, X_te_leak, y_tr_leak, y_te_leak = train_test_split(
    X_leak_scaled, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)

lr_leak = LogisticRegression(class_weight='balanced', max_iter=5000)
lr_leak.fit(X_tr_leak, y_tr_leak)
y_prob_leak = lr_leak.predict_proba(X_te_leak)[:, 1]
```

泄漏内容：测试集每个特征的均值（μ）和标准差（σ）被泄露到训练集的标准化参数中。

### Block 3: 实验 1——标准化泄漏（正确流程）

```python
# ✅ 正确：先划分，再在训练集上 fit
X_tr, X_te, y_tr, y_te = train_test_split(
    X_raw, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)

imputer_tr = SimpleImputer(strategy='median')
scaler_tr = StandardScaler()
X_tr_scaled = scaler_tr.fit_transform(imputer_tr.fit_transform(X_tr))
#         只在训练集上 fit_transform
X_te_scaled = scaler_tr.transform(imputer_tr.transform(X_te))
#         测试集仅 transform
```

**差异分析**：

```python
diff = auc_leak - auc_correct
print(f"AUC 差异 (泄漏 - 正确) = {diff:.6f}")
# 典型输出：ΔAUC ≈ 0.001，几乎无差异
```

要点：
1. 标准化泄漏效应极小（ΔAUC < 0.001），因为标准化仅泄漏"每特征 2 个参数"
2. **教学重点**：即使效应小，错误的习惯会在更危险的场景中带来严重后果

### Block 4: 实验 3——特征选择泄漏（错误流程）

```python
# ❌ 错误：在全数据上做特征选择，再划分
X_fs_full = imputer_full.fit_transform(X_raw)
X_fs_full = scaler_full_leak.fit_transform(X_fs_full)

selector_leak = SelectKBest(f_classif, k=4)
X_fs_selected = selector_leak.fit_transform(X_fs_full, y)
#          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#          fit_transform 使用测试集参与了"哪些特征被选中"的判断！

X_tr_fs, X_te_fs, y_tr_fs, y_te_fs = train_test_split(
    X_fs_selected, y, test_size=0.3, random_state=RANDOM_STATE)
```

泄漏内容：测试集中每个特征与目标变量的相关性信息被用于特征选择决策。

### Block 5: 实验 3——特征选择泄漏（正确流程）

```python
# ✅ 正确：先划分，在训练集上做特征选择
X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X_raw, y, ...)

# 插补 + 标准化只在训练集
X_tr1_pp = scaler_tr.fit_transform(imputer_tr.fit_transform(X_tr1))
X_te1_pp = scaler_tr.transform(imputer_tr.transform(X_te1))

# 特征选择只在训练集
selector_correct = SelectKBest(f_classif, k=4)
X_tr1_sel = selector_correct.fit_transform(X_tr1_pp, y_tr1)
X_te1_sel = selector_correct.transform(X_te1_pp)
```

### Block 6: 不同 K 值下的泄漏效应

```python
k_values = [2, 3, 4, 5, 6, 7]
leak_aucs = []
correct_aucs = []

for k in k_values:
    # 泄漏版：全数据 FS → 划分 → 模型
    sl = SelectKBest(f_classif, k=k)
    X_sl = sl.fit_transform(X_fs_full, y)     # ❌ 在全数据上 fit
    ...

    # 正确版：划分 → 训练集 FS → 模型
    sc_c = SelectKBest(f_classif, k=k)
    X_tr_sc = sc_c.fit_transform(X_tr1_pp, y_tr1)  # ✅ 只在训练集 fit
    ...
```

**结果**：在所有 K 值下，泄漏版和正确版的 AUC 几乎完全相同（ΔAUC < 0.001），因为 8 个特征均有预测力，选出的特征集相同。

**教学要点**：这是"低维安全"的特例。当特征数增长到 100+ 且大部分是噪声时，泄漏版的 AUC 虚高会显著增加。

### Block 7: 扩展——综合泄漏对比

```python
# 1) 完全正确版（Pipeline 方法）
pipe_correct = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('select', SelectKBest(f_classif, k=6)),
    ('lr', LogisticRegression(class_weight='balanced', max_iter=5000))
])
pipe_correct.fit(X_tr, y_tr)
auc_pc = roc_auc_score(y_te, pipe_correct.predict_proba(X_te)[:, 1])

# 2) 标准化泄漏版
X_both = scaler_full.fit_transform(imputer_full.fit_transform(X_raw))  # 全数据
X_tr_b, X_te_b, ... = train_test_split(X_both, y, ...)                  # 再划分

# 5) CV 泄漏版
gs = GridSearchCV(lr, param_grid, cv=5)
gs.fit(X_cv_full, y)                  # ❌ 全数据 CV → 最优参数含测试集信息
best_C = gs.best_params_['C']
lr_best = LogisticRegression(C=best_C, ...)
X_tr_cv, X_te_cv, ... = train_test_split(X_cv_full, y, ...)
lr_best.fit(X_tr_cv, y_tr_cv)         # 用泄漏来的最优参数训练
```

**综合对比结果**：

| 场景 | AUC | Δ（vs Correct） | 泄漏类型 |
|------|-----|----------------|---------|
| Correct Pipeline | 基线 | — | 无 |
| Standardization Leak | 基线 + 0.001 | 标准化 |
| Feature Select Leak | 基线 + 0.000 | 特征选择 |
| CV Leak | 基线 + 0.001 | 交叉验证 |

### Block 8: 综合对比可视化

```python
ax.axhline(y=baseline, color='#2ecc71', linestyle='--',
           label=f'Correct Baseline (AUC={baseline:.4f})')

for bar, val in zip(bars, vals):
    if val != baseline:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'Δ=+{val-baseline:.4f}', color='white', fontweight='bold')
```

---

## 关键收获

1. **任何使用测试集信息的操作 = 数据泄漏**，无论泄漏量大小。数据划分后必须将测试集视为"未来数据"，不可触碰。

2. **标准化泄漏效应小，但不能成为容忍的理由**：StandardScaler 每个特征仅泄漏 2 个参数（μ、σ），在本数据集上 ΔAUC < 0.001。但"先全数据标准化再划分"的坏习惯在特征选择、目标编码等高信息量操作中会产生ΔAUC = 0.05-0.20 的严重后果。

3. **特征选择泄漏是医学 AI 论文中最常见的泄漏来源**：在高维小样本场景下（500 个特征中 480 个是噪声），全数据特征选择会让噪声特征"碰巧"在测试集上表现出与目标的相关性，AUC 虚高可达 0.20。

4. **CV 泄漏隐蔽性最强**：看似"做了交叉验证"，但如果参数搜索涉及全数据（而非训练集内部的 CV），选出的最优参数已经"偷看"了测试集。正确做法是嵌套交叉验证——外层评估泛化能力，内层选参数。

5. **Pipeline 是天然的防泄漏工具**：sklearn 的 Pipeline 确保每折交叉验证独立完成预处理——插补的均值、标准化的参数、特征选择的结果都只在训练折上拟合，测试折仅 `transform`。

6. **泄漏的危害有"剂量效应"**：泄漏的信息量越大，AUC 虚高越严重。标准化仅泄漏 2 个参数（危害小），特征选择泄漏 K 个特征（中-高危害），患者去重失败导致同一患者的记录分到训练/测试集（极高危害）。

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 标准化泄漏后 AUC 几乎不变 | 标准化仅泄漏 2 个参数，效应极小 | 即使效应小，也需要养成"先划分再标准化"的习惯 |
| 特征选择泄漏在本数据集不明显 | 8 个特征均有预测力，同一组特征在任何划分下都被选中 | 在高维场景下（100+ 特征）重复实验以验证泄漏效应 |
| CV 内选择了最优参数后直接评估 | 参数选择使用了测试集信息 | 使用嵌套交叉验证，或 Pipeline + GridSearchCV | 确保只在训练集上搜索参数 |
| SMOTE 产生了泄漏 | 在划分前做 SMOTE，生成的人工样本包含了测试集信息 | 先划分，再对训练集做 SMOTE |
| 同一患者的多条记录被分到不同折 | Patient.Code 相同但在不同折中 | 按 Patient.Code 分组划分（GroupKFold） |

---

## 与其他模块的联系

- **前置模块**：Module 03（缺失值插补）中的"插补器在训练集上 fit、测试集上 transform"已体现了防泄漏规范；Module 04（特征工程）和 Module 05（特征选择）的防泄漏实践（标准化/特征选择的 fit/transform 分离）是该规范的自然延续
- **与研究工作的联系**：在 EcMurJ 虚拟筛选中，当使用交叉验证评估模型的虚拟筛选能力时，必须确保每个化合物的特征计算、缺失值处理、标准化都在各折内独立完成。特别需要注意的是：同一配体结合不同蛋白构象的数据可能产生"结构泄漏"（口袋相似导致的信息泄露）。在宏基因组研究中，同一受试者的多次采样数据不能分散到训练集和测试集（GroupKFold）。此外，微生物丰度数据的 CLR 变换必须基于训练集的几何均值——这是标准化泄漏在组学数据中的典型体现。

---

## 参考资料

- 教程原文：`ml4health-main/jupyter/07_data_leakage.ipynb`
- 讲义：`ml4health-main/lectures/07_data_leakage_teaching_doc.md`
- Kaufman, S., Rosset, S., & Perlich, C. (2012). Leakage in Data Mining: Formulation, Detection, and Avoidance. *ACM TKDD*, 6(4), 15.
- Kapoor, S. & Narayanan, A. (2023). Leakage and the Reproducibility Crisis in ML-based Science. *Patterns*, 4(9).
- Varma, S. & Simon, R. (2006). Bias in Error Estimation When Using Cross-Validation for Model Selection. *BMC Bioinformatics*, 7, 91.
