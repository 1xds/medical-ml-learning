# Module 21 笔记: TCGA 基因组学多分类 — 基因表达签名

> 本模块的核心命题：**在 20531 维 TCGA 基因表达数据（p/n≈25 的极度高维场景）上，通过 ANOVA 预筛 → 相关性去冗余 → LASSO 稀疏三步法构建 92 基因的多分类签名，并用 Cohen's κ/MCC 等多分类专用指标验证模型对 5 种癌症的稳健判别能力。**

---

## 核心概念梳理

### 基因组学数据与 p >> n 问题

TCGA (The Cancer Genome Atlas) 是癌症基因组学领域最具影响力的公共数据资源，收录了数万种肿瘤的多组学分子图谱。本模块使用 801 例肿瘤的 20531 个基因表达矩阵，任务为 5 种癌症的多分类：BRCA (乳腺癌, 300 例)、KIRC (肾透明细胞癌, 146)、LUAD (肺腺癌, 141)、PRAD (前列腺癌, 136)、COAD (结肠腺癌, 78)。

与 Module 20 影像组学（93 维 vs 100 样本）相比，本模块将 p >> n 问题推向极端：20531 维特征 vs 801 样本，p/n ≈ 25。在此条件下，普通逻辑回归方程数少于未知数而无解，必须通过正则化求解。基因组数据的另一特性是高度共表达——功能相关的基因在表达水平上高度协同，导致特征间存在大量信息冗余，需要多层特征工程来消除。

### Gene Signature 的概念

Gene Signature（基因签名）是基因组学论文的核心产出形式。其定义为从成千上万基因中筛选出的数十至数百个判别基因集，常用于命名为"X-gene signature"（如本模块的"92-gene cancer-type signature"）。Gene Signature 的价值体现在三个层面：

1. **降维到可解释规模**：从 20531 维降至 92 维，使临床 PCR 验证成为可能（qPCR panel 通常支持 20-100 个靶标）
2. **每个基因有明确系数**：LASSO 的线性特性使每个入选基因对不同癌症类别有方向性贡献，可直接解读生物学意义
3. **跨器官验证**：组织特异标志基因在不同癌症中展现一致的类别推动方向（如前列腺特异基因推高 PRAD 概率）

### 三步特征选择流程

与 Module 20 直接使用 LASSO 不同，本模块因 p 极大 (20531) 而必须采用三步渐进式筛选：

| 步骤 | 方法 | 输入 → 输出 | 目的 |
|------|------|-----------|------|
| 第1步 | ANOVA F-test 单变量预筛 | 20531 → 2000 | 剔除区分度最低的基因，降低 LASSO 搜索空间 |
| 第2步 | Pearson 相关性去冗余 | 2000 → 1817 | 剔除共表达冗余，避免 LASSO 选中冗余基因集 |
| 第3步 | LASSO (L1, multinomial, saga) | 1817 → 92 | 稀疏化选择，生成最终 gene signature |

ANOVA 预筛使用 F-test（多分类变体），以 F-value 排序取 Top-2000。相关性阈值 |r| > 0.9 剔除共表达冗余——基因组学中功能相关的基因常同升同降，若不剔除，LASSO 可能随机选择共表达模块中的任一成员，导致 signature 不稳定。

### 多分类 LASSO 的技术细节

多分类 L1 正则化有严格的技术约束：
- **Solver 必须为 `'saga'`**：`'liblinear'` 不支持 multinomial + L1，`'lbfgs'` 不支持 L1 惩罚
- **签名判定为"任意类非零"**：系数矩阵形状 `(5类, 基因数)`，基因在至少一个类别的系数非零才算入选
- **正则强度 C 的对数搜索**：`Cs=np.logspace(-3, 1, 15)` 在 10⁻³ 至 10¹ 之间对数等距搜索，C 越小正则化越强、signature 越稀疏

---

## 代码精读

### Block 1: 数据加载与特征统计

```python
X_df = pd.read_csv(DATA_CSV, index_col=0)      # 基因表达矩阵 (801 × 20531)
y_df = pd.read_csv(LABELS_CSV, index_col=0)    # 癌症类型标签
le = LabelEncoder()
y = le.fit_transform(y_df.iloc[:, 0].values)   # 5 类 → 0,1,2,3,4
class_names = list(le.classes_)                # ['BRCA','COAD','KIRC','LUAD','PRAD']
n_classes = len(class_names)
X = X_df.values.astype(float)
```

```python
# 防泄漏第 1 步: 先划分 train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
```

**要点**：LabelEncoder 按字母序自动编码，类别顺序为 BRCA(0)、COAD(1)、KIRC(2)、LUAD(3)、PRAD(4)。分层划分保证 5 类比例在 train/test 中一致。

### Block 2: EDA — 基因稀疏性分析

```python
zero_frac = pd.Series((X == 0).mean(axis=0))       # 每个基因的零值比例
print(f"超过 50% 零值的基因: {(zero_frac > 0.5).sum()} 个")  # 基因组学特有现象

# Top-6 高方差基因按癌症类别分组的 boxplot
top_var_genes = pd.Series(X.std(axis=0), index=feat_names).sort_values(
    ascending=False).head(6).index
plot_df = pd.DataFrame(X, columns=feat_names).loc[:, top_var_genes]
plot_df['Class'] = [class_names[i] for i in y]
plot_df_melt = plot_df.melt(id_vars='Class', var_name='gene', value_name='expression')
sns.boxplot(data=plot_df_melt, x='gene', y='expression', hue='Class')
```

**要点**：基因表达的零值比例（稀疏性）是基因组数据的独特属性——大量基因在多数样本中表达量接近零（沉默或低表达），方差预筛可优先剔除这些无区分度的基因。

### Block 3: 多分类统计检验 — ANOVA + η² + BH-FDR

```python
groups = [X[y == c] for c in range(n_classes)]
for j in range(X.shape[1]):
    col_groups = [g[:, j] for g in groups]
    # 跳过方差为 0 的基因
    if all(np.var(g) == 0 for g in col_groups):
        stat_rows.append({'gene': feat_names[j], 'p_value': 1.0, 'eta2': 0.0})
        continue
    f_val, p_val = stats.f_oneway(*col_groups)          # ANOVA F-test
    # 效应量 η² = SS_between / SS_total
    all_vals = X[:, j]
    ss_total = np.sum((all_vals - all_vals.mean()) ** 2)
    ss_between = sum(len(g) * (g[:, j].mean() - all_vals.mean()) ** 2
                     for g in groups)
    eta2 = ss_between / ss_total if ss_total > 0 else 0.0

# BH-FDR 校正
pvals = stat_df['p_value'].values
order = np.argsort(pvals)
ranked = pvals[order]
m = len(pvals)
q = ranked * m / (np.arange(m) + 1)
fdr = np.minimum.accumulate(q[::-1])[::-1]
stat_df['fdr_bh'] = np.clip(fdr_sorted, 0, 1)
```

**要点**：多分类用 ANOVA F-test 替代二分类的 T 检验，效应量改用 η² (eta-squared) 替代 Cohen's d。η² = 组间方差/总方差，衡量基因表达的组间差异占比。基因组学中"95% 基因都显著"是正常现象——大样本 + 强效应意味着单变量筛不够，需要多变量 LASSO 进行稀疏化。

### Block 4: 三步特征选择 — Gene Signature 构建

```python
# 第1步: ANOVA 单变量预筛 (F-test Top-2000)
anova_selector = SelectKBest(f_classif, k=ANOVA_TOP_K).fit(X_train_sc, y_train)
X_train_anova = anova_selector.transform(X_train_sc)

# 第2步: 相关性去冗余 (|r| > 0.9 任选一个保留)
corr_train = pd.DataFrame(X_train_anova, columns=feat_after_anova).corr().abs()
upper_t = corr_train.where(np.triu(np.ones(corr_train.shape, dtype=bool), k=1))
to_drop = set()
for col in upper_t.columns:
    if col in to_drop:
        continue
    high = upper_t.index[upper_t[col] > CORR_THRESHOLD].tolist()
    to_drop.update(high)

# 第3步: LASSO (L1 多项逻辑回归, CV 选 C) — gene signature 核心
lasso = LogisticRegressionCV(
    Cs=np.logspace(-3, 1, 15),           # C ∈ [10⁻³, 10¹]
    penalty='l1', solver='saga',          # saga = multinomial + L1 唯一求解器
    multi_class='multinomial',
    cv=5, scoring='accuracy',
    max_iter=3000, n_jobs=-1, random_state=RANDOM_STATE)
lasso.fit(X_train_corr, y_train)
best_C = lasso.C_[0]
coef_matrix = lasso.coef_                # shape (5类, 基因数)
sig_mask = np.abs(coef_matrix).sum(axis=0) != 0  # 任意类非零 → 入选
signature_genes = np.array(keep_after_corr)[sig_mask]
```

**要点**：`np.abs(coef_matrix).sum(axis=0) != 0` 是精妙的多分类签名判定——只要基因在任一癌症类别有非零系数即保留，因不同癌症由不同基因集推动。系数矩阵每行对应一个癌症类别，可读出每类癌症的标志基因。

### Block 5: 多分类评估指标全家桶

```python
# Cohen's Kappa: 扣除随机一致的一致性
cohen_kappa_score(y_test, best_pred)
# 公式: κ = (p_o - p_e) / (1 - p_e)
# p_o=观测一致率, p_e=期望随机一致率

# MCC (Matthews Correlation Coefficient): 多分类最稳健单值
matthews_corrcoef(y_test, best_pred)
# 基于 TP/FP/TN/FN 全四象限的相关系数, 不受类别分布影响

# Top-K Accuracy: 真实类在预测 Top-K 中即算对
def top_k_accuracy(y_true, prob, k=2):
    topk_idx = np.argsort(-prob, axis=1)[:, :k]
    return float(np.mean([y_true[i] in topk_idx[i] for i in range(len(y_true))]))
```

**要点**：多分类评估的核心教训是"Accuracy 单指标会掩盖少数类表现"——全猜 BRCA 的 Accuracy 也有 37%。κ (Kappa) 扣除了类别先验分布的"假一致"，MCC 是最不会被类别分布欺骗的指标。κ ≈ MCC ≈ Macro-F1 三者高度一致是模型可信度的关键判据。

### Block 6: 多分类 SHAP 可解释性

```python
explainer = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(X_test_sig)
sv_arr = np.asarray(shap_values)          # (n_samples, n_genes, n_classes) 3D!

# 全局重要性: 各类 |SHAP| 均值后对类别求和
global_importance = np.abs(sv_arr).mean(axis=0).sum(axis=1)
# .mean(axis=0) → 每基因每类的平均绝对贡献 (n_genes, n_classes)
# .sum(axis=1)  → 跨所有类汇总 (n_genes,)

# 蜂群图: 跨类求和压回 2D
shap.summary_plot(sv_arr.sum(axis=2), X_test_sig,    # (n_samples, n_genes)
                  feature_names=list(signature_genes), max_display=top_n_display)

# 类别标志基因热力图 (多分类特有)
class_mean_shap = sv_arr.mean(axis=0)    # (n_genes, n_classes)
heat_df = pd.DataFrame(class_mean_shap[top15, :],
                      index=[signature_genes[i] for i in top15],
                      columns=class_names)
sns.heatmap(heat_df, cmap='RdBu_r', center=0, annot=True)
```

**要点**：多分类 SHAP 返回 3D 张量 `(样本, 基因, 类别)`，这是与二分类 (2D) 的根本区别。聚合策略为：先对样本取均值（得每基因每类的平均贡献），再对类别求和（得跨所有类的总重要性）。热力图直观展示哪些基因是哪类癌症的特异标志——正值 (红) 推高该类概率、负值 (蓝) 推低。

---

## 关键收获

1. **极端 p >> n 下的三步特征选择是必由之路**：直接对 20531 基因上 LASSO 不仅计算缓慢且求解不稳定。ANOVA 预筛剔除低区分度基因、相关性去冗余消除共表达模块的随机性、LASSO 稀疏产生可解释签名——三步法在基因组学中已成为标准实践。

2. **多分类 LASSO 的系数矩阵揭示了组织特异性**：系数矩阵的每行对应一个癌症类别，每列为一个基因。基因在各癌症上的系数符号反映了"组织特异标志基因"的生物学本质——前列腺特异基因正系数推高 PRAD、负系数压低其他类别。

3. **κ/MCC 是多分类评估的必备指标**：Accuracy 在类别不平衡时由多数类主导，κ 扣除随机一致的贡献，MCC 考虑全混淆矩阵四象限。三个指标同时高才说明模型真正稳健。在极难任务（如亚型分类，各类差异细微）中，κ/MCC 将成为模型选型的决定性依据。

4. **SHAP 3D 张量需要跨类别聚合**：多分类 SHAP 值的 3D 形态 `(样本, 基因, 类别)` 意味着可视化前必须降维。跨类求和 (`sum(axis=2)`) 适用于全局重要性，按类分面适用于标志基因发现。

5. **与影像组学的对照揭示了模块化迁移能力**：基因组学 (20531 维) 与影像组学 (93 维) 共享同样的流程骨架，差异仅在于 p >> n 程度不同导致特征选择层数不同——验证了 ML 工程能力可跨模态复用。

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| LASSO 选 0 个基因 (C 过大) | C 搜索范围过小、惩罚过强 | 上移 C 上界至 `np.logspace(-2, 2, 20)` |
| LASSO 选全 1817 基因 (稀疏失败) | C 过大、惩罚过弱 | 下移 C 范围至 `np.logspace(-4, 0, 20)` |
| `liblinear` solver 报错 | liblinear 不支持 multinomial + L1 | 必须用 `solver='saga'` |
| 95% 基因 ANOVA 显著 | 大样本 + 强效应，基因组学正常现象 | 说明需要多变量 LASSO，不能仅靠单变量 |
| SHAP 蜂群图报 3D 错误 | shap 期望 2D 输入 | 先做 `sv_arr.sum(axis=2)` 压回 2D |
| SHAP TreeExplainer 不支持 LogReg | LogReg 不是树模型 | 改用 XGBoost/LightGBM 做解释模型 |

---

## 与其他模块的联系

- **前置模块**：Module 01-12 (表格数据 ML 流程) — 复用完整 10 步骨架；Module 20 (影像组学 ML 流水线) — 与影像组学并列构成"新模态迁移"的两条主线，p >> n 程度决定特征选择层数
- **后续模块**：Module 22 (CGM 血糖回归) — 从分类切换到回归任务，损失函数和评估指标相应改变
- **与研究工作的联系**：Gene Signature 的构建方法可迁移至 EcMurJ 研究中——若通过 RNA-seq 获得不同条件下的大肠杆菌转录组数据，可构建"噬菌体感染响应 signature"或"耐药表型 signature"。多分类评估中 κ/MCC 的思想可应用于宏基因组物种分类任务中的模型可靠性验证。

---

## 参考资料

- 教程原文：`ml4health-main/jupyter/21_gene_ml_pipeline.ipynb`
- 讲义：`ml4health-main/lectures/21_gene_case_study_teaching_doc.md`
- Golub, T. R., et al. (1999). "Molecular Classification of Cancer: Class Discovery and Class Prediction by Gene Expression Monitoring." *Science*, 286(5439), 531-537.
- Tibshirani, R. (1996). "Regression Shrinkage and Selection via the Lasso." *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288.
- The Cancer Genome Atlas Research Network. "Comprehensive molecular characterization of human colon and rectal cancer." *Nature*, 487, 330-337 (2012).
- Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*, 4765-4774.
- Cohen, J. (1960). "A Coefficient of Agreement for Nominal Scales." *Educational and Psychological Measurement*, 20(1), 37-46.
