# Module 12 笔记: SHAP + LIME 可解释性分析

> 本模块的核心命题：**打开机器学习的"黑箱"——SHAP 基于博弈论提供严谨的特征贡献分配，LIME 通过局部线性近似实现灵活解释；高级 SHAP 可视化进一步揭示特征的非线性关系与交互结构。**

---

## 核心概念梳理

### 模型可解释性是什么？

模型可解释性（Model Interpretability）是指理解和解释模型决策过程的能力。在医学 AI 领域，可解释性不仅是技术需求，更是伦理和监管要求——临床医生需要知道模型为何给出某个预测，才能信任并将其纳入决策流程。

可解释性分为两个层次：

| 层次 | 问题 | 工具 | 听众 |
|------|------|------|------|
| **全局解释** | "整体上哪些特征最重要？" | Feature Importance, Permutation, SHAP Summary | 研究者、模型开发者 |
| **局部解释** | "为什么这个患者被预测为高风险？" | SHAP Waterfall, LIME | 医生、患者、监管者 |

### SHAP 的理论基础

SHAP（SHapley Additive exPlanations）基于博弈论中的 Shapley Value，将每个特征视为"玩家"，预测结果视为"收益"，通过计算每个特征在所有可能特征组合中的边际贡献来分配其对预测的影响。

| 公理 | 含义 | 重要性 |
|------|------|--------|
| 效率 | 所有特征贡献之和 = 总预测 - 基准值 | 保证解释完整 |
| 对称性 | 贡献相同的特征分配相同 | 公平性 |
| 虚拟性 | 不贡献的特征分配 0 | 不引入噪声 |
| 可加性 | 合并两个模型的 SHAP = 分别计算之和 | 一致性保证 |

### 三种全局重要性方法对比

| 方法 | 原理 | 优点 | 缺点 | year 排名 |
|------|------|------|------|-----------|
| Gini Importance | 树分裂时 Gini 减少量总和 | 计算快、内置 | 偏向高基数特征 | 0.7009 |
| Permutation Importance | 打乱特征后 AUC 下降量 | 与模型无关 | 计算慢、需要基准 | 0.0527 |
| SHAP mean | SHAP 值绝对值的平均 | 理论最严谨、可视化丰富 | 计算成本 | 0.274 |

---

## 代码精读

### Block 1: 模型训练与 SHAP 计算

```python
# 训练 Random Forest 模型
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=8, class_weight='balanced',          # 200棵树,深度8
    random_state=RANDOM_STATE, n_jobs=-1)
rf_model.fit(X_tr_final, y_tr)
y_prob = rf_model.predict_proba(X_te_final)[:, 1]                    # 预测概率
auc = roc_auc_score(y_te, y_prob)                                    # AUC评估

# SHAP 计算 (TreeExplainer)
explainer = shap.TreeExplainer(rf_model)                             # 树模型专用解释器
X_shap = X_te_final[:500]                                           # 取500样本加速
shap_values = explainer.shap_values(X_shap)                          # 计算SHAP值

# 二分类: 取正类(VIVO)的 SHAP 值
if isinstance(shap_values, list):
    sv = shap_values[1]                                              # 正类SHAP值
else:
    sv = shap_values
    if sv.ndim == 3:
        sv = sv[:, :, 1]                                             # 取正类维度

# base value = 数据集中平均预测概率
# 预测概率 = base value + Σ(各特征的 SHAP 值)
```

**SHAP 的核心公式：**

```
base value = 平均预测概率 ≈ 0.049 (VIVO 比例)
预测概率 = base value + Σ(SHAP 值)
SHAP > 0 → 推高预测概率 (→ VIVO)
SHAP < 0 → 推低预测概率 (→ MORTO)
```

### Block 2: 全局解释 — 三种重要性方法

```python
# A1: Gini Importance (内置)
importances = rf_model.feature_importances_                          # Gini重要性

# A2: Permutation Importance
perm_result = permutation_importance(
    rf_model, X_te_final, y_te, n_repeats=10, random_state=RANDOM_STATE)  # 打乱10次

# A3: SHAP 全局重要性
shap_importance = np.abs(sv).mean(0)                                # mean |SHAP|

# 蜂群图 (Bee Swarm)
shap.summary_plot(sv, X_shap, feature_names=feature_names_short)    # 蜂群图

# 依赖图 (Dependence Plot)
for rank, idx in enumerate(top2_idx):
    correlations = []
    for j in range(X_shap.shape[1]):
        if j != idx:
            corr, _ = pearsonr(X_shap[:, j], sv[:, idx])            # 找交互特征
            correlations.append((j, abs(corr)))
    interaction_idx = max(correlations, key=lambda x: x[1])[0]      # 最强交互
    plt.scatter(X_shap[:, idx], sv[:, idx], c=X_shap[:, interaction_idx])  # 散点+颜色

# 交互热图
for i in range(n_features):
    for j in range(n_features):
        if i != j:
            cov = np.cov(sv[:, i], sv[:, j])[0, 1]                  # SHAP协方差
            shap_interaction[i, j] = abs(cov)
```

**三种方法排名对比：**

| 特征 | Gini | Permutation (ΔAUC) | SHAP mean |value| |
|------|------|--------------------|--------------------|
| year | 0.7009 | 0.0527 | 0.274 |
| Code.Profession | 0.0655 | 0.0142 | 0.036 |
| Diagnostic.means | 0.1071 | 0.0137 | 0.021 |
| Age | 0.0725 | -0.0034 | — |
| Extension | 0.0311 | 0.0026 | — |
| Raca.Color | 0.0228 | 0.0066 | — |

**关键教学发现：** Age 的 Permutation Importance 为负值（-0.0034），意味着打乱 Age 后 AUC 反而上升。这并非 Age 是噪声，而是 Age 与 year 之间存在部分冗余，打乱 Age 后模型可依赖 year 补偿，AUC 因减少冗余而略微上升。

### Block 3: 局部解释 — SHAP Waterfall

```python
# 选取代表性样本: 预测概率最高(VIVO)和最低(MORTO)
idx_vivo = np.argmax(probs_vivo)                                    # 最确信VIVO
idx_morto = np.argmin(probs_vivo)                                   # 最确信MORTO

# SHAP Waterfall Plot
shap.waterfall_plot(
    shap.Explanation(values=sv[sample_idx],                         # 该样本SHAP值
                      base_values=base_val,                          # 基准概率
                      data=X_shap[sample_idx],                       # 特征值
                      feature_names=feature_names_short))
```

**VIVO 高概率样本的 Waterfall 解读：**

```
base value: 0.049 (平均 VIVO 概率)
                    ↓
year = +0.3271      ─────────→ 大幅推高
Code.Profession = +0.0638 ──→ 推高
Diagnostic.means = +0.0350 ──→ 推高
Raca.Color = +0.0138      ──→ 推高
Extension = +0.0051       ──→ 轻微推高
Age = -0.0135             ──→ 轻微推低
                    ↓
最终预测概率: ≈ 0.94
```

**教学要点：** Waterfall Plot 从基准概率出发，各特征"增减"的叙事方式是向临床医生解释模型最直觉的工具。

### Block 4: 局部解释 — LIME

```python
# LIME 解释器
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_tr_final, feature_names=feature_names_short,                   # 训练数据
    class_names=['MORTO (Death)', 'VIVO (Alive)'],
    mode='classification', random_state=RANDOM_STATE)

# 解释单个样本
exp = lime_explainer.explain_instance(
    X_shap[sample_idx], rf_model.predict_proba,                     # 模型预测函数
    num_features=len(feature_names_short), labels=[1])              # 解释正类

# LIME 输出: 特征条件 + 贡献值
lime_text = exp.as_list()
# 输出示例:
#   year > 0.82           → +0.2806
#   Diagnostic.means ≤ -0.11 → +0.2048
#   Raca.Color ≤ -0.45    → +0.0995
```

**LIME 的三步工作原理：**

1. 在目标样本附近随机生成扰动样本
2. 用黑箱模型对扰动样本预测
3. 在扰动样本上训练一个简单模型（线性回归），系数即局部特征贡献

### Block 5: SHAP vs LIME 深度对比

```python
# 同一患者的 SHAP vs LIME 对比
# SHAP Top 5
sv_sample = sv[sample_idx_cmp]
sv_sorted_idx = np.argsort(abs(sv_sample))[::-1][:5]               # SHAP排序

# LIME Top 5
exp_cmp = lime_explainer.explain_instance(X_shap[sample_idx_cmp], ...)
lime_top5 = exp_cmp.as_list(label=1)[:5]                           # LIME排序

# 并排对比柱状图
ax.bar(x_pos_shap, shap_contrib, color=..., label='SHAP')          # SHAP贡献
ax.bar(x_pos_lime, lime_vals, color=..., label='LIME', hatch='//') # LIME贡献
```

**对比结果：**

| 维度 | SHAP | LIME |
|------|------|------|
| 最重要的特征 | year (+0.3271) | year (> 0.82, +0.2806) |
| 排名 #2 | Code.Profession (+0.0638) | Diagnostic.means (≤ -0.11, +0.2048) |
| 排名 #3 | Diagnostic.means (+0.0350) | Raca.Color (≤ -0.45, +0.0995) |
| Age 方向 | 负 (-0.0135) | 负 (-0.0085) |
| 理论基础 | Shapley Value (博弈论) | 局部线性近似 |
| 稳定性 | 确定性 | 有随机性 (扰动采样) |
| 输出类型 | 原始 SHAP 值 | 离散化阈值 |

**一致性：** 两个方法都认为 year 是最重要特征，Age 负贡献——在最重要的判断上一致。
**差异：** Code.Profession 和 Diagnostic.means 的排名互换，这是方法差异导致的正常现象。

### Block 6: LIME 稳定性实验

```python
# 找两个特征值最相近的样本
from sklearn.metrics.pairwise import euclidean_distances
dist_matrix = euclidean_distances(X_shap_subset)                    # 距离矩阵
np.fill_diagonal(dist_matrix, np.inf)                               # 排除自身
nearest_idx = np.argmin(dist_matrix[idx_vivo])                      # 最近邻

# 对两个样本分别做 LIME
exp_a = lime_explainer.explain_instance(X_shap[idx_vivo], ...)      # 样本A
exp_b = lime_explainer.explain_instance(X_shap[nearest_idx], ...)   # 样本B(最近邻)

# 对比 Top 5 特征
```

**实验结果：**

| 样本 | 预测概率 | 真实标签 | Top 5 特征一致性 |
|------|---------|---------|----------------|
| A (idx=229) | 0.9400 | VIVO | year > 0.82, D.means ≤ -0.11, Raca ≤ -0.45, Ext ≤ -0.21, Age |
| B (最近邻) | 0.8988 | VIVO | Top 4 完全相同，仅 Age 区间不同 |

**教学发现：** 在数据密度充足的场景下 LIME 稳定性良好，但在小样本或稀疏区域，扰动样本可能进入不现实的区域，导致线性近似失效。

### Block 7: 高级 SHAP — 二次拟合依赖图

```python
# 对每个特征同时做线性和二次拟合，比较 R²
for rank, idx in enumerate(feature_order[:6]):
    x_vals = X_shap[:, idx]                                         # 特征值
    y_vals = sv[:, idx]                                             # SHAP值

    # 二次多项式拟合
    z2 = np.polyfit(x_vals, y_vals, 2)                             # 二次拟合
    p2 = np.poly1d(z2)
    r2_quad = 1 - ss_res / ss_tot                                  # R²(二次)

    # 线性拟合
    z1 = np.polyfit(x_vals, y_vals, 1)                             # 线性拟合
    p1 = np.poly1d(z1)
    r2_lin = 1 - ss_res_lin / ss_tot                               # R²(线性)

    delta_r2 = r2_quad - r2_lin                                    # ΔR²
```

**R² 对比结果：**

| 特征 | R² 线性 | R² 二次 | ΔR² | 判断 |
|------|---------|---------|-----|------|
| Diagnostic.means | 0.0457 | 0.7638 | +0.7181 | 高度非线性 |
| year | 0.6276 | 0.8595 | +0.2319 | 非线性 |
| Age | 0.1648 | 0.3586 | +0.1938 | 非线性 |
| Code.Profession | 0.4523 | 0.5984 | +0.1461 | 一定非线性 |
| Extension | 0.5698 | 0.6070 | +0.0372 | 接近线性 |
| Raca.Color | 0.7477 | 0.7485 | +0.0008 | 本质线性 |

**核心教学点：** Diagnostic.means 的 ΔR²=+0.7181，线性回归几乎完全漏掉其与 SHAP 值的关系。ΔR² 的意义在于指导特征工程——对于高度非线性的特征，添加平方项或分箱可能大幅提升线性模型性能。

### Block 8: 高级 SHAP — 交互网络图与趋势图

```python
# 交互网络图: 节点=特征, 边=|SHAP值相关系数|
for i in range(top_n_net):
    for j in range(top_n_net):
        if i != j:
            corr, _ = pearsonr(sv[:, idx1], sv[:, idx2])            # SHAP值相关
            interaction_matrix[i, j] = abs(corr)

# 只显示 |r| > 0.1 的边
if strength > 0.1:
    ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]],            # 绘制边
            linewidth=strength * 8, alpha=min(strength*1.5, 0.9))

# 趋势图: 按预测概率排序，观察 SHAP 值变化
sorted_order = np.argsort(y_prob_shap)                              # 按概率排序
for feat_idx in top_trend_idx:
    ax.plot(sv[sorted_order, feat_idx],                             # SHAP趋势
            label=f'{feature_names[feat_idx]}')
```

**交互强度排名：**

| 交互对 | |r| | 解读 |
|--------|-----|------|
| Diagnostic.means × year | 0.156 | 最强交互 |
| Diagnostic.means × Extension | 0.155 | 诊断方式与肿瘤扩展相关 |
| Raca.Color × Extension | 0.133 | 种族与肿瘤扩展的关联 |

**趋势图发现：** Top 3 特征的 SHAP 趋势与预测概率参考线高度同步，说明这三个特征共同解释了大部分预测方差。

---

## SHAP vs LIME 核心差异

| 维度 | SHAP | LIME |
|------|------|------|
| 理论基础 | Shapley Value (博弈论) | 局部线性近似 |
| 计算速度 | TreeExplainer 快 | 需大量扰动采样 |
| 输出类型 | 原始 SHAP 值（连续） | 离散化阈值条件 |
| 稳定性 | 确定性结果 | 有随机性 |
| 医学论文使用率 | 95% 以上 | 较少 |
| 适用模型 | TreeExplainer(树模型) / KernelExplainer(任意) | 任意模型 |

### 选择指南

| 场景 | 推荐方法 |
|------|---------|
| 树模型 + 论文发表 | SHAP TreeExplainer |
| 任意模型 + 快速原型 | LIME |
| 需要确定性解释 | SHAP |
| 临床决策支持系统 | SHAP + LIME 互补 |

---

## 高级 SHAP 五维特征评估框架

| 特征 | 重要性 | 非线性程度 (ΔR²) | 交互强度 | 趋势一致性 | 分位数跨度 |
|------|--------|-----------------|---------|-----------|-----------|
| year | 极高 | 强 (0.23) | 中 (0.16) | 高 | 0.662 |
| Diagnostic.means | 高 | 极强 (0.72) | 强 (0.16) | 中 | 0.060 |
| Code.Profession | 中 | 中 (0.15) | 中 (0.11) | 高 | 0.125 |
| Raca.Color | 中 | 无 (0.001) | 中 (0.13) | 高 | 0.053 |
| Extension | 低 | 弱 (0.04) | 中 (0.16) | 低 | 0.024 |
| Age | 低 | 中 (0.19) | 弱 (0.06) | 低 | — |

---

## 关键收获

1. **三种全局重要性方法高度一致**：year 是最重要特征，三种方法排名相似。但 Permutation Importance 发现 Age 的负值揭示了与 year 的冗余关系，这是其他方法无法检测的。
2. **SHAP 的理论基础最严谨**：基于博弈论 Shapley Value，满足效率、对称性、虚拟性、可加性四条公理，是唯一有完备数学保证的特征贡献分配方法。
3. **SHAP Waterfall 是向医生解释模型的最佳工具**：从基准概率出发，各特征"增减"的叙事方式最为自然直觉，可直接回答"这个患者为什么被预测为高风险"。
4. **SHAP 与 LIME 在最重要的判断上一致，但细节有差异**：SHAP 给出连续的 SHAP 值，LIME 给出离散化的阈值条件。两种方法互补使用可提供更全面的解释。
5. **线性拟合 ≠ 全部真相**：Diagnostic.means 的 ΔR²=+0.7181，线性方法几乎完全漏掉其与 SHAP 值的 U 型关系。量化非线性程度对特征工程具有重要指导意义。
6. **模型可解释性不是选做题**：在医学 AI 论文中，SHAP 已成为审稿人的标准要求。基础 SHAP（蜂群图、条形图）是必选项，高级 SHAP（二次拟合、交互网络）可作为高分量论文的加分项。

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| SHAP KernelExplainer 极慢 | 需要计算所有特征组合 | 使用 TreeExplainer（仅树模型） |
| LIME 两次运行结果不同 | 扰动采样的随机性 | 固定 random_state 或增大 num_samples |
| Gini Importance 偏向连续特征 | 高基数特征有更多分裂点 | 参考 Permutation Importance 交叉验证 |
| Permutation Importance 出现负值 | 特征间存在冗余 | 检查特征相关性，考虑去除冗余特征 |
| SHAP 值相关性 ≠ 真实交互 | 仅捕捉线性相关 | 使用 SHAP Interaction Values（计算成本更高） |
| LIME 在稀疏区域不稳定 | 扰动样本进入不现实区域 | 增大采样数或与 SHAP 结果交叉验证 |

---

## 与其他模块的联系

- **前置模块**：Module 09（建模对比）— 本模块解释的 RF 模型是 Module 09 中 AUC 排名第三的模型；Module 11（校准分析）— 校准分析关注概率值的可信度，SHAP 关注概率值的来源，两者互补构建模型可信度。
- **后续模块**：本模块是 ml4health 系列的最后一个模块，将前面所有模块的建模决策（特征选择、交叉验证、重采样、校准）通过可解释性串联起来。
- **与研究工作的联系**：在 EcMurJ 虚拟筛选中，SHAP 可解释性分析可用于揭示哪些分子描述符（如分子量、LogP、氢键供体数）对活性预测贡献最大，指导药物设计；SHAP 依赖图可揭示描述符与活性的非线性关系（如 U 型关系提示最优范围），指导先导化合物优化。在宏基因组研究中，SHAP 可识别与长寿表型最相关的微生物特征，为机制研究提供假说。LIME 可用于解释单个样本（如某位百岁老人）的预测依据，增强结果的可解释性和临床转化价值。需要注意的是，SHAP 值反映的是模型内部贡献模式，不等同于因果关系，需与生物学实验验证结合。

---

## 参考资料

- 教程原文：`ml4health-main/jupyter/12_model_interpretation.ipynb`、`ml4health-main/jupyter/12b_advanced_shap.ipynb`
- 讲义：`ml4health-main/lectures/12_model_interpretation_teaching_doc.md`、`ml4health-main/lectures/12b_advanced_shap_teaching_doc.md`
- Lundberg, S. M., & Lee, S. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
- Ribeiro, M. T. et al. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* KDD.
- Shapley, L. S. (1953). *A Value for n-Person Games.* Contributions to the Theory of Games.
- Molnar, C. (2020). *Interpretable Machine Learning.* Lulu.com.
