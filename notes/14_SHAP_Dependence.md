# Module 14 笔记: 带特征分布的 SHAP 依赖图

> 本模块的核心命题：**SHAP 依赖图中的模式是否可信，取决于该模式发生在数据密集区域还是稀疏区域——引入数据密度维度，将模型解释的可信度量化。**

---

## 核心概念梳理

### 为什么需要数据密度维度？

传统 SHAP 依赖图展示两个维度：特征值（x 轴）与 SHAP 值（y 轴），但缺少关键信息——特征值在不同区域的**样本密度**。若某个区域的 SHAP 模式剧烈变化但仅有 2 个样本支撑，该模式的可信度远低于数据密集区域的一致趋势。本教程通过在依赖图底层叠加特征分布直方图，构建"双轴依赖图"，实现密度与模式的一体化评估。

在医学预测中，这一问题尤为突出。某些罕见诊断编码或极端年龄值的样本数量极少，但其 SHAP 值可能剧烈波动。若不加密度评估，可能将离群点噪声误读为真实信号，导致错误的临床结论。

### 三种图表对比

| 图表类型 | 展示信息 | 缺失信息 | 本教程改进 |
|---------|---------|---------|----------|
| 标准依赖图 | 特征值 × SHAP 值 | 数据密度分布 | 叠加分布直方图 |
| 直方图 | 特征值分布 | SHAP 模式 | 叠加依赖散点图 |
| **双轴依赖图** | 特征值 × SHAP 值 × 数据密度 | — | 三者合一 |

### 密度风险评估规则

```
Range Ratio = SHAP Range(Sparse Region) / SHAP Range(Dense Region)

Ratio < 2:   Low Risk    → 密集区与稀疏区 SHAP 变化接近 → 模式可信
Ratio 2-4:   Medium Risk → 稀疏区变化略大 → 需交叉验证
Ratio > 4:   High Risk   → 稀疏区变化远大于密集区 → 极端值驱动，谨慎解读
```

### 特征可信度映射

```
           Ratio < 2       2 ≤ Ratio ≤ 4       Ratio > 4
          ┌───────────────────────────────────────────
重要性高  │  黄金特征        慎重解读             需验证
重要性中  │  可靠特征        注意边缘             边缘主导
重要性低  │ 不重要特征       不重要特征           不重要特征
```

---

## 代码精读

### Block 1: 数据加载与模型训练

```python
# 加载癌症数据，映射二分类目标
df['target'] = df['Status.Vital'].map({'VIVO': 1, 'MORTO': 0})
# 使用 Boruta 精选的 6 个特征
feature_cols = ['Age', 'year', 'Code.Profession', 'Diagnostic.means',
                'Extension', 'Raca.Color']
# 分类变量编码 + 标准化
X_tr_final = scaler.fit_transform(X_tr_imp)
X_te_final = scaler.transform(X_te_imp)
```

### Block 2: SHAP 计算

```python
# 训练随机森林
model = RandomForestClassifier(n_estimators=200, max_depth=8,
                                class_weight='balanced', random_state=RANDOM_STATE)
# TreeExplainer 计算 SHAP
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_shap)[1]  # VIVO 类别的 SHAP 值
# 特征重要性排序
shap_importance = np.abs(sv).mean(0)
feature_order = np.argsort(shap_importance)[::-1]
```

### Block 3: 双轴依赖图——核心实现

```python
# 2×3 网格，每面板一个特征
for rank, feat_idx in enumerate(top6_idx):
    ax1 = axes[row, col]      # 主坐标轴: 直方图
    ax2 = ax1.twinx()         # 共享 x 轴: SHAP 散点图

    x_vals = X_shap[:, feat_idx]   # 特征值
    shap_vals = sv[:, feat_idx]    # SHAP 值

    # 左Y轴: 分布直方图（紫色半透明）
    counts, bin_edges = np.histogram(x_vals, bins=30)
    ax1.bar(bin_centers, counts, width=bin_width * 0.7,
            color='#4B0082', alpha=0.25, label='Frequency')

    # 右Y轴: SHAP 依赖散点图（按目标值着色）
    scatter = ax2.scatter(x_vals, shap_vals, c=y_shap, cmap='coolwarm', alpha=0.65)

    # 二次拟合趋势线
    z2 = np.polyfit(x_vals, shap_vals, 2)
    p2 = np.poly1d(z2)
    # R² 计算
    ss_res = np.sum((shap_vals - p2(x_vals))**2)
    ss_tot = np.sum((shap_vals - np.mean(shap_vals))**2)
    r2 = 1 - ss_res / ss_tot
```

**要点**：双轴设计将分布直方图（左Y轴）与 SHAP 散点图（右Y轴）叠加于同一 x 轴，实现密度与模式的一体化展示。二次拟合 R² 标注于趋势线上。

### Block 4: 密集区与稀疏区划分

```python
# 密集区: 25%-75% 分位数范围内的样本
dense_mask = x_vals >= np.percentile(x_vals, 25)
dense_mask &= x_vals <= np.percentile(x_vals, 75)
# 稀疏区: 密集区之外的样本
sparse_mask = ~dense_mask

# 计算 SHAP 值范围
dense_shap_range = np.ptp(shap_vals[dense_mask])  # 密集区 SHAP 范围
sparse_shap_range = np.ptp(shap_vals[sparse_mask]) # 稀疏区 SHAP 范围
# Range Ratio
range_ratio = sparse_shap_range / (dense_shap_range + 1e-8)
```

**要点**：使用 25%-75% 分位数定义密集区为默认推荐标准。密集区 SHAP 范围大于稀疏区时 Ratio < 1（如 Diagnostic.means 的 Ratio=0.13），说明模式由密集数据驱动——这正是期望的信号。

### Block 5: 密度对比汇总表

```python
# 输出特征评估汇总
print(f"{'Feature':<22} {'Importance':>10} {'R²_quad':>8} "
      f"{'Dense Range':>12} {'Sparse Range':>13} {'Ratio':>8} {'Risk':>10}")
for r in results_text:
    risk = "Low" if r['range_ratio'] < 2 else \
           ("Medium" if r['range_ratio'] < 4 else "High")
```

---

## 关键收获

1. **Diagnostic.means 是"全范围可信"特征**（Ratio=0.13）：密集区 SHAP 变化远大于稀疏区，模式由数据密集区域驱动
2. **Extension 的 Ratio=11.02 是极端教学示例**：密集区几乎无变化（SHAP 范围仅 0.048），稀疏区剧烈变化（0.525），模式可能被少数极端值驱动
3. **高 R² 不等于可信赖**：Extension R²=0.8298 但 Ratio=11.02——少数极端值的杠杆效应造成"虚假高 R²"；Diagnostic.means R²=0.9518 且 Ratio=0.13——可信的非线性模式
4. **数据密度提供模型解释的"置信度"维度**：每次看到依赖图应问"模式发生在数据密集区还是边缘？"
5. **Ratio < 1 也是好信号**：密集区 SHAP 范围 > 稀疏区范围时，模式集中在可信数据区域
6. **三维评估框架**：结合重要性（Module 12）、非线性 R²（Module 12b）、交互强度（Module 13）、密度风险 Ratio（本模块），形成完整的特征可信度诊断体系

| 特征 | 重要性 | 非线性 (R²_quad) | 交互强度 | 密度风险 (Ratio) | 综合诊断 |
|------|--------|----------------|---------|----------------|---------|
| Diagnostic.means | 极强 | 0.9518 | 高 | 0.13 (Low) | 第一可信特征 |
| year | 强 | 0.7618 | 高 | 1.09 (Low) | 全范围可信 |
| Extension | 低 | 0.8298 | 中 | 11.02 (High) | 模式可能不可靠 |
| Age | 低 | 0.7871 | 中 | 2.12 (Medium) | 主流模式可信 |

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| Extension 高 R²=0.83 但 Ratio=11.02？ | 少数极端值的杠杆效应拉高 R² | 报告 Ratio，标注"稀疏区驱动" |
| Ratio > 4 是否意味特征完全不可信？ | 稀疏区模式可能是真实罕见医学信号 | 与临床专家确认极端值合理性 |
| 25%-75% 分位数边界主观 | 不同定义影响密集区范围 | 使用多个分位数定义做敏感性分析 |
| 高重要性 + 高 Ratio 的解读？ | 特征重要性可能被极端值虚假抬升 | 截尾后重训模型，比较 AUC 变化 |
| 小样本密集/稀疏划分不稳 | 500 样本的划分可能不稳健 | 增加样本量或使用 Bootstrap |

---

## 与其他模块的联系

- **前置模块**：Module 12 SHAP 概述（依赖图基础）、Module 12b 高级 SHAP（二次拟合 R²）、Module 13 交互分析（交互依赖图）
- **后续模块**：Module 15 SHAP 聚类（不同子群可能有不同的密度风险）、Module 17 Bootstrap（用重采样评估密度划分的稳健性）
- **与研究工作的联系**：在 AutoDock Vina 虚拟筛选中，某些化合物描述符（如分子量、logP）可能在极端值区域对结合亲和力产生剧烈影响，但样本极少——需通过密度 Ratio 评估 docking 评分模式的可信度

---

## 参考资料

- 教程原文：`ml4health-main/jupyter/14_shap_dependence_distribution.py.ipynb`
- 讲义：`ml4health-main/lectures/14_shap_dependence_distribution_teaching_doc.md`
- Lundberg SM et al. From local explanations to global understanding with explainable AI for trees. *Nat Mach Intell* 2020.
- Apley DW, Zhu J. Visualizing the effects of predictor variables in black box supervised learning models. *J R Stat Soc B* 2020.
