# Module 16 笔记: SHAP 决策路径分析

> 本模块的核心命题：**对于单个患者，从模型基准值到最终预测值之间，每个特征如何一步步将预测推向最终结果——将"黑箱预测"转化为可追踪的决策步骤。**

---

## 核心概念梳理

### SHAP 决策路径是什么？

SHAP 决策路径分析将单个预测过程分解为一系列可追踪的步骤：从基准值（Base Value）出发，按 |SHAP| 从大到小排序，逐一累加每个特征的贡献，最终到达该患者的预测值。这一分析将"黑箱"预测的白盒化，回答"中间发生了什么"这一局部解释的核心问题。

三个核心概念：

| 概念 | 定义 | 含义 |
|------|------|------|
| 基准值 (Base Value) | log(p̄/(1-p̄))，p̄ 为训练集中 VIVO 平均概率 | 模型"默认预测"——无任何患者信息时的预测 |
| SHAP 值 | 每个特征对单个预测的贡献（log-odds 变化量） | 正→推高存活概率；负→推低存活概率 |
| 决策路径 | 从基准值出发，按 |SHAP| 逐步累加至最终预测 | 将预测过程"分段叙述" |

### 从全局到局部再到个体路径

| 教程 | 视角 | 回答的问题 |
|------|------|-----------|
| 12 | 全局 | 所有患者中哪些特征总体最重要？ |
| 13 | 交互 | 哪些特征对相互影响？ |
| 14 | 可信度 | 依赖图模式在稀疏区还可信吗？ |
| 15 | 子群 | 哪些患者的决策模式相似？ |
| **16** | **个体** | **这个患者的预测是如何一步步生成的？** |

### 三列可视化设计

每行对应一个患者样本，三列分别回答三个问题：

| 列 | 图表类型 | 回答的问题 | 关键元素 |
|----|---------|-----------|---------|
| 列 1 | 决策路径折线图 | "如何从起点到终点？" | Base→F1→F2→...→Final 的累积 SHAP |
| 列 2 | SHAP 贡献条形图 | "每个特征贡献了多少？" | 绿色=推高VIVO，红色=推低 |
| 列 3 | 特征值雷达图 | "这些特征的值是多少？" | 归一化特征值的六边形画像 |

### 五组代表性样本选取

| # | 标签 | P(VIVO) | 实际 | 选样目的 |
|---|------|---------|------|---------|
| 1 | 低(死亡) | ~0.003 | MORTO | 模型几乎确定死亡——驱动因素 |
| 2 | 中(死亡) | ~0.013 | MORTO | 与低死亡的驱动因素是否不同 |
| 3 | 低(存活) | ~0.606 | VIVO | "边际患者"——信心不足的存活判断 |
| 4 | 中(存活) | ~0.817 | VIVO | 信心中等的存活判断 |
| 5 | 高(存活) | ~0.897 | VIVO | 哪些特征推动大幅提升 |

---

## 代码精读

### Block 1: SHAP 计算与基准值提取

```python
# SHAP TreeExplainer 计算
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_shap)[1]  # VIVO 类别 SHAP 值

# 基准值提取
expected_value = explainer.expected_value
base_value = expected_value[1]  # VIVO 类别的基准 log-odds
# base_value ≈ 0.5002 → 对应 p ≈ 62.2% VIVO 概率
```

**要点**：基准值 0.5002 对应约 62.2% 存活概率。由于使用了 class_weight='balanced'，基准值不等于训练集的简单类别比例。

### Block 2: 代表性样本选取

```python
def find_representative_samples(probs, quantiles=[0.05, 0.50, 0.95]):
    """找到概率接近指定分位数的样本索引"""
    for q in quantiles:
        target = np.quantile(probs, q)
        idx = np.argmin(np.abs(probs - target))
        indices.append(idx)
    return indices

# 分别为 VIVO 和 MORTO 样本选取代表性点
vivo_rep = find_representative_samples(vivo_probs, [0.2, 0.5, 0.8])
morto_rep = find_representative_samples(morto_probs, [0.1, 0.3])
```

**要点**：按预测概率分位数选样而非随机选样，确保覆盖不同预测置信度区间，便于比较决策路径的差异。

### Block 3: 决策路径图（列 1）——核心实现

```python
# 按 |SHAP| 排序特征
sorted_idx = np.argsort(np.abs(sample_shap))[::-1][:top_k]

# 累积 SHAP 值计算
cum_shap = np.zeros(top_k + 1)
cum_shap[0] = 0  # 从 0 开始（基准值为中心点）
for i, idx in enumerate(sorted_idx):
    cum_shap[i + 1] = cum_shap[i] + sample_shap[idx]  # 逐步累加

# 折线图绘制
path_color = '#2ecc71' if cfg['actual'] == 1 else '#e74c3c'
ax.plot(range(n_steps + 1), cum_shap[:n_steps + 1], 'o-', linewidth=2.5)

# 基准值 + 预测值参考线
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.4)
ax.axhline(y=np.log(prob / (1 - prob + 1e-10)), color='blue', linestyle='--')
```

**要点**：路径颜色按实际结局区分（绿=存活，红=死亡）。每步标注特征名、特征值与 SHAP 值。最终累积值等于该样本所有 SHAP 值之和。

### Block 4: SHAP 贡献条形图（列 2）

```python
# 水平条形图：特征名 → SHAP 值
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_shap_vals = sample_shap[sorted_idx]
bar_colors = ['#27ae60' if s > 0 else '#e74c3c' for s in sorted_shap_vals]
ax_bar.barh(range(len(sorted_features)), sorted_shap_vals, color=bar_colors)
```

**要点**：条形图特征顺序与决策路径一致（按 |SHAP| 排序），便于将"单项贡献"与"累积路径"一一对应。

### Block 5: 特征值雷达图（列 3）

```python
# 归一化特征值至 [0, 1]
global_min = X_shap[:, sorted_idx].min(axis=0)
global_max = X_shap[:, sorted_idx].max(axis=0)
feature_vals_norm = (sample_feat[sorted_idx] - global_min) / (global_max - global_min + 1e-8)

# 雷达图绘制
angles = np.linspace(0, 2 * np.pi, len(sorted_idx), endpoint=False).tolist()
ax_radar.plot(angles, vals_plot, 'o-', linewidth=2, color=path_color)
ax_radar.fill(angles, vals_plot, alpha=0.2, color=path_color)
```

**要点**：雷达图将特征值（归一化后）与 SHAP 方向关联。例如 year=0.82 + SHAP=+0.30 的组合解读："year 推高了预测，因为该患者的 year 值很高"。

---

## 关键收获

1. **决策路径将"黑箱预测"转化为可追踪步骤**：从基准值出发，每一步是特征贡献的累加，终点为最终预测
2. **五组样本展示不同预测置信度下的决策逻辑差异**：死亡预测路径以"下降"为主，存活预测以"上升"为主
3. **雷达图连接特征值与 SHAP 贡献**：将标准化的特征值与 SHAP 方向关联，实现"为什么 year 推高了预测？因为该患者 year 值很高"的双维解读
4. **路径排序影响"叙事"但不改变终点**：关注累积值比关注步骤顺序更重要
5. **路径形状可分类**："陡升型"（少数强特征主导）、"缓升型"（多特征均衡贡献）、"陡降型"（year/Age 为主要死亡指示符）

| 路径形状 | 特征 | 含义 |
|---------|------|------|
| 陡升型 | 样本 4, 5 | 前 2-3 步完成大部分上升，少数强特征主导 |
| 缓升型 | 样本 3 | 多特征共同贡献，每步增量不大 |
| 陡降型 | 样本 1, 2 | 前几步快速下降，year/Age 为主要死亡指示 |

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 路径顺序是人为的？ | 按 |SHAP| 排序是常见选择，其他排序产生不同叙事 | 关注累积值本身，而非顺序 |
| 只展示 Top K 特征？ | 低贡献特征被隐藏 | 检查未展示特征的贡献之和 |
| 不能反映交互效应？ | 路径是可加的，但模型可能有交互 | 结合 Module 13 交互分析 |
| 单个样本解读无法推广？ | 个体路径不能代表群体 | 结合 Module 15 聚类分析 |
| log-odds 空间的路径 vs 概率空间？ | 非线性转换改变路径形状 | 建议在 log-odds 空间展示以保持可加性 |

---

## 与其他模块的联系

- **前置模块**：Module 12 SHAP 概述（Waterfall Plot 局部解释）、Module 15 SHAP 聚类（子群解释模式）
- **后续模块**：Module 17 Bootstrap（评估单样本 SHAP 值的稳定性——同一个患者在不同 Bootstrap 模型下的决策路径是否一致？）
- **与研究工作的联系**：在 EcMurJ 蛋白-化合物对接场景中，对单个化合物的 docking 评分决策路径分析可揭示哪些分子描述符（疏水性、柔性、电荷）一步步将结合亲和力推向高分或低分，指导化合物优化方向

---

## 参考资料

- 教程原文：`ml4health-main/jupyter/16_shap_decision_path.ipynb`
- 讲义：`ml4health-main/lectures/16_shap_decision_path_teaching_doc.md`
- Lundberg SM et al. From local explanations to global understanding with explainable AI for trees. *Nat Mach Intell* 2020.
- Ribeiro MT, Singh S, Guestrin C. "Why should I trust you?": explaining the predictions of any classifier. *KDD* 2016. (LIME 对比)
