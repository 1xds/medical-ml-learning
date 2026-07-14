# Module 15 笔记: SHAP 聚类分析

> 本模块的核心命题：**通过对 SHAP 值（而非原始特征值）聚类，发现数据集中具有不同模型解释模式的隐藏子群，从"全局解释"与"个体解释"之间搭建"子群解释"的桥梁。**

---

## 核心概念梳理

### SHAP 聚类是什么？

SHAP 聚类将每个样本的 SHAP 值向量视为该样本的"解释指纹"，在 SHAP 空间中进行聚类分析。与传统基于原始特征值的聚类不同，SHAP 聚类的分组依据不是"患者的特征值相似"，而是"模型的决策逻辑相似"。这种分析视角连接了全局解释（哪些特征总体重要）与局部解释（这个患者为何这样预测），填补了"哪些患者属于同一决策模式"这一中间层次的空白。

在医学场景中，SHAP 聚类可揭示具有不同预后驱动因素的亚群。例如，部分患者的存活预测主要由诊断方式驱动，而另一些患者则由年份与年龄共同决定——这种差异无法通过全局特征重要性排名捕捉。

### 原始特征聚类 vs SHAP 值聚类

| 维度 | 原始特征聚类 | SHAP 值聚类 |
|------|------------|------------|
| 输入 | 患者特征值（Age, year, ...） | 模型对每个患者的解释（SHAP 值） |
| 分组依据 | 特征值相似 | 决策逻辑相似 |
| 产出 | 临床表型群组 | 决策模式群组 |
| 解释 | "这些患者年龄相近" | "这些患者的预测主要由 year 驱动" |

### 相对重要性（Relative Importance）——创新指标

```python
# 聚类内特征重要性 / 全局特征重要性
rel_imp = this_cluster_importance / global_importance
```

- rel_imp > 1.0：该特征在此聚类中比全局平均更重要
- rel_imp < 1.0：该特征在此聚类中比全局平均更不重要

该指标使不同聚类之间的特征重要性可进行标准化比较，而非仅看绝对值。

---

## 代码精读

### Block 1: 模型训练与 SHAP 计算

```python
# 随机森林分类器
model = RandomForestClassifier(n_estimators=200, max_depth=8,
                                class_weight='balanced', random_state=RANDOM_STATE)
# 整个测试集用于 SHAP 聚类（更多样本使聚类更可靠）
X_shap = X_te_final[:]
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_shap)[1]  # VIVO 类别的 SHAP 值
```

**要点**：与交互分析模块使用 500 样本不同，聚类分析使用整个测试集（3000 样本），以确保聚类结构的可靠性。

### Block 2: 确定最佳聚类数

```python
# 尝试 2-5 个聚类，计算轮廓系数
for n in range(2, 6):
    km = KMeans(n_clusters=n, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(sv)
    score = silhouette_score(sv, labels)
    silhouette_scores[n] = score
# 选择轮廓系数最高的聚类数
best_n = max(silhouette_scores, key=silhouette_scores.get)
```

**要点**：轮廓系数衡量聚类的紧密度与分离度。本实验结果：k=2 的轮廓系数为 0.6209（最优），说明 SHAP 空间中存在两种截然不同的决策模式。

### Block 3: PCA 降维与聚类可视化

```python
# PCA 投影至 2D 用于可视化
pca = PCA(n_components=2, random_state=RANDOM_STATE)
sv_pca = pca.fit_transform(sv)
# 方差解释率
print(f"PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")
# 仅 2 个 PC 解释 89.28% 的 SHAP 方差 → 空间结构清晰
```

**要点**：2 个主成分即解释 89.28% 的方差，表明 SHAP 空间本质上是低维的——6 个特征的贡献可用 2 维有效表示。

### Block 4: 2×3 综合可视化面板

```python
# 子图 1: PCA 投影聚类结构
for c in range(best_n):
    mask = clusters == c
    ax.scatter(sv_pca[mask, 0], sv_pca[mask, 1], c=[cluster_colors[c]])

# 子图 2: 聚类特征重要性对比（分组条形图）
cluster_mean_shap = []
for c in range(best_n):
    mean_shap = np.abs(sv[mask]).mean(axis=0)  # 每聚类平均 |SHAP|

# 子图 3: 目标分布（VIVO 比例）per cluster
vivo_pct = y_shap[mask].mean() * 100

# 子图 5: 聚类特征均值热图
cluster_mean_feat[c] = X_shap[mask][:, top_heat_idx].mean(axis=0)
```

**要点**：2×3 面板整合了 PCA 投影、特征重要性、目标分布、聚类大小、特征均值与画像文本，提供全面的子群诊断。

### Block 5: 聚类画像与相对重要性

```python
# 相对重要性计算
this_imp = np.abs(sv[mask]).mean(axis=0)  # 聚类内平均 |SHAP|
global_imp = np.abs(sv).mean(axis=0)       # 全局平均 |SHAP|
rel_imp = this_imp / (global_imp + 1e-8)   # 相对重要性

# 找出该聚类中相对最重要的 3 个特征
top_rel_idx = np.argsort(rel_imp)[::-1][:3]
top_rel_feats = [f"{feature_names[i]}(x{rel_imp[i]:.2f})" for i in top_rel_idx]
```

**要点**：Cluster 1 的 Diagnostic.means 相对重要性为 1.65x——"若属于 Cluster 1，诊断方式对预测的影响是全局平均的 1.65 倍"。

---

## 关键收获

1. **SHAP 空间有清晰的聚类结构**：轮廓系数 0.6209，最佳聚类数 k=2，说明数据集中存在两种截然不同的决策模式
2. **Cluster 1（33.4%）= "特殊模式"**：Diagnostic.means（1.65x）异常突出，VIVO 仅 0.3%——几乎全部预测死亡，预测主要由诊断方式驱动
3. **Cluster 2（66.6%）= "主流模式"**：Extension（1.22x）、Code.Profession（1.20x）、Age（1.17x）相对突出，VIVO 61.8%——多个特征均衡贡献
4. **PCA 仅 2 个主成分解释 89.28% 方差**：SHAP 空间本质低维，结构清晰
5. **对 SHAP 值聚类 vs 对原始特征聚类**：Module 6 原始特征聚类 ARI ≈ -0.001（无结构），SHAP 聚类轮廓系数 0.63——患者特征不一定形成族群，但模型的解释模式形成了清晰的族群
6. **相对重要性是标准化比较的关键工具**：消除了聚类大小差异对绝对 |SHAP| 值的影响

| 维度 | Cluster 1 | Cluster 2 |
|------|-----------|-----------|
| 样本数 | 1,002 (33.4%) | 1,998 (66.6%) |
| VIVO 比例 | 0.3% | 61.8% |
| 突出特征 | Diagnostic.means (1.65x) | Extension (1.22x) |
| 决策模式 | 诊断方式主导 | 多特征均衡 |

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 为什么 SHAP 聚类有结构但原始特征聚类无结构？ | SHAP 反映模型决策逻辑而非数据表型 | 正是模型"对不同患者适用不同逻辑"的体现 |
| k=3 与 k=2 轮廓系数几乎相同？ | k=3 仅细分已有聚类而非发现新模式 | 选择更简洁的 k=2 |
| 聚类标签无临床含义？ | "Cluster 1" 不是诊断代码 | 通过特征画像赋予临床解释 |
| 聚类结果依赖模型？ | 不同模型产生不同 SHAP 值 | 在多个模型上验证聚类稳定性 |
| K-Means 假设球形簇？ | SHAP 空间可能非球形分布 | 使用 DBSCAN 或层次聚类交叉验证 |

---

## 与其他模块的联系

- **前置模块**：Module 12 SHAP 概述（SHAP 值基础）、Module 6 降维（PCA + K-Means 原始特征聚类、轮廓系数）
- **后续模块**：Module 16 SHAP 决策路径（个体化解释补充聚类层面的子群解释）、Module 17 Bootstrap（评估聚类结构的稳定性）
- **与研究工作的联系**：在宏基因组研究中，对 SHAP 值聚类可揭示长寿人群中具有不同代谢通路驱动因素的亚群——某些个体的长寿可能与特定菌群属的丰度驱动有关，而另一些则由饮食-菌群交互决定

---

## 参考资料

- 教程原文：`ml4health-main/jupyter/15_shap_clustering_analysis.ipynb`
- 讲义：`ml4health-main/lectures/15_shap_clustering_analysis_teaching_doc.md`
- Lundberg SM, Lee SI. A unified approach to interpreting model predictions. *NeurIPS* 2017.
- Rousseeuw PJ. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *J Comput Appl Math* 1987.
