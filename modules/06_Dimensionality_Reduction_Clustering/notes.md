# Module 06 笔记: 降维与聚类

> 本模块的核心命题：**理解高维数据的空间结构——通过降维技术可视化数据结构、通过聚类分析验证天然可分性、通过 PCA 建模实现信息压缩。**

---

## 核心概念梳理

### 降维与聚类的核心问题

当特征维度超过 3 时，人类无法直接通过散点图观察数据分布。降维技术的本质是将高维数据映射到低维空间，使研究者能够从宏观层面理解数据结构。聚类分析则回答一个更根本的问题：**如果不使用标签信息，数据中自然存在的分组结构是什么？这些分组是否对应真实的临床有意义类别？**

本模块围绕三个递进问题展开：
1. **良恶性患者是否天然可分？**（PCA/t-SNE/UMAP 可视化 + 聚类分析）
2. **特征是否存在冗余？**（PCA 方差解释率 + 维度灾难实验）
3. **降维是否会损失信息？**（不同 PCA 维度下的模型性能对比）

### PCA 与特征选择的本质区别

| 对比维度 | 特征选择 | PCA |
|---------|---------|-----|
| 输出 | 原始特征的子集 | 原始特征的线性组合 |
| 可解释性 | 高（"Age 是重要特征"） | 低（"PC1 是 Age+year+Code.Profession 的组合"） |
| 数学本质 | 子集选择 | 基变换 + 数据投影 |
| 维度可调 | 离散（保留哪些） | 连续（选择几个 PC） |

### 降维方法对比

| 方法 | 类型 | 保留结构 | 速度 | 适用场景 |
|------|------|---------|------|---------|
| PCA | 线性 | 全局方差最大方向 | 快 | 基线降维、去相关性、ML 预处理 |
| t-SNE | 非线性（概率） | 局部邻域 | 慢 O(n²) | 可视化探索、发现局部簇结构 |
| UMAP | 非线性（拓扑） | 局部 + 全局 | 中 O(n log n) | 大规模数据可视化、论文趋势方法 |

---

## 代码精读

### Block 1: 高维特征集构造（~22 维）

```python
from sklearn.decomposition import PCA                                  # 主成分分析
from sklearn.manifold import TSNE                                     # t-SNE
from sklearn.cluster import KMeans, AgglomerativeClustering           # 聚类方法
from sklearn.neighbors import KNeighborsClassifier                    # 维度灾难实验

# 11 个原始特征
raw_feats = ['Age', 'year', 'Gender', 'Code.Profession', 'Code.of.Morphology',
             'Diagnostic.means', 'Extension', 'Laterality',
             'Raca.Color', 'State.Civil', 'Degree.of.Education']

# 11 个派生特征
df_feat['Age_Group'] = ...
df_feat['Age_Sq'] = df_feat['Age'] ** 2
df_feat['Age_Log'] = np.log1p(df_feat['Age'])        # 对数变换
df_feat['Profession_Log'] = np.log1p(df_feat['Code.Profession'])
df_feat['Age_x_Year'] = df_feat['Age'] * df_feat['Year_From_2000']  # 交互项
# ... 等共11个派生特征

n_dims = len(all_features)  # 22 维
# 插补 + 标准化
X_full_arr = scaler.fit_transform(imputer.fit_transform(X_full))

# 流形学习用小样本（t-SNE/UMAP计算量大）
X_manifold = X_full_arr[midx]  # 15,000 样本
```

要点：
1. 构造 22 维特征集：11 个原始特征 + 11 个派生特征（含平方、对数、交互）
2. 计算量分级：KNN/PCA 用 50,000 样本；t-SNE/UMAP/聚类用 15,000 样本

### Block 2: 维度灾难实验——KNN 性能随维度变化

```python
dims_to_test = [2, 4, 6, 8, 10, 14, 18, 22, n_dims]

for d in dims_to_test:
    pca_d = PCA(n_components=min(d, X_train.shape[1]))
    X_pca_d = pca_d.fit_transform(X_train)         # PCA 降维到 d 维
    X_test_pca_d = pca_d.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5)      # k=5
    knn.fit(X_pca_d, y_train)
    auc = roc_auc_score(y_test, knn.predict_proba(X_test_pca_d)[:, 1])
```

**实验数据**：

| 维度 | KNN AUC | 解读 |
|------|---------|------|
| 2 | 0.7591 | 信息不足，无法有效区分 |
| 6 | 0.8836 | 信息量快速增大 |
| **10** | **0.9025** | **接近性能峰值** |
| 14 | 0.9068 | 趋于平稳 |
| 22 | 0.9072 | 平台期，额外维度收益递减 |

要点：
1. 前 10 个维度提供了大部分有效信息
2. 14 维后 AUC 几乎不再增长——存在显著的维度冗余
3. 这从性能角度验证了"特征选择可以降维不降质"的结论

### Block 3: PCA 方差解释率

```python
pca_full = PCA()
pca_full.fit(X_full_arr)

explained_ratio = pca_full.explained_variance_ratio_     # 每个 PC 的方差解释比例
cumulative_ratio = np.cumsum(explained_ratio)             # 累积比例

# 达到阈值需要的 PC 数
for threshold, label in [(0.90, '90%'), (0.95, '95%'), (0.99, '99%')]:
    n = np.argmax(cumulative_ratio >= threshold) + 1
```

**方差解释率**：

| 主成分 | 解释方差 | 累积 | 解读 |
|--------|---------|------|------|
| PC1 | **27.53%** | 27.53% | 最大方差方向，超过 1/4 的信息 |
| PC2 | 15.40% | 42.93% | 与 PC1 正交的第二大方向 |
| PC5 | 6.94% | 68.52% | 前 5 个 PC 解释约 2/3 信息 |
| **PC10** | **3.48%** | **91.19%** | **10 个 PC 即达 91% 方差** |
| PC15 | 0.37% | 99.14% | 15 个 PC 到 99% |

要点：
1. 方差解释呈长尾分布——前几个 PC 贡献大，尾部 PC 贡献极小
2. PC1 本身解释了 27.53%——数据在某个方向上高度结构化

### Block 4: PCA 二维投影可视化

```python
X_pca_2d = pca_full.transform(X_manifold)[:, :2]       # 投影到 PC1-PC2

ax.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
           c=['#3498db' if y==0 else '#e74c3c' for y in y_manifold],
           alpha=0.4, s=5)                              # 半透明小点，展示密度分布

# PC3 补充可视化
ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 2], ...)        # PC1 vs PC3
ax.scatter(X_pca_3d[:, 1], X_pca_3d[:, 2], ...)        # PC2 vs PC3
```

要点：
1. PC1-PC2 投影中，VIVO（红色）和 MORTO（蓝色）呈现部分分离趋势但不彻底
2. 两类在中心区域高度重叠——线性模型无法完全分离，与 AUC ≈ 0.90 一致
3. PC3 补充视角提供了额外的分离证据

### Block 5: t-SNE 降维与随机性演示

```python
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=RANDOM_STATE)
X_tsne = tsne.fit_transform(X_manifold)

# 不同 perplexity 对比
for perp in [5, 30, 80]:
    tsne_p = TSNE(n_components=2, perplexity=perp, max_iter=500, random_state=RANDOM_STATE)
    X_tsne_p = tsne_p.fit_transform(X_manifold)

# 【关键演示】t-SNE 随机性
for seed in [0, 42, 99]:
    tsne_r = TSNE(random_state=seed)
    # 同一数据、不同种子 → 完全不同的布局！
```

要点：
1. t-SNE 的 perplexity 控制"每个点考虑多少邻居"——小值关注局部，大值关注全局
2. **核心误区警告**：t-SNE 图中类间分离明显 ≠ 模型性能好

### Block 6: UMAP 降维与三方法对比

```python
import umap
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
X_umap = reducer.fit_transform(X_manifold)

# PCA vs t-SNE vs UMAP 三图并排
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, title, data in zip(axes, ['PCA', 't-SNE', 'UMAP'], [X_pca_2d, X_tsne, X_umap]):
    ax.scatter(data[:, 0], data[:, 1], c=colors_pca, alpha=0.4, s=3)
```

要点：
1. UMAP 比 t-SNE 快 2-10 倍（O(n log n) vs O(n²)）
2. UMAP 保留更多全局结构，在医学 AI 论文中使用频率上升

### Block 7: 聚类分析——无监督能发现天然分组吗？

```python
for k in [2, 3, 5]:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels_km = km.fit_predict(X_manifold)
    sil = silhouette_score(X_manifold, labels_km)         # 轮廓系数（聚类质量）
    ari = adjusted_rand_score(y_manifold, labels_km)      # ARI（与真实标签的一致性）
```

**聚类结果**：

| 方法 | K | Silhouette | ARI | 解读 |
|------|---|-----------|-----|------|
| K-Means | 2 | 0.1588 | **0.0081** | 与真实标签几乎无关联 |
| K-Means | 3 | — | 0.0135 | 略有改善但仍极低 |
| K-Means | 5 | — | 0.0179 | 无实质性改善 |

要点：
1. ARI ≈ 0 → 聚类结果与临床标签几乎没有重合
2. **结论**：存活和死亡患者的特征差异不足以在无监督环境下被自动发现，有监督学习至关重要

### Block 8: PCA 用于机器学习建模

```python
pca_configs = [
    ('Original (all 22 dims)', None),      # 全特征基线
    ('PCA-5', PCA(n_components=5)),         # 压缩到 5 维
    ('PCA-10', PCA(n_components=10)),       # 压缩到 10 维
    ('PCA-15', PCA(n_components=15)),       # 压缩到 15 维
    ('PCA-20', PCA(n_components=20)),       # 压缩到 20 维
]

for name, pca_obj in pca_configs:
    if pca_obj is not None:
        X_tr_pca = pca_obj.fit_transform(X_train)   # 训练集fit
        X_te_pca = pca_obj.transform(X_test)         # 测试集transform
    lr = LogisticRegression(class_weight='balanced', max_iter=5000)
    lr.fit(X_tr_pca, y_train)
```

**PCA 降维后模型性能**：

| 方法 | AUC | Recall | Brier | 解读 |
|------|-----|--------|-------|------|
| Original (22d) | **0.9024** | 0.8843 | 0.1250 | 全特征基线 |
| PCA-5 | 0.8222 | 0.7727 | 0.1734 | 信息损失较大（仅 68.5% 方差） |
| PCA-10 | 0.8935 | 0.8763 | 0.1312 | 达全特征 AUC 的 99% |
| PCA-15 | 0.9018 | 0.8845 | 0.1255 | 接近原始性能 |
| PCA-20 | 0.9023 | 0.8842 | 0.1250 | 完全恢复 |

### Block 9: PCA vs 特征选择对比

```python
sets_compare = {
    'Full (22 dims)': (X_train, X_test),
    'Filter Top 10': (X_train_f, X_test_f),   # ANOVA Top 10
    'RF Top 10': (X_train_rf, X_test_rf),     # RF Importance Top 10
    'PCA-10': (X_train_p10, X_test_p10),      # 10 个主成分
}
```

**10 维对比**：

| 方法 | AUC | 可解释性 |
|------|-----|---------|
| Full (22d) | 0.9024 | 中 |
| Filter Top 10 | 0.8902 | **高**（保留原始特征含义） |
| RF Top 10 | 0.8919 | **高** |
| PCA-10 | **0.8935** | 低（PC 无物理含义） |

---

## 关键收获

1. **高维数据存在显著的维度冗余**：KNN 性能在 10 维后趋于平稳，PCA 的 10 个主成分解释了 91% 的方差。前 6 个 PC 的累积方差约 73%，后 15+ 个 PC 的贡献微乎其微。

2. **PCA 是线性特征变换而非特征选择**：PCA 生成的是原始特征的线性组合，丢弃了可解释性但获得了信息压缩效率。使用场景取决于目标——需要可解释性则用特征选择，需要压缩去噪则用 PCA。

3. **t-SNE 是可视化工具，不能用于模型评价**：三个随机种子给出三种完全不同的布局，说明 t-SNE 的视觉分离效果不代表分类性能。t-SNE 只能用于探索数据结构、辅助提出假设。

4. **UMAP 保留全局+局部结构**：比 t-SNE 快 2-10 倍，对参数更稳健，当前的医学 AI 论文趋势偏好 UMAP 作为标准降维可视化工具。

5. **聚类分析揭示了数据天然结构的真相**：K-Means 的 ARI ≈ 0，说明无监督方法无法自动发现存活/死亡标签对应的分组。PCA 中可见的分离趋势不足以支撑"天然可分"的判断——有监督学习利用标签信息是关键。

6. **PCA-10 等价于全特征性能的 99%**：在实际应用中，10 个主成分可以替代 22 个全特征进行后续建模，在保持性能的同时减少特征数量和训练开销。

7. **PCA vs 特征选择不矛盾**：两者解决不同的问题——信息压缩用 PCA，可解释性用特征选择。四者在 10 维下 AUC 差距仅 0.012，可根据实际需求选择。

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| PCA 投影后类别重叠严重 | 数据在高维空间的类别边界模糊 | 尝试非线性降维（t-SNE/UMAP）或非线性分类器 |
| K-Means 的 ARI 近似于 0 | 数据中的类别在高维空间中不是球形簇 | 尝试 DBSCAN、Gaussian Mixture 等非球形聚类方法 |
| t-SNE 结果不可复现 | t-SNE 的目标函数是非凸的，不同初始条件收敛到不同局部最优 | 固定 random_state，或使用 UMAP（更稳定） |
| 维度灾难实验 AUC 先升后平 | 前几维提供关键信息，后续维度主要是冗余和噪声 | 使用 PCA 方差解释率确定合理的降维维度 |
| PCA-5 AUC 远低于 PCA-10 | 5 个 PC 仅解释 68.5% 方差，信息损失过大 | 至少使用达到 90% 方差的 PC 数量 |

---

## 与其他模块的联系

- **前置模块**：Module 04（特征工程）和 Module 05（特征选择）分别通过构造和筛选处理特征冗余——本模块从"特征变换"角度提供了第三种策略。PCA 生成的正交主成分天然打破了 Module 05 中的共线性问题。
- **后续模块**：Module 07（数据泄漏）中将讨论 PCA 的防泄漏规范——必须在训练集上 `fit`，在测试集上 `transform`
- **与研究工作的联系**：在 EcMurJ 分子对接研究中，分子描述符的 PCA 可视化可用于探索 236K 化合物库的化学空间分布，判断对接打分最高的化合物是否聚集在特定的化学空间中。t-SNE/UMAP 可用于展示活性化合物与非活性化合物在化学空间中的分离趋势。在宏基因组研究中，PCoA（基于距离矩阵的主坐标分析）是微生物群落 beta 多样性分析的标准方法，聚类分析（如基于 Bray-Curtis 距离的层次聚类）用于发现微生物群落类型（enterotypes）。

---

## 参考资料

- 教程原文：`ml4health-main/jupyter/06_dim_reduction_clustering.ipynb`
- 讲义：`ml4health-main/lectures/06_dim_reduction_teaching_doc.md`
- van der Maaten, L. & Hinton, G. (2008). Visualizing Data using t-SNE. *JMLR*, 9, 2579-2605.
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection. *arXiv:1802.03426*.
- Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
