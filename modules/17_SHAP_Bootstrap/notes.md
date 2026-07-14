# Module 17 笔记: SHAP 稳定性 Bootstrap 分析

> 本模块的核心命题：**单次模型训练得出的特征重要性结论，在数据扰动下是否仍然成立——通过 Bootstrap 重采样将"点估计"转化为"区间估计"，量化特征重要性的可信度与排名稳定性。**

---

## 核心概念梳理

### 为什么需要稳定性分析？

Module 12-16 所有 SHAP 分析均基于一次训练的模型。然而，若训练数据略有变化，特征重要性排序可能发生显著改变——year 是否永远排名第一？Code.Profession 的"重要性"是真实信号还是抽样噪声？Bootstrap 稳定性分析通过多次有放回重采样训练模型，为每个特征的重要性赋予置信区间与变异系数，回答"特征重要性的结论是否可靠"这一根本问题。

在医学论文中，审稿人常质疑"单次报告"的可靠性。Bootstrap 将"Diagnostic.means 的 SHAP=0.24"升级为"0.21 ± 0.01, 95% CI [0.19, 0.23]"，从点估计到区间估计，显著提升结论的科学可信度。

### Bootstrap 方法原理

```
原始数据 (n=5000)
  → Bootstrap 样本 1: 有放回抽样 n 个 → 训练模型 → SHAP 重要性
  → Bootstrap 样本 2: 有放回抽样 n 个 → 训练模型 → SHAP 重要性
  → ... 重复 30 次 ...
  → 30 组 SHAP 重要性 → 均值、标准差、CI、CV、排名相关性
```

每次 Bootstrap 抽样模拟"略有差异的训练数据"，某些患者出现多次、某些被遗漏，从而评估结论对数据扰动的敏感性。

### 五个量化指标

| 指标 | 公式 | 含义 | 判断标准 |
|------|------|------|---------|
| Mean SHAP | E[importance] | 30 次 Bootstrap 平均重要性 | 越高则特征越重要 |
| Std SHAP | σ[importance] | 重要性波动程度 | 越低越稳定 |
| CV (变异系数) | σ / μ | 相对波动量 | <0.1: 高度稳定; 0.1-0.3: 中等; >0.3: 不稳定 |
| CI Width | P97.5 - P2.5 | 95% 置信区间宽度 | 越窄越可信 |
| Rank ρ | Spearman(rank_i, rank_mean) | 单次排名与平均排名的一致性 | >0.9: 稳定; 0.7-0.9: 中等; <0.7: 不稳定 |

### 交叉验证 vs Bootstrap 稳定性

| 方法 | 评估对象 | 回答的问题 |
|------|---------|-----------|
| K-Fold CV | 模型泛化性能 | 模型在不同数据划分下 AUC 波动多少？ |
| Bootstrap SHAP | 特征重要性 | 特征重要性在不同数据扰动下波动多少？ |

两者互补：CV 好的模型不保证解释稳定性；Bootstrap 稳定的特征不保证模型性能。好的论文需同时报告两者。

---

## 代码精读

### Block 1: 数据加载与 Bootstrap 配置

```python
N_SAMPLES = 5000  # Bootstrap 多轮训练，样本量不能太大
N_BOOTSTRAP = 30  # 30 次迭代：合理的精度-时间平衡

# 6 个 Boruta 精选特征
feature_cols = ['Age', 'year', 'Code.Profession', 'Diagnostic.means',
                'Extension', 'Raca.Color']
```

**要点**：样本量降至 5000（而非 10000）以平衡计算成本。30 次 Bootstrap 为标准推荐——正式论文可增至 100-200 次。

### Block 2: Bootstrap 迭代训练与 SHAP 计算

```python
bootstrap_shap_importance = []  # 存储每次全局重要性
bootstrap_aucs = []             # 存储每次 AUC

for i in range(N_BOOTSTRAP):
    # 有放回抽样
    indices = np.random.choice(len(X_all), size=len(X_all), replace=True)
    X_boot, y_boot = X_all[indices], y_all[indices]

    # 在 bootstrap 样本内标准化
    scaler = StandardScaler()
    X_boot_scaled = scaler.fit_transform(X_boot)

    # 训练模型（减少树数加速）
    model = RandomForestClassifier(
        n_estimators=100, max_depth=6,  # 加速配置
        class_weight='balanced', random_state=i, n_jobs=-1, oob_score=True)
    model.fit(X_boot_scaled, y_boot)

    # SHAP 分析（用 bootstrap 样本的子集）
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_shap_sample)[1]
    importance = np.abs(sv).mean(axis=0)  # 全局特征重要性
    bootstrap_shap_importance.append(importance)
```

**要点**：每次迭代独立标准化、训练、计算 SHAP。使用 reduced 配置（100 棵树、深度 6）加速，OOB 评分替代测试集 AUC。

### Block 3: 稳定性统计量计算

```python
# 30×6 重要性矩阵
bootstrap_shap_importance = np.array(bootstrap_shap_importance)

# 基本统计量
mean_imp = bootstrap_shap_importance.mean(axis=0)  # 平均重要性
std_imp = bootstrap_shap_importance.std(axis=0)     # 标准差
cv_imp = std_imp / (mean_imp + 1e-8)               # 变异系数

# 95% 置信区间
ci_lower = np.percentile(bootstrap_shap_importance, 2.5, axis=0)
ci_upper = np.percentile(bootstrap_shap_importance, 97.5, axis=0)
ci_width = ci_upper - ci_lower                      # CI 宽度

# 排名稳定性
base_ranking = np.argsort(mean_imp)[::-1]           # 平均排名
rank_correlations = []
for boot_imp in bootstrap_shap_importance:
    boot_ranking = np.argsort(boot_imp)[::-1]
    corr, _ = spearmanr(base_ranking, boot_ranking)
    rank_correlations.append(corr)
```

**要点**：变异系数 CV 是无量纲的稳定性度量，消除了均值大小的影响。排名稳定性使用 Spearman 等级相关，评估不同 Bootstrap 迭代间的排序一致性。

### Block 4: 六子图可视化

```python
# 子图 1: 特征重要性与 95% CI（条形图 + 误差棒）
ax1.bar(x_pos, mean_imp[top_idx],
        yerr=[mean_imp - ci_lower, ci_upper - mean_imp], capsize=5)

# 子图 2: CV 稳定性排名（水平条形图，绿=稳定 红=不稳定）
colors = plt.cm.RdYlGn_r(1 - cv_imp[cv_sorted] / cv_imp.max())
ax2.barh(range(top_n), cv_imp[cv_sorted], color=colors)

# 子图 3: Bootstrap 分布箱线图（Top 3 特征）
ax3.boxplot(box_data, labels=[feature_names[idx] for idx in top_3_idx])

# 子图 4: 重要性轨迹时间序列（每条线=一个特征）
for k, idx in enumerate(top_5_idx):
    ax4.plot(bootstrap_shap_importance[:, idx], label=feature_names[idx])

# 子图 5: 排名稳定性分布直方图
ax5.hist(rank_correlations, bins=15)
ax5.axvline(np.mean(rank_correlations), color='red', linestyle='--')

# 子图 6: CI 宽度排名（越高=不确定性越大）
ax6.bar(range(top_n), ci_width[ci_sorted])
```

**要点**：六子图从不同角度呈现稳定性信息——重要性+CI、CV排名、分布形态、时间轨迹、排名一致性与不确定性排名。

---

## 关键收获

1. **Diagnostic.means 和 Extension 是高度稳定的特征**（CV < 0.07）：数据扰动下的一致性确认了"真正的信号"。越重要的特征通常越稳定——真实信号在数据扰动下"幸存"
2. **Code.Profession 是相对不稳定的特征**（CV=0.200）：其重要性受抽样噪声影响大。低重要性 + 高 CV 意味着"重要性主要由抽样噪声决定"
3. **平均排名稳定性 ρ=0.4114**——整体排序不稳定，存在迭代中排名完全相反的情况（ρ=-0.4286）。在论文中不应只报告"唯一点排名"，应报告排名范围或置信区间
4. **Bootstrap 将点估计变为区间估计**："Diagnostic.means 重要性 0.21 [95% CI 0.19-0.23]"比"0.24"更科学
5. **交叉验证与 Bootstrap 互补**：CV 评估性能稳定性，Bootstrap 评估解释稳定性——好的论文需同时报告两者
6. **"高 CV 但低重要性"的解读陷阱**：不应简单判断"不稳定→不可信→放弃"，而应理解为"重要性主要由噪声决定，应谨慎将其作为重要特征报告"

### 完整稳定性表格

| 排名 | 特征 | Mean SHAP | Std | CV | CI Width | 稳定性 |
|------|------|-----------|-----|-----|---------|-------|
| 1 | Diagnostic.means | 0.2106 | 0.0092 | 0.0439 | 0.0314 | 高度稳定 |
| 2 | Extension | 0.1622 | 0.0108 | 0.0664 | 0.0386 | 高度稳定 |
| 3 | year | 0.1044 | 0.0086 | 0.0824 | 0.0304 | 高度稳定 |
| 4 | Age | 0.0310 | 0.0040 | 0.1294 | 0.0148 | 中等稳定 |
| 5 | Raca.Color | 0.0288 | 0.0049 | 0.1712 | 0.0168 | 中等稳定 |
| 6 | Code.Profession | 0.0172 | 0.0034 | 0.2000 | 0.0128 | 不稳定 |

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| Bootstrap 次数够吗？ | 30 次为标准推荐，100-200 次用于正式论文 | 观察累计均值是否收敛（SEM < 0.01|mean） |
| CV 对均值敏感？ | 均值低 → CV 自动升高 | 同时看 Mean 和 CV |
| 排名稳定性 ρ=0.41 很低？ | 6 个特征的排名空间小，Spearman 可能不敏感 | 报告完整排名分布而非仅均值 |
| Bootstrap 只引入采样变异？ | 不涵盖模型超参数等不确定性来源 | 结合模型不确定性（随机种子、超参数变化） |
| 计算成本高？ | 30×训练+SHAP = 30 倍成本 | 减少 n_estimators、n_shap；或使用 Sub-sampling |

---

## 与其他模块的联系

- **前置模块**：Module 12 SHAP 概述（单次 SHAP 重要性为点估计）、Module 8 交叉验证（CV 评估模型性能稳定性）、Module 15 SHAP 聚类（Code.Profession 在 Cluster 2 中 rel_imp=1.20x——局部稳定但全局不稳定）
- **后续模块**：本模块为 Module 12-16 所有结论的可靠性提供了量化评估框架
- **与研究工作的联系**：在 236K 化合物虚拟筛选中，Bootstrap 稳定性分析可评估分子描述符对 docking 评分重要性结论的可靠性——哪些描述符的重要性在化合物子集扰动下保持稳定？在宏基因组研究中，Bootstrap 可验证菌群属重要性结论是否受样本组成变化影响

---

## 参考资料

- 教程原文：`ml4health-main/jupyter/17_shap_stability_bootstrap.ipynb`
- 讲义：`ml4health-main/lectures/17_shap_stability_bootstrap_teaching_doc.md`
- Lundberg SM, Lee SI. A unified approach to interpreting model predictions. *NeurIPS* 2017.
- Efron B, Tibshirani RJ. An introduction to the bootstrap. *Chapman & Hall/CRC* 1994.
- Meinshausen N, Bühlmann P. Stability selection. *J R Stat Soc B* 2010.
