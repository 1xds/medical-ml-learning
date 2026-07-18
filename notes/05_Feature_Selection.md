# Module 05 笔记: 特征选择

> 本模块的核心命题：**从六个不同角度对特征进行层层筛选，在保留预测能力的前提下最小化特征数量，验证"降维不降质"的可行性。**

---

## 核心概念梳理

### 特征选择是什么？

特征选择是从全部候选特征中选取最优子集的过程，其目的不是简单地减少特征数量，而是在信息保留与模型简化之间找到平衡。当特征数量超过信息量需求时，冗余特征不仅增加计算成本，还可能引入噪声导致过拟合。特征选择遵循奥卡姆剃刀原则：**在性能相近的前提下，更简单的模型（更少的特征）具有更好的泛化能力。**

医学 AI 研究尤其重视特征选择的严谨性。临床预测模型需要可解释性——医生需要理解"为什么"某个患者被判定为高风险。因此，特征选择方法本身需要具有统计严谨性和可复现性。

### 六层特征选择体系

```
相关性分析 → VIF → Filter(ANOVA+MI) → LASSO → RF Importance → Boruta
```

| 层 | 方法 | 类型 | 核心视角 | 计算成本 |
|---|------|------|---------|---------|
| 1 | 相关性分析 | 无监督 | 两两线性关系，删除 |r| > 0.8 的对 | 极低 |
| 2 | VIF | 无监督 | 一个特征 vs 所有其他特征的多重共线性 | 低 |
| 3 | Filter (ANOVA/MI) | 单变量 | 特征与目标的线性/信息论关联 | 低 |
| 4 | LASSO | 嵌入法 | L1 正则化路径，系数逐步压缩至零 | 中 |
| 5 | RF Importance | 嵌入法 | 树模型节点分裂的 Gini 重要性 | 中 |
| 6 | Boruta | 包装法 | 真实特征与随机阴影变量竞争，二项检验 | 高 |

### 为什么不同方法给出的"最佳特征集"不同？

| 方法 | 认为最重要的特征 | 原因 |
|------|----------------|------|
| ANOVA | Diagnostic.means | 线性视角下，该特征对两组均值差异最大 |
| MI | Code.of.Morphology | 信息论视角下，该特征与目标共享信息最多 |
| LASSO | Diagnostic.means | 正则化路径上，该特征贡献最稳定 |
| RF | Diagnostic.means（重要性 0.32） | 树模型视角下，该特征最常用于节点分裂 |
| Boruta | 11 个 Confirmed | 统计检验视角下，11 个特征显著优于随机 |

---

## 代码精读

### Block 1: 数据准备与特征构造

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LassoCV, Lasso
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy                        # Boruta 算法

df = pd.read_csv(DATA_PATH, low_memory=False)       # 读取数据
df['target'] = df['Status.Vital'].map({'VIVO': 1, 'MORTO': 0})

# 8 个基础特征 + 4 个构造特征
base_feat = ['Age', 'year', 'Gender', 'Code.Profession', 'Code.of.Morphology',
             'Diagnostic.means', 'Extension', 'Raca.Color']

df_feat['Age_Group'] = df_feat['Age'].apply(age_group)    # 年龄分组
df_feat['Age_Sq'] = df_feat['Age'] ** 2                    # 平方项
df_feat['Year_From_2000'] = df_feat['year'] - 2000         # 连续趋势
df_feat['Is_Child'] = (df_feat['Age'] < 18).astype(float)  # 儿童标志

# 插补 + 标准化
imputer = SimpleImputer(strategy='mean')
X_scaled = scaler.fit_transform(imputer.fit_transform(X_full))
```

要点：
1. N_SAMPLES = 50,000（Boruta 计算量大，适度采样）
2. 12 个特征包含 8 个原始特征和 4 个领域构造特征

### Block 2: 第一层——相关性分析

```python
corr_matrix = pd.DataFrame(X_imp, columns=all_features).corr()  # 计算相关系数矩阵

high_corr_pairs = []
removed_by_corr = set()
for i in range(n_feat):
    for j in range(i+1, n_feat):
        if abs(corr_matrix.iloc[i, j]) > 0.8:       # 阈值 > 0.8
            high_corr_pairs.append((all_features[i], all_features[j], corr_matrix.iloc[i, j]))
            # 保留方差较小者（策略：删除冗余信息更多的一方）
            if var1 >= var2:
                removed_by_corr.add(f2)
            else:
                removed_by_corr.add(f1)
features_after_corr = [f for f in all_features if f not in removed_by_corr]
```

要点：
1. 高度相关对：Age_Group ↔ Age（由 Age 直接派生），Age ↔ Age_Sq（r≈0.98）
2. 移除策略：保留方差较小的特征以最小化信息损失
3. 相关性分析结果：9/12 特征保留，移除 3 个

### Block 3: 第二层——VIF 多重共线性诊断

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

X_vif_sm = sm.add_constant(X_vif)                          # 添加截距项
for col in remaining:
    vif_val = variance_inflation_factor(X_sub_sm.values, i + 1)  # 计算每个特征的VIF

# 迭代删除 VIF > 10 的特征
if max_vif[1] > 10:
    remaining.remove(max_vif[0])
    print(f"移除 {max_vif[0]} (VIF = {max_vif[1]:.2f} > 10)")
else:
    print(f"所有特征 VIF ≤ 10 (最高 VIF = {max_vif[1]:.2f})")
    break
```

要点：
1. VIF = 1/(1-R²)，衡量某个特征能被其他特征线性预测的程度
2. 临界值：VIF > 5 为中度共线性，VIF > 10 为严重共线性
3. 本数据集所有特征 VIF ≤ 1.64，无多重共线性问题——VIF 层无进一步移除

### Block 4: 第三层——Filter 方法（ANOVA + 互信息）

```python
# ANOVA F-test
anova_selector = SelectKBest(f_classif, k='all')
anova_selector.fit(X_train_s, y_train)
anova_scores = anova_selector.scores_

# Mutual Information（信息论视角）
mi_scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)

# 综合排名：取两种方法的平均排名
filter_df['Avg_Rank'] = (np.argsort(np.argsort(-anova_scores)) +
                          np.argsort(np.argsort(-mi_scores))) / 2 + 1

# 选取 Top 60%
top_filter_features = filter_df.head(int(np.ceil(n_feat * 0.6)))['Feature'].tolist()
```

要点：
1. ANOVA F-test 度量的是特征在不同目标类之间的均值差异（线性视角）
2. 互信息度量的是特征与目标之间的"共享信息量"（非线性视角）
3. 综合排名策略：两种方法平均排名，避免单一视角的偏倚
4. Filter 层结果：8/12 特征保留

### Block 5: 第四层——LASSO（L1 正则化）

```python
from sklearn.linear_model import LassoCV, Lasso

# 5 折交叉验证选择最优 alpha
lasso_cv = LassoCV(cv=5, max_iter=10000, alphas=np.logspace(-4, 0, 50))
lasso_cv.fit(X_train_s, y_train)
best_alpha = lasso_cv.alpha_

# 用最优 alpha 重新训练
lasso = Lasso(alpha=best_alpha, max_iter=10000)
lasso.fit(X_train_s, y_train)

# 系数为零 = 被淘汰；系数非零 = 保留
lasso_zero = lasso_coef[lasso_coef['Coefficient'] == 0]
lasso_nonzero = lasso_coef[lasso_coef['Coefficient'] != 0]
```

要点：
1. LASSO 的 L1 正则化将不重要特征的系数逐步压缩至零
2. 正则化路径图展示：α 增大时，最不重要特征先被淘汰，最重要特征最后淘汰
3. 最优 α 处仅 Age 被压缩至 0（11/12 保留）
4. LASSO 不仅做选择，也是重要性排序器——被最后淘汰的特征最重要

### Block 6: 第五层——Random Forest 重要性

```python
rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)

rf_importance = pd.DataFrame({
    'Feature': all_features,
    'Importance': rf.feature_importances_       # Gini 重要性
}).sort_values('Importance', ascending=False)

# 累积重要性达 90% 的特征
for _, row in rf_importance.iterrows():
    cumulative += row['Importance']
    rf_top_features.append(row['Feature'])
    if cumulative >= 0.90: break
```

要点：
1. 使用 200 棵树、最大深度 10（控制过拟合）
2. `class_weight='balanced'` 保持与逻辑回归一致的类别处理
3. 前 6 个特征累积重要性已达 91.8%——存在显著的长尾效应
4. RF 层结果：6/12 特征保留（最保守的选择）

### Block 7: 第六层——Boruta

```python
from boruta import BorutaPy

rf_boruta = RandomForestClassifier(n_estimators=100, max_depth=8,
                                    class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
boruta = BorutaPy(rf_boruta, n_estimators='auto', perc=100,
                  alpha=0.05, random_state=RANDOM_STATE, max_iter=20)
boruta.fit(X_train, y_train)

# 三类结果
confirmed = boruta_result[boruta_result['Support']]          # 显著优于随机
tentative = boruta_result[~boruta_result['Support'] & boruta_result['Support_Weak']]  # 无法确定
rejected = boruta_result[~boruta_result['Support'] & ~boruta_result['Support_Weak']] # 不优于随机
```

要点：
1. Boruta 的核心机制：创建随机阴影特征（shuffle 副本），让真实特征与阴影特征竞争
2. 统计检验：多次迭代后，比较真实特征 vs 阴影特征最大重要性的分布
3. α=0.05：显著性水平为 5%
4. 结果：11 Confirmed, 0 Tentative, 1 Rejected（Is_Child）
5. Boruta 的统计严谨性使其在医学论文中越来越流行——不依赖人工阈值，给出正式统计结论

### Block 8: 逻辑回归验证

```python
feature_sets = {
    'All Features': all_features,           # 12 特征
    'Correlation': features_after_corr,     # 9 特征
    'Filter(ANOVA+MI)': top_filter_features, # 8 特征
    'LASSO': features_lasso,                # 11 特征
    'RF Importance': features_rf,           # 6 特征
    'Boruta': features_boruta,              # 11 特征
}

for name, feat_list in feature_sets.items():
    idxs = [all_features.index(f) for f in feat_list]
    lr = LogisticRegression(class_weight='balanced', max_iter=5000)
    lr.fit(X_train_s[:, idxs], y_train)
    auc = roc_auc_score(y_test, lr.predict_proba(X_test_s[:, idxs])[:, 1])
```

**验证结果**：

| 特征集 | 特征数 | AUC | Recall | Brier |
|--------|-------|-----|--------|-------|
| All Features | 12 | 0.8980 | 0.8874 | 0.1281 |
| Correlation | 9 | 0.8980 | 0.8871 | 0.1281 |
| Filter | 8 | 0.8926 | 0.8957 | 0.1310 |
| **RF Importance** | **6** | **0.8883** | **0.8996** | **0.1342** |
| **Boruta** | **11** | **0.8979** | **0.8872** | **0.1281** |

---

## 关键收获

1. **特征选择不等于牺牲性能**：用 9 个特征（Correlation）可达到与 12 个全特征相同的 AUC（0.8980），验证了信息冗余的存在。前 6 个特征的 RF 累积重要性已达 91.8%。

2. **不同方法从不同角度定义"重要性"**：相关性关注线性共变，VIF 关注多重共线性，Filter 关注单变量关联，LASSO 关注正则化路径，RF 关注树模型贡献，Boruta 关注统计显著性。推荐多种方法交叉验证，关注所有方法都认可的特征（交集）。

3. **VIF 是相关性的多维扩展**：即使两两相关性不高，一个特征也可能由其他多个特征的线性组合所预测。VIF = 1/(1-R²) 捕捉了"一个 vs 所有"的关系，比"一对一"的相关性分析更全面。

4. **LASSO 的路径图是重要性排序器**：正则化强度 α 从 0 增大时，不重要特征先被压缩至零。被最后淘汰的特征即最重要的特征。LASSO 在本数据中只淘汰了 Age（系数被压缩至零），说明在给定 α 下 11/12 特征均有独立贡献。

5. **Boruta 的统计框架最严谨**：通过阴影特征构建零分布，对每个特征进行形式化的二项检验。α=0.05 的显著性水平、内置的多重比较校正、不依赖人工阈值——这些特性使其在医学论文中的使用频率逐步增长。

6. **特征选择提高泛化能力**：更少的特征意味着更简单的模型，更低的过拟合风险。RF 选择的 6 个特征在 AUC 仅下降约 1% 的情况下将特征数减半。

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| Filter 排名和 RF 重要性不一致 | 线性方法 vs 树模型视角不同 | 多方法验证，取交集或加权投票 |
| LASSO 只淘汰 1 个特征 | 12 个特征均有独立贡献，正则化强度不够 | 增大 α 或使用更严格的正则化策略 |
| Boruta 的 Rejected 特征实际有意义 | Boruta 对弱信号不敏感，可能漏掉微弱但真实的特征 | 检查 Tentative 特征，结合领域知识判断 |
| 不同方法选择结果不一致 | 每种方法有自己的"重要性"定义 | 关注方法间的共识（交集），结合领域知识 |
| VIF 迭代删除后特征数太少 | 高维数据集普遍存在多重共线性 | 使用 Ridge 回归替代删除，或使用 PCA 降维 |

---

## 与其他模块的联系

- **前置模块**：Module 04（特征工程）构造的 Age_Group 和 Age_Sq 在本模块中因高度共线性被相关性分析层移除，验证了"构造特征需后续筛选"的工作流；Module 02（统计检验）发现所有特征均显著但效应量差异大——本模块在此基础上做二次筛选
- **后续模块**：Module 06（降维与聚类）将通过 PCA 特征变换处理冗余，与本模块的特征选择（保留子集）形成方法对比


---

## 参考资料

- 教程原文：`ml4health-main/jupyter/05_feature_selection.ipynb`
- 讲义：`ml4health-main/lectures/05_feature_selection_teaching_doc.md`
- Kursa, M. B. & Rudnicki, W. R. (2010). Feature Selection with the Boruta Package. *Journal of Statistical Software*, 36(11).
- Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *JRSS-B*, 58(1), 267-288.
- Guyon, I. & Elisseeff, A. (2003). An Introduction to Variable and Feature Selection. *JMLR*, 3, 1157-1182.
