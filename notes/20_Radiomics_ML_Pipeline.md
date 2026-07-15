# Module 20 笔记: 影像组学端到端机器学习流水线

> 本模块的核心命题：**在 Module 19 提取的 93 维影像组学特征上，跑通从 EDA、统计检验、LASSO 特征选择、交叉验证、多模型对比到校准分析和 SHAP 可解释性的完整 ML 流水线，验证"流程骨架完全通用，仅前端需特征化"的迁移法则。**

---

## 核心概念梳理

### 影像组学 ML 流水线概览

本模块复用 Module 01-12 建立的通用 ML 流程骨架（EDA → 统计 → 预处理 → 特征选择 → CV → 建模 → 校准 → SHAP），迁移到影像组学特征表上。核心教学价值在于回答一个问题：**从表格数据切换到影像数据后，哪些步骤保持不变，哪些步骤因数据特性需要调整？**

| 步骤 | 表格数据 (Module 01-12) | 影像组学 (本模块) | 差异 |
|------|----------------------|-----------------|------|
| 数据加载 | `pd.read_csv` 直接读取 | Module 19 已完成特征化 | 前端不同 |
| EDA | 缺失值/分布/离群检查 | 特征量级跨度极大 (13个数量级)、相关性高度冗余 | 关注点不同 |
| 统计检验 | T检验/卡方 | Mann-Whitney U + BH-FDR校正 | 影像特征偏态，用非参数 |
| 预处理 | 均值插补/标准化 | StandardScaler 必须 | 标准化是必选项 |
| 特征选择 | 相关性→VIF→LASSO→Boruta | LASSO 是影像组学事实标准 | 方法子集相同 |
| 防泄漏 | Pipeline + 训练集内 fit | 完全相同 | 通用原则 |
| 交叉验证 | K-Fold/重复/嵌套 | 重复分层 5-Fold×10 | 完全相同 |
| 建模 | LR/SVM/RF/XGB/LGBM | 相同模型池 | 完全相同 |
| 校准 DCA | 校准曲线/Brier/DCA | 相同方法 | 完全相同 |
| SHAP | 蜂群/依赖/瀑布 | 相同可视化 | 完全相同 |

### 影像组学 Signature

影像组学 Signature 是本模块的核心产出——通过 LASSO (L1 正则化逻辑回归) 从高维特征空间中稀疏选择出一组具有区分度且可解释的定量特征。这一范式由 Aerts 等人在 *Nature Communications* (2014) 中确立，引用超过 5000 次，已成为影像组学领域的标准方法论。

LASSO 在本场景中具备三重优势：
1. **自动稀疏**：L1 惩罚将冗余特征的系数精确压缩至零，实现内嵌式特征选择
2. **线性可解释**：保留系数符号和大小，每个入选特征可直接解释为风险/保护因子
3. **高维适应性**：在 p ≫ n (93特征 vs 80训练样本) 条件下仍稳定收敛

### 本案例的标准特征选择流程

1. **零方差/低方差过滤**：本数据无零方差特征（跳过）
2. **相关性去冗余**：|r| > 0.9 的特征对中任选一个保留，93 → 39 特征
3. **LASSO 稀疏化**：L1 逻辑回归 + 5-fold CV 选正则强度 C，39 → 13 个 signature 特征
4. **后续建模仅在 13 个 signature 特征上进行**

---

## 代码精读

### Block 1: 环境配置与数据加载

```python
from sklearn.model_selection import (train_test_split, RepeatedStratifiedKFold,
                                     cross_val_score, StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV         # LASSO CV
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                             accuracy_score, recall_score, precision_score, f1_score)
import xgboost as xgb
import lightgbm as lgb
import shap
```

```python
df = pd.read_csv(FEATURES_CSV)                 # 读取 Module 19 输出的特征表
y = df['Contrast'].values                      # 目标: 增强=1, 平扫=0
non_feat_cols = ['id', 'Contrast']
feat_cols = [c for c in df.columns if c not in non_feat_cols]
X = df[feat_cols].astype(float).values         # 特征: Age + 93 影像组学特征
```

**要点**：特征列为 Age（临床变量）和 93 个影像组学特征，共计 94 维。目标变量 50/50 完全平衡。

### Block 2: 防泄漏第 1 步 — 先划分数据集

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
# 训练集 80 样本, 测试集 20 样本, 分层划分保留类别比例
```

**要点**：这是防泄漏原则的第一道防线——数据集中划分必须在任何 fit 操作之前完成，确保特征选择、标准化的统计量全部来自训练集。

### Block 3: EDA — 特征量级跨度与相关性冗余

```python
# 特征量级跨度 (影像组学特有问题)
ranges = pd.Series(X.max(axis=0) - X.min(axis=0), index=feat_names)
print(f"跨度跨越 {np.log10(ranges.max()/(ranges.min()+1e-9)):.0f} 个数量级")

# 相关性冗余检测
corr = df[feat_cols].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr_pairs = upper.stack()[upper.stack() > 0.9]  # |r|>0.9 的高相关对
# 相关性热力图: 取 Top 20 方差特征
top_var = pd.Series(X.std(axis=0), index=feat_names).sort_values(ascending=False).head(20).index
sns.heatmap(df[top_var].corr(), cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True)
```

**要点**：影像组学特征跨 13 个数量级的特性（一阶统计均值 ~10² vs 高阶纹理 ~10⁻⁵）使得标准化不是可选优化，而是必须预处理。相关性热力图直观展示了特征的高度冗余——大量特征对 |r| > 0.9 说明同一图像属性被多个特征重复表征。

### Block 4: 统计分析 — Mann-Whitney U + BH-FDR

```python
stat_rows = []
for j, name in enumerate(feat_names):
    g1 = X[y == 1, j]   # 增强组
    g0 = X[y == 0, j]   # 平扫组
    u, p = stats.mannwhitneyu(g1, g0, alternative='two-sided')  # 非参数检验
    r = 1 - 2 * u / (len(g1) * len(g0))  # 效应量: rank-biserial correlation
    stat_rows.append({'feature': name, 'p_value': p, 'effect_r': abs(r)})

stat_df = pd.DataFrame(stat_rows)
# BH-FDR 校正: 从尾部累计最小保证单调性
pvals = stat_df['p_value'].values
order = np.argsort(pvals)
ranked = pvals[order]
m = len(pvals)
q = ranked * m / (np.arange(m) + 1)
fdr = np.minimum.accumulate(q[::-1])[::-1]
stat_df['fdr_bh'] = np.clip(fdr_sorted, 0, 1)
```

**要点**：使用 Mann-Whitney U 检验而非 T 检验是影像组学统计的关键决策——特征不服从正态分布，秩和检验的分布自由性质在此场景下更可靠。BH-FDR 多重比较校正控制了假发现率，`np.minimum.accumulate` 确保校正后 p 值的单调性。

### Block 5: 预处理 + LASSO 特征选择 (Signature 构建)

```python
# 防泄漏: 标准化仅用训练集的均值和方差
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# 第2步: 相关性去冗余 (|r|>0.9 剔除 — 在训练集上)
corr_train = pd.DataFrame(X_train_sc, columns=feat_names).corr().abs()
upper_t = corr_train.where(np.triu(np.ones(corr_train.shape), k=1).astype(bool))
to_drop = set()
for col in upper_t.columns:
    if col in to_drop:
        continue
    high = upper_t.index[upper_t[col] > 0.9].tolist()
    to_drop.update(high)
keep_after_corr = [c for c in feat_names if c not in to_drop]

# 第3步: LASSO (L1, 5-fold CV 选 C) — 影像组学 signature 核心
X_train_lasso = X_train_sc[:, keep_idx]
lasso = LogisticRegressionCV(
    Cs=np.logspace(-3, 2, 30),        # C 在 10^-3 ~ 10^2 对数网格搜索
    penalty='l1', solver='liblinear',  # L1 正则, liblinear 支持 L1
    cv=5, scoring='roc_auc',          # 5-fold CV, 最大化 AUC
    max_iter=5000, random_state=RANDOM_STATE, n_jobs=-1)
lasso.fit(X_train_lasso, y_train)
best_C = lasso.C_[0]                  # 最佳正则强度
coef = lasso.coef_.ravel()
sig_mask = coef != 0                  # 非零系数 = 入选特征
signature_feats = np.array(keep_after_corr)[sig_mask]

# 提取 signature 特征子集用于后续建模
X_train_sig = X_train_sc[:, sig_idx]   # 训练 signature
X_test_sig = X_test_sc[:, sig_idx]     # 测试 signature
```

**要点**：LogisticRegressionCV 在 30 个对数等距的 C 值上做 5-fold CV，以 ROC-AUC 为评分准则选择最佳正则强度。L1 惩罚自动将冗余特征的系数压缩至零，结果为 13 个非零特征的 signature。防泄漏贯穿始终——相关性去冗余和 LASSO 的 fit 全部在训练集上完成。

### Block 6: 交叉验证与多模型对比

```python
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)

models = {
    'LogReg':  LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    'SVM':     SVC(probability=True, random_state=RANDOM_STATE),
    'RF':      RandomForestClassifier(n_estimators=200, max_depth=8,
                                      random_state=RANDOM_STATE, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                 eval_metric='logloss', random_state=RANDOM_STATE,
                                 n_jobs=-1, verbosity=0),
    'LightGBM': lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                   random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
}

# CV 评估
for name, model in models.items():
    auc_scores = cross_val_score(model, X_train_sig, y_train, cv=cv, scoring='roc_auc')
    cv_results[name] = {'auc': auc_scores}

# 测试集评估
for name, model in models.items():
    model.fit(X_train_sig, y_train)
    prob = model.predict_proba(X_test_sig)[:, 1]           # 正类概率
    pred = (prob >= 0.5).astype(int)                        # 0.5 阈值分类
    test_results[name] = {
        'auc': roc_auc_score(y_test, prob),
        'acc': accuracy_score(y_test, pred),
        'recall': recall_score(y_test, pred),
        'f1': f1_score(y_test, pred),
        'brier': brier_score_loss(y_test, prob),
    }
```

**要点**：模型池覆盖线性 (LogReg)、核方法 (SVM)、树集成 (RF/XGBoost/LightGBM) 三种范式，在小样本 (80 训练) 情境下 LR 和 SVM 通常表现最优（更少的待估参数）。RepeatedStratifiedKFold (5×10) 通过多次重复消除了单次随机划分的偶然性。

### Block 7: 类别不平衡检查

```python
ratio = n_pos / n_neg  # 50/50 = 1.0
print("数据完全平衡 → 无需 SMOTE/重采样/class_weight")
```

**要点**：本数据集标签平衡，但代码中保留检查逻辑——在实际影像组学任务（如肿瘤检出，阳性率常不足 5%）中，需启用 Module 10 的 SMOTE、class_weight 或阈值调整等方法。

### Block 8: 校准分析与决策曲线 DCA

```python
# 校准曲线
frac_pos, mean_pred = calibration_curve(y_test, best_prob, n_bins=5)
axes[0].plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
axes[0].plot(mean_pred, frac_pos, 's-', label=best_name)

# DCA 决策曲线: 净获益 = (TP - FP * pt/(1-pt)) / N
def decision_curve(y_true, prob, pt):
    tp = np.sum((prob >= pt) & (y_true == 1))
    fp = np.sum((prob >= pt) & (y_true == 0))
    n = len(y_true)
    return tp / n - fp * (pt / (1 - pt)) / n

pts = np.linspace(0.01, 0.99, 99)
net_benefit = [decision_curve(y_test, best_prob, pt) for pt in pts]
treat_all = [(y_test.mean() - (1 - y_test.mean()) * pt / (1 - pt)) for pt in pts]
```

**要点**：Brier Score 评估概率校准质量（越小越好，0 为完美）。DCA 的净获益公式中 `pt/(1-pt)` 为决策阈值对应的 odds，`tp/n` 为真阳性收益，`fp * odds / n` 为假阳性"代价"加权——当净获益超过"全部处理"和"全部不处理"时，模型具有临床决策价值。

### Block 9: SHAP 可解释性分析

```python
# 优先使用树模型的 TreeExplainer (高效精确)
shap_model = models['XGBoost']
shap_model.fit(X_train_sig, y_train)
explainer = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(X_test_sig)

# 蜂群图: 全局特征贡献
shap.summary_plot(sv_plot, X_test_sig, feature_names=short_names, max_display=len(signature_feats))

# 柱状图: 特征重要性排序 (|SHAP| 均值)
shap.summary_plot(sv_plot, X_test_sig, feature_names=short_names, plot_type='bar')

# 瀑布图: 单样本局部解释
shap.waterfall_plot(shap.Explanation(values=sv_plot[0],
    base_values=explainer.expected_value, data=X_test_sig[0],
    feature_names=short_names), max_display=len(signature_feats))
```

**要点**：三种 SHAP 可视化覆盖三个解释层次——蜂群图（全局：各特征的贡献分布和方向）、柱状图（全局：特征重要性排序）、瀑布图（局部：单样本的加性贡献分解）。**关键教学发现**：SHAP 指向的一阶密度统计（firstorder_RobustMAD, 90Percentile）为最强预测因子，这直接体现造影剂提升组织密度的物理机制——模型学到了物理上可解释的特征。

---

## 关键收获

1. **ML 流程骨架的通用性得到验证**：从表格数据到影像数据，10 步标准流程中 7 步完全相同、3 步仅需微调，证明了"先掌握流程骨架，再迁移到新模态"的学习路径的有效性。

2. **LASSO 在高维小样本下的不可替代性**：93 维特征仅 80 训练样本构成了典型的 p > n 场景。相关性过滤 + LASSO 的两阶段策略将特征维度压缩至 13，既保留了判别信息又避免了过拟合。这一策略直接源于影像组学领域的最佳实践（Aerts et al., 2014）。

3. **防泄漏是贯穿全流程的铁律**：数据划分 → 标准化 → 相关性去冗余 → LASSO 选择 → 交叉验证，每一步的 fit 操作严格限定在训练集内。任何对测试集信息的"偷窥"都会导致过于乐观的评估。

4. **特征可解释性的物理验证**：SHAP 分析显示增强/平扫 CT 的最强区分特征恰好是"密度类"的一阶统计量，这与"碘造影剂提升组织密度"的物理事实一致——模型学到的不是伪相关，而是物理可解释的生物物理信号。

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| CV AUC 高但测试 AUC 低 | 过拟合或数据泄露 | 检查特征选择是否在切分前执行；减小模型复杂度 |
| LR 优于 RF/XGBoost | 小样本下简单模型泛化更好 | 预期之内；可增加树模型正则化 (`max_depth`, `min_samples_leaf`) |
| LASSO 选 0 个特征 | C 参数过大（惩罚过强） | 增大 C 搜索范围的上界（如 `np.logspace(-3, 3, 30)`） |
| SHAP waterfall_plot 报错 | shap 版本兼容性 | 检查 Explanation 对象构造方式；必要时降级到 `shap.waterfall_plot(explainer.expected_value, sv_plot[0], ...)` |
| 相关性热力图过于密集 | 93 维全量无法可视化 | 限制为 Top 20 方差特征，或使用聚类热力图 |

---

## 与其他模块的联系

- **前置模块**：Module 19 (影像组学特征提取) — 直接读取其输出的 `radiomics_features.csv`；Module 01-12 (表格数据 ML) — 完全复用其 10 步标准流程
- **后续模块**：Module 21 (TCGA 基因多分类) — 同样演示"新模态 + 完整 ML 流程"的迁移能力，LASSO 特征选择策略可对照
- **与研究工作的联系**：影像组学 Signature 的构建方法论可直接迁移至分子影像学场景——若在 EcMurJ 研究中引入电镜图像或分子表面图像，可沿用本模块的"特征提取 + LASSO 稀疏 + SHAP 验证"三板斧。在宏基因组研究中，微生物群落的共丰度矩阵可类比为"影像"，纹理类特征可捕捉群落结构的空间关联模式。

---

## 参考资料

- 教程原文：`ml4health-main/jupyter/20_radiomics_ml_pipeline.ipynb`
- 讲义：`ml4health-main/lectures/19_20_radiomics_case_study_teaching_doc.md`
- Aerts, H. J., et al. (2014). "Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach." *Nature Communications*, 5, 4006.
- Lambin, P., et al. (2017). "Radiomics: the bridge between medical imaging and personalized medicine." *Nature Reviews Clinical Oncology*, 14(12), 749-762.
- Vickers, A. J., & Elkin, E. B. (2006). "Decision curve analysis: a novel method for evaluating prediction models." *Medical Decision Making*, 26(6), 565-574.
- Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*, 4765-4774.
