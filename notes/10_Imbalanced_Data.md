# Module 10 笔记: 类别不平衡问题与样本重采样

> 本模块的核心命题：**在类别严重不平衡的医学数据中，高准确率并不等同于好模型；重采样的核心作用是改变决策阈值而非提升排序能力，且 SMOTE 泄漏是医学论文中最常见且最危险的错误之一。**

---

## 核心概念梳理

### 类别不平衡是什么？

类别不平衡（Class Imbalance）是指数据集中各类别的样本数量存在显著差异的现象。在医学场景中，这是极为常见的数据特征：罕见病诊断、癌症筛查、药物不良反应检测等场景下，阳性样本（少数类）通常远少于阴性样本（多数类）。本模块通过欠采样少数类 VIVO（存活）构造了 Imbalance Ratio（IR）= 10:1 的严重不平衡数据集（MORTO 11,770 例 vs VIVO 1,177 例），以充分展示重采样策略的差异。

### 不平衡程度的分级

| IR 范围 | 不平衡程度 | 处理建议 |
|---------|-----------|---------|
| IR < 2 | 轻微 | 可不处理，使用 class_weight='balanced' 即可 |
| 2 ≤ IR < 10 | 中等 | 建议重采样 |
| 10 ≤ IR < 50 | 严重 | 必须重采样 |
| IR ≥ 50 | 极严重 | 需考虑异常检测思路（如 Isolation Forest） |

### 关键概念

| 概念 | 说明 | 与不平衡数据的关系 |
|------|------|-------------------|
| 混淆矩阵 | TP / TN / FP / FN | 理解不平衡评价的基础 |
| Accuracy | (TP+TN)/N | 在不平衡数据中严重误导 |
| Recall | TP/(TP+FN) | 医学场景的核心指标（漏诊率） |
| Precision | TP/(TP+FP) | 阳性预测值（误诊率） |
| PR Curve | Precision vs Recall | 不平衡数据下比 ROC 更敏感 |
| ROC-AUC | 排序能力 | 偏高，被多数类贡献稀释 |
| PR-AUC | 少数类排序能力 | 对少数类数量敏感，更诚实 |

---

## 代码精读

### Block 1: 构造不平衡数据集

```python
# 原始数据 IR≈1.4:1 (轻度不平衡), 重采样效果不明显
# 通过欠采样少数类 VIVO 构造 IR=10:1 (严重不平衡)
TARGET_IR = 10                                                      # 目标不平衡比

df_neg = df[df['target'] == 0].copy()                              # 多数类全部保留
df_pos = df[df['target'] == 1].copy()                              # 少数类
n_pos_target = len(df_neg) // TARGET_IR                            # 计算目标数量
df_pos_sampled = df_pos.sample(n=n_pos_target, random_state=RANDOM_STATE)  # 欠采样
df = pd.concat([df_neg, df_pos_sampled]).reset_index(drop=True)   # 合并

# 结果: MORTO 11,770 (90.91%) vs VIVO 1,177 (9.09%), IR=10:1
```

**要点说明：** 原始数据为轻度不平衡（IR≈1.4:1），重采样效果不明显。为充分展示各策略的差异，通过欠采样少数类构造 IR=10:1 的严重不平衡场景。总样本数 12,947。

### Block 2: Accuracy Paradox（准确率悖论）

```python
# 模型 A: 总是预测多数类 (MORTO)
y_pred_always_dead = np.zeros_like(y_te)                           # 全部预测为0(死亡)
acc_always_dead = accuracy_score(y_te, y_pred_always_dead)         # Accuracy=0.9091
rec_always_dead = recall_score(y_te, y_pred_always_dead, pos_label=1)  # Recall=0

# 模型 B: 加权 Logistic Regression
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', LogisticRegression(class_weight='balanced',         # 平衡权重
                                  max_iter=5000, random_state=RANDOM_STATE))
])
pipe.fit(X_tr, y_tr)
y_prob_lr = pipe.predict_proba(X_te)[:, 1]
y_pred_lr = (y_prob_lr >= 0.5).astype(int)
```

**对比结果：**

| 指标 | "全预测死亡"模型 | 加权 Logistic Regression |
|------|----------------|------------------------|
| Accuracy | 0.9091 | 0.7614 |
| Recall | 0.0000 | 0.8782 |
| Precision | 0.0000 | 0.2596 |
| F1 | 0.0000 | 0.4008 |
| ROC-AUC | 0.5000 | 0.8974 |

**核心教学点：** "全预测死亡"模型的 Accuracy 高达 90.91%，但 Recall 为零，在临床上毫无价值。Accuracy 在不平衡数据中是一个危险的指标。

### Block 3: 四种重采样策略对比

```python
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

resamplers = {
    'No Resampling': None,                                          # 不重采样(基线)
    'Random UnderSampling': RandomUnderSampler(random_state=RANDOM_STATE),  # 欠采样
    'Random OverSampling': RandomOverSampler(random_state=RANDOM_STATE),    # 过采样
    'SMOTE': SMOTE(random_state=RANDOM_STATE),                      # 合成少数类过采样
    'ADASYN': ADASYN(random_state=RANDOM_STATE),                    # 自适应合成采样
}

for name, sampler in resamplers.items():
    # 先插补 (所有方法共享相同的 imputation)
    X_tr_imp = imp.fit_transform(X_tr)                              # 训练集插补
    X_te_imp = imp.transform(X_te)                                  # 测试集变换

    if sampler is None:
        X_tr_r, y_tr_r = X_tr_imp.copy(), y_tr.copy()              # 不重采样
    else:
        X_tr_r, y_tr_r = sampler.fit_resample(X_tr_imp, y_tr)     # 仅在训练集重采样

    lr = LogisticRegression(class_weight=None, max_iter=5000)      # 不加权(展示重采样效果)
    lr.fit(X_tr_r, y_tr_r)
    y_prob = lr.predict_proba(X_te_imp)[:, 1]                      # 在原始测试集评估
```

**实验结果：**

| 方法 | AUC | Recall | Precision | F1 | 训练样本数 |
|------|-----|--------|-----------|-----|-----------|
| No Resampling | 0.9007 | 0.1813 | 0.6214 | 0.2807 | 9,062 |
| Random UnderSampling | 0.8971 | 0.8810 | 0.2620 | 0.4039 | 1,648 |
| Random OverSampling | 0.8976 | 0.8754 | 0.2619 | 0.4031 | 16,476 |
| SMOTE | 0.8976 | 0.8640 | 0.2694 | 0.4108 | 16,476 |
| ADASYN | 0.8947 | 0.8952 | 0.2555 | 0.3975 | 16,494 |

**关键发现：**

1. **重采样不提升 AUC（排序能力）**：所有方法 AUC 在 0.8947-0.9007 之间，差异 ≤ 0.006。重采样改变的是决策阈值（Recall 从 0.18 提升至 0.90），而非排序能力。
2. **No Resampling 的 Recall 极低（0.18）**：模型几乎无法检出少数类，因为训练集中正类仅占 9.1%。
3. **SMOTE 的 F1 最高（0.4108）**：在 Recall 和 Precision 之间取得了最佳平衡。
4. **ADASYN 的 Recall 最高（0.8952）**：在难分类样本周围生成更多合成样本。

### Block 4: SMOTE 泄漏案例

```python
# ---- 泄漏做法: 全数据 SMOTE → Train/Test Split ----
X_full_imp = SimpleImputer(strategy='median').fit_transform(X)     # 全数据插补
X_smote_full, y_smote_full = SMOTE(random_state=RANDOM_STATE).fit_resample(X_full_imp, y)  # 全数据SMOTE
X_tr_leak, X_te_leak, y_tr_leak, y_te_leak = train_test_split(     # 划分
    X_smote_full, y_smote_full, test_size=0.3, random_state=RANDOM_STATE, stratify=y_smote_full)

# ---- 正确做法: Train/Test Split → 仅训练集 SMOTE ----
X_tr_orig = imp.fit_transform(X_tr)                                # 仅训练集插补
X_te_orig = imp.transform(X_te)                                    # 测试集变换
X_tr_smote, y_tr_smote = SMOTE(random_state=RANDOM_STATE).fit_resample(X_tr_orig, y_tr)  # 仅训练集SMOTE
```

**泄漏机制：** SMOTE 在少数类样本之间插值生成合成样本。若在全数据上做 SMOTE 后再划分，测试集中的合成样本是训练集样本的线性组合，模型已"见过"其来源样本，导致 AUC 虚高。

**实验对比：**

| 做法 | AUC | Recall | 说明 |
|------|-----|--------|------|
| 泄漏版（SMOTE→Split） | 0.8990 | 0.8779 | 测试集含合成样本，AUC 虚高 |
| 泄漏版（原始测试集） | 0.8981 | 0.8640 | 揭穿真相 |
| 正确版（Split→SMOTE） | 0.8976 | 0.8640 | 真实泛化能力 |
| Δ（虚高量） | +0.0014 | +0.0139 | 看似小，高维场景会放大 |

### Block 5: 交叉验证中的重采样

```python
# 正确做法: 每折内部对训练折做 SMOTE
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X_cv, y)):
    X_tr_cv, X_te_cv = X_cv[tr_idx], X_cv[te_idx]                 # 外层划分
    y_tr_cv, y_te_cv = y[tr_idx], y[te_idx]

    # 仅在训练折上做 SMOTE
    smote = SMOTE(random_state=RANDOM_STATE)
    X_tr_res, y_tr_res = smote.fit_resample(X_tr_cv, y_tr_cv)     # 训练折SMOTE
    lr = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
    lr.fit(X_tr_res, y_tr_res)                                     # 训练
    auc_cv = roc_auc_score(y_te_cv, lr.predict_proba(X_te_cv)[:, 1])  # 原始验证折评估

# 泄漏版: 全数据 SMOTE → CV (错误)
X_cv_full_smote, y_cv_full_smote = SMOTE(random_state=RANDOM_STATE).fit_resample(X_cv, y)  # 全数据SMOTE
for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X_cv_full_smote, y_cv_full_smote)):
    # 验证折中也含有合成样本 → 泄漏
    ...
```

**CV 泄漏对比：**

| 方法 | Mean AUC | σ | 说明 |
|------|---------|------|------|
| 正确 CV（每折独立 SMOTE） | 0.8902 | 0.0090 | 真实泛化能力 |
| 泄漏 CV（全数据 SMOTE→CV） | 0.8942 | 0.0022 | AUC 虚高 +0.0040 |
| Δ | 0.0040 | — | 泄漏让 σ 更小，是危险的假象 |

**核心要点：** 泄漏 CV 不仅 AUC 虚高，而且标准差更小（0.0022 vs 0.0090），会让审稿人误以为模型"非常稳定"。在严重不平衡场景下，泄漏效应比轻度不平衡更明显。

---

## 重采样方法对比

### 四种方法的核心差异

| 方法 | 原理 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|---------|
| Random UnderSampling | 随机丢弃多数类样本 | 训练快，减少计算成本 | 丢失信息 | 计算资源受限 |
| Random OverSampling | 复制少数类样本 | 简单直接 | 加剧过拟合 | 数据量较小时 |
| SMOTE | 在少数类样本间插值生成合成样本 | 不会精确复制，降低过拟合 | 可能生成边界噪声 | 通用首选 |
| ADASYN | 在难分类区域生成更多样本 | 关注难分类样本 | 更可能生成噪声 | 少数类分布不均匀时 |

### SMOTE vs ADASYN

| 对比 | SMOTE | ADASYN |
|------|-------|--------|
| 年份 | 2002 | 2008 |
| 核心策略 | 所有少数类样本一视同仁 | 密度越低、"越难"的区域生成越多 |
| 本实验 AUC | 0.8976 | 0.8947 |
| 本实验 Recall | 0.8640 | 0.8952 |
| 风险 | 可能生成边界样本 | 更可能生成噪声 |

---

## 关键收获

1. **高准确率 ≠ 好模型**：在 IR=10:1 的数据中，"全预测死亡"模型 Accuracy=90.91% 但 Recall=0，毫无临床价值。不平衡数据中应优先关注 Recall、PR-AUC 等指标。
2. **重采样改变决策阈值而非排序能力**：所有重采样方法的 AUC 差异 ≤ 0.006，但 Recall 从 0.18 提升至 0.90。若仅需排序准确的概率估计，可能不需要重采样；若需检测少数类，重采样是必须的。
3. **SMOTE 泄漏是医学论文中最危险的错误之一**：在全数据上做 SMOTE 后再划分，会使测试集包含训练集样本的线性组合，导致 AUC 虚高。正确做法是先划分再在训练集上做 SMOTE。
4. **CV 中的重采样必须在每折训练集上独立执行**：泄漏 CV 不仅 AUC 虚高（+0.0040），且标准差被人为减小（0.0022 vs 0.0090），制造"模型很稳定"的假象。
5. **PR-AUC 比 ROC-AUC 更诚实**：PR-AUC 只关注正类的 Precision 和 Recall，不会被多数类（91%）的 TN 数量稀释，在不平衡数据中是更可靠的评价指标。

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 不重采样时 Recall 极低 | 模型偏向多数类 | 使用 SMOTE 或 class_weight='balanced' |
| SMOTE 后 AUC 没有提升 | 重采样不改变排序能力 | 这属正常现象，关注 Recall 变化 |
| 泄漏版 CV 方差更小 | 合成样本使各折分布一致 | 切勿被假象误导，必须每折独立 SMOTE |
| ADASYN 生成了噪声 | 在难分类区域过度生成 | 改用 SMOTE 或 SMOTEENN |
| Precision 大幅下降 | 重采样后模型预测更多正类 | 调整决策阈值或使用 F1 评估 |
| 高维数据中泄漏效应放大 | 特征空间更复杂 | 严格使用 Pipeline + imblearn |

---

## 与其他模块的联系

- **前置模块**：Module 07（数据泄漏）— SMOTE 泄漏是数据泄漏在不平衡数据场景下的特殊表现形式；Module 08（交叉验证）— CV 中的重采样策略直接依赖 Pipeline 思想；Module 09（建模对比）— class_weight='balanced' 是处理轻度不平衡的基础手段，本模块深入探讨严重不平衡场景。
- **后续模块**：Module 11（校准分析）— 重采样可能影响概率校准质量，需联合分析；Module 12（可解释性）— 重采样后的模型解释需注意合成样本的影响。


---

## 参考资料

- 教程原文：`ml4health-main/jupyter/10_imbalanced_data.ipynb`
- 讲义：`ml4health-main/lectures/10_imbalanced_data_teaching_doc.md`
- Chawla, N. V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR.
- He, H. et al. (2008). *ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning.* IEEE IJCNN.
- Lemaitre, G. et al. (2017). *Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning.* JMLR.
