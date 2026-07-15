# Module 09 笔记: 机器学习建模 — 七模型对比

> 本模块的核心命题：**在统一评估框架下系统比较七种机器学习模型在医学数据上的性能、稳定性与计算成本差异，理解不同模型的行为特征与适用场景。**

---

## 核心概念梳理

### 模型对比的框架设计

模型对比的核心在于控制变量：同一数据集、同一评估方法（Stratified 5-Fold CV）、同一评估指标体系，仅变化模型本身。本模块在 20,000 例癌症患者数据（VIVO 存活 41.15%、MORTO 死亡 58.85%）上比较七种模型，涵盖线性模型、非参数方法、决策树及集成学习三大流派。

评估指标采用四维体系：AUC（综合排序能力）、Recall（少数类检出能力）、Brier Score（概率校准质量）、训练时间（计算成本），同时关注跨折标准差（模型稳定性）。

### 七种模型概览

| 模型 | 数学直觉 | 工作原理 | 关键参数 |
|------|---------|---------|---------|
| Logistic Regression | 用一条线把两类分开 | 线性决策边界 + Sigmoid | C=1.0, class_weight='balanced' |
| SVM (Linear) | 找到边界最宽的分隔线 | 最大化支持向量到超平面间隔 | class_weight='balanced' |
| KNN | 看最近 K 个邻居投票 | 距离度量 + 邻居投票 | k=15 |
| Decision Tree | 提一系列"是否"问题 | 递归二分，最大化信息增益 | max_depth=10 |
| Random Forest | 建一批树投票 | Bagging + 随机特征子空间 | 200 棵树, max_depth=12 |
| XGBoost | 逐棵纠正前一棵的错误 | 梯度提升 + 正则化 + 二阶优化 | 300 棵树, lr=0.1 |
| LightGBM | 同 XGBoost 但更快 | GOSS + EFB 优化 | 300 棵树, lr=0.1 |

---

## 代码精读

### Block 1: 统一 Pipeline 构建与评估函数

```python
def build_pipeline(model, scale_needed=True):
    """构建统一的预处理+建模 Pipeline"""
    steps = []
    steps.append(('imputer', SimpleImputer(strategy='median')))    # 中位数插补
    if scale_needed:                                               # 需要标准化时
        steps.append(('scaler', StandardScaler()))                 # 标准化
    steps.append(('model', model))                                 # 添加模型
    return Pipeline(steps)

def evaluate_model_cv(X, y, model_name, model, scale_needed=True, n_splits=5):
    """Stratified K-Fold 评估模型性能与稳定性"""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,          # 分层5折
                          random_state=RANDOM_STATE)
    pipe = build_pipeline(model, scale_needed)

    for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
        pipe.fit(X[tr_idx], y[tr_idx])                             # 训练
        y_prob = pipe.predict_proba(X_te)[:, 1]                    # 预测概率
        y_pred = (y_prob >= 0.5).astype(int)                       # 二值化

        auc = roc_auc_score(y_te, y_prob)                          # AUC
        rec = recall_score(y_te, y_pred, pos_label=1)             # Recall
        brier = brier_score_loss(y_te, y_prob)                    # Brier
```

**要点说明：**

1. **统一 Pipeline**：所有模型共享相同的预处理流程（中位数插补 + 标准化），确保差异仅来自模型本身。
2. **scale_needed 参数**：决策树和随机森林不需要标准化（基于阈值分裂），其余模型需要。
3. **Stratified 5-Fold**：沿用 Module 08 验证的最优评估策略。

### Block 2: 七个模型的定义

```python
models = [
    ('Logistic Regression',
     LogisticRegression(class_weight='balanced', max_iter=5000,    # 平衡类别权重
                        random_state=RANDOM_STATE), True),
    ('SVM (Linear)',
     LinearSVC(class_weight='balanced', max_iter=5000,             # 线性SVM
               loss='hinge', random_state=RANDOM_STATE), True),
    ('KNN (k=15)',
     KNeighborsClassifier(n_neighbors=15, n_jobs=N_JOBS),          # 15近邻
     True),
    ('Decision Tree',
     DecisionTreeClassifier(class_weight='balanced', max_depth=10, # 深度限制10
                            random_state=RANDOM_STATE), False),
    ('Random Forest',
     RandomForestClassifier(n_estimators=200, max_depth=12,        # 200棵树
                            class_weight='balanced',
                            random_state=RANDOM_STATE, n_jobs=N_JOBS), False),
    ('XGBoost',
     xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,          # 300棵树
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),          # 正类权重≈1.43
        random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0,
        eval_metric='logloss', use_label_encoder=False), True),
    ('LightGBM',
     lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,          # 300棵树
        class_weight='balanced',                                   # 平衡类别权重
        random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=-1), True),
]
```

**关键参数说明：**

| 模型 | 不平衡处理方式 | 标准化需求 | 说明 |
|------|-------------|----------|------|
| LR | class_weight='balanced' | 需要 | 自动调整类别权重 |
| SVM | class_weight='balanced' | 需要 | 同 LR |
| KNN | 不支持 | 需要 | 依赖距离度量 |
| DT | class_weight='balanced' | 不需要 | 基于阈值分裂 |
| RF | class_weight='balanced' | 不需要 | 同 DT |
| XGBoost | scale_pos_weight≈1.43 | 不影响 | 正类权重 = n_negative / n_positive |
| LightGBM | class_weight='balanced' | 不影响 | 同 LR |

### Block 3: 模型评估执行

```python
# 遍历所有模型进行评估
for name, model, scale_needed in models:
    result = evaluate_model_cv(X, y, name, model, scale_needed)    # CV评估
    all_results.append(result)
    print(f"    AUC = {result['auc_mean']:.4f} ± {result['auc_std']:.4f}")
    print(f"    Recall = {result['recall_mean']:.4f} ± {result['recall_std']:.4f}")
    print(f"    Brier = {result['brier_mean']:.4f} ± {result['brier_std']:.4f}")
    print(f"    Time = {result['time_total']:.2f}s")
```

**实验结果：**

| 排名 | 模型 | AUC | σ(AUC) | Recall | Brier | 时间(s) |
|------|------|-----|--------|--------|-------|---------|
| 1 | **LightGBM** | **0.9423** | 0.0016 | 0.8899 | **0.0974** | 64.64 |
| 2 | XGBoost | 0.9415 | 0.0019 | 0.8863 | 0.0982 | 22.79 |
| 3 | Random Forest | 0.9391 | 0.0022 | **0.8984** | 0.1002 | 1.83 |
| 4 | Decision Tree | 0.9149 | 0.0018 | 0.8869 | 0.1145 | 0.12 |
| 5 | KNN (k=15) | 0.9111 | **0.0012** | 0.8335 | 0.1185 | 0.06 |
| 6 | SVM (Linear) | 0.8946 | 0.0033 | **0.8984** | 0.1347 | 0.18 |
| 7 | Logistic Regression | 0.8936 | 0.0033 | 0.8786 | 0.1310 | 0.06 |

### Block 4: 可视化分析

```python
# 1. 性能条形图 (AUC + Recall + Brier)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))                    # 三列子图
for idx, (mean_key, std_key, title, _, higher_better) in enumerate(metrics):
    bars = ax.bar(range(len(names)), means, yerr=stds,             # 带误差棒
                  color=colors, edgecolor='white', capsize=5)

# 2. 稳定性雷达图
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))  # 极坐标
for i, name in enumerate(names):
    values = [stability_scores[k][i] for k in metric_keys]        # 标准化稳定性
    ax.plot(angles, values, 'o-', linewidth=2, label=name)        # 雷达线

# 3. 速度 vs 性能散点图
ax.scatter(r['time_total'], r['auc_mean'], s=150)                  # 训练时间 vs AUC

# 4. ROC 曲线 (Top 5 模型)
fpr, tpr, _ = roc_curve(y_test, y_prob)                           # 计算 ROC
ax.plot(fpr, tpr, linewidth=2.5, label=f'{name} (AUC={auc:.4f})')

# 5. Precision-Recall 曲线
prec, rec, _ = precision_recall_curve(y_test, y_prob)             # 计算 PR 曲线

# 6. 逐折 AUC 稳定性图
for fold_i in range(5):
    fold_aucs = [r['aucs'][fold_i] for r in all_results]          # 每折AUC
    ax.bar(x_pos + offset, fold_aucs, width, label=f'Fold {fold_i+1}')
```

**可视化要点：**

| 图表 | 维度 | 核心信息 |
|------|------|---------|
| 性能条形图 | AUC / Recall / Brier | 模型间性能排序 |
| 稳定性雷达图 | 三指标的标准差 | 模型跨折稳定性 |
| 速度 vs 性能 | 训练时间 vs AUC | 性价比分析 |
| ROC 曲线 | FPR vs TPR | 综合排序能力 |
| PR 曲线 | Recall vs Precision | 不平衡数据下的表现 |
| 逐折 AUC | 各折 AUC 分布 | 跨折方差来源 |

---

## 核心发现

### 发现 1: 集成树模型占据前三

LightGBM（0.9423）、XGBoost（0.9415）、Random Forest（0.9391）的 AUC 差距小于 0.003，均能很好地捕捉非线性关系。梯度提升与 Bagging 两种集成策略在该数据集上表现相当。

### 发现 2: KNN 在轻度不平衡下仍可工作

KNN 的 Recall（0.8335）在七个模型中最低，但远非"灾难"级别。不平衡程度是决定 KNN 是否失效的关键：在严重不平衡下 KNN 几乎检测不到少数类，而在轻度不平衡（VIVO 约 41%）下近邻中有足够的正类样本。

### 发现 3: SVM 与 RF 的 Recall 最高

经 class_weight='balanced' 加权后，SVM 和 RF 的 Recall 均达到 0.8984。在医学场景中，高 Recall 意味着较低的假阴性率（漏诊率），具有重要的临床价值。

### 发现 4: 线性模型跨折方差最大

Logistic Regression 和 SVM 的 σ=0.0033，是 KNN（0.0012）的近 3 倍。线性模型对数据分布变化更敏感，而 KNN 基于局部邻域的投票机制天然具有较低的方差。

### 发现 5: 性能与成本的权衡

LightGBM 性能最优但训练耗时 64.64s，是 Random Forest（1.83s）的 35 倍。RF 在性价比上具有明显优势。

---

## 集成学习三大流派

| 流派 | 代表模型 | 核心思想 | 优势 |
|------|---------|---------|------|
| Bagging | Random Forest | 并行构建多棵树，投票决策 | 降低方差 |
| Boosting | XGBoost, LightGBM | 串行构建，每棵纠正前一棵错误 | 降低偏差 |
| Stacking | 元学习器组合 | 训练元学习器学习如何组合基模型 | 理论性能 ≥ 最优单模型 |

### XGBoost vs LightGBM

| 特性 | XGBoost | LightGBM |
|------|---------|----------|
| 分裂策略 | Level-wise（逐层分裂） | Leaf-wise（逐叶分裂） |
| 采样策略 | 无特殊采样 | GOSS（梯度单边采样） |
| 特征处理 | 预排序 | EFB（互斥特征捆绑） |
| 速度 | 基线 | 快 2-10 倍 |
| 小样本 | 更好 | 可能过拟合 |

---

## 超参数优化方法层级

| 层级 | 方法 | 原理 | 适用场景 |
|------|------|------|---------|
| 第一层 | Grid Search | 穷举所有参数组合 | 参数 ≤ 3 个 |
| 第一层 | Random Search | 参数空间随机采样 | 参数 ≥ 4 个 |
| 第一层 | Bayesian Optimization | 高斯过程建模参数→性能映射 | 中等参数空间 |
| 第二层 | 遗传算法 (GA) | 自然选择 + 遗传学机制 | 高维空间，可并行 |
| 第二层 | 粒子群优化 (PSO) | 模拟鸟群觅食 | 无梯度需求，收敛快 |
| 第二层 | 灰狼优化 (GWO) | 灰狼社会等级引导搜索 | 收敛快，参数少 |

**无免费午餐定理（NFL）**：没有一种优化算法在所有问题上都优于其他算法，算法性能取决于与问题的匹配程度。

---

## 关键收获

1. **没有"最好"的模型**：模型选择取决于可解释性、性能、稳定性与速度的权衡。LightGBM 性能最优但最慢，Logistic Regression 最快且可解释但性能最低。
2. **集成树模型在表格数据上占主导**：前三名均为集成树模型，AUC 差距极小（< 0.003），验证了树模型对异构特征和非线性关系的强适应性。
3. **class_weight 是处理轻度不平衡的有效手段**：经平衡权重调整后，即使线性模型的 Recall 也能达到 0.88 以上。
4. **Brier Score 需结合 Recall 解读**：在轻度不平衡下 Brier Score 的陷阱减弱，但仍建议联合查看以确保模型对少数类的识别能力。
5. **模型选择指南**：可解释性优先选 Logistic Regression / Decision Tree；性能优先选 XGBoost / LightGBM；稳定性优先选 Random Forest。

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| LightGBM 训练时间过长 | 300 棵树 + 叶子优化的计算开销 | 减少 n_estimators 或使用早停 |
| KNN Recall 偏低 | 不支持 class_weight，依赖距离 | 增大 k 或使用距离加权 |
| SVM 与 LR 方差最大 | 线性模型对数据分布敏感 | 增加 CV 重复次数或使用 Repeated CV |
| XGBoost 性能未达预期 | scale_pos_weight 在轻度不平衡下作用减弱 | 调整学习率或增加树的数量 |
| Decision Tree 过拟合 | 单棵树容易记忆噪声 | 限制 max_depth 或使用 RF 替代 |
| ROC 曲线区分度低 | 模型间 AUC 差距小（< 0.003） | 参考 PR 曲线或逐折 AUC 图 |

---

## 与其他模块的联系

- **前置模块**：Module 08（交叉验证）— 本模块统一使用 Stratified 5-Fold CV，是 Module 08 结论的直接应用；Module 07（数据泄漏）— Pipeline 确保每折独立预处理。
- **后续模块**：Module 10（类别不平衡处理）— 本模块使用 class_weight='balanced' 作为基础手段，Module 10 将深入探讨 SMOTE 等高级方法；Module 11（校准分析）— Brier Score 初步反映校准质量，Module 11 将系统分析校准曲线；Module 12（可解释性）— 本模块的树模型和线性模型都将是 SHAP/LIME 解释的对象。
- **与研究工作的联系**：在 EcMurJ 虚拟筛选中，化合物活性预测属于典型表格数据分类问题，XGBoost/LightGBM 可作为基线模型；在宏基因组研究中，样本特征异构性强，Random Forest 的鲁棒性优势值得关注。选择模型时需考虑下游需求：若需向生物学家解释预测逻辑，Logistic Regression + SHAP 可能优于黑箱集成模型。

---

## 参考资料

- 教程原文：`ml4health-main/jupyter/09_modeling_comparison.ipynb`
- 讲义：`ml4health-main/lectures/09_modeling_comparison_teaching_doc.md`
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS.
- Bergstra, J., & Bengio, Y. (2012). *Random Search for Hyper-Parameter Optimization.* JMLR.
- Zhou, Z.-H., & Feng, J. (2017). *Deep Forest.* arXiv.
