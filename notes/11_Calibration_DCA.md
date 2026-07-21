# Module 11 笔记: 校准分析与决策曲线分析 (DCA)

> 本模块的核心命题：**高 AUC 不等于高校准，统计显著不等于临床获益；校准分析回答"预测概率是否可信"，DCA 回答"模型是否具有临床价值"。**

---

## 核心概念梳理

### 校准分析与 DCA 是什么？

校准分析（Calibration Analysis）评估模型的预测概率与实际观测比例之间的一致性。一个完美校准的模型，当其预测概率为 0.8 时，实际应有 80% 的样本为正类。AUC 仅衡量排序能力（正类概率是否系统性高于负类），而不关心概率的绝对数值是否准确。

决策曲线分析（Decision Curve Analysis, DCA）是一种将统计指标转化为临床决策价值的评估方法。通过计算不同决策阈值下的净获益（Net Benefit），DCA 回答"模型在哪些阈值范围内比'全治'或'全不治'更有价值"。DCA 已成为医学 AI 论文的新标准。

### 三个核心问题

| 问题 | 回答工具 | 教学要点 |
|------|---------|---------|
| 模型预测的概率有多可信？ | 校准曲线 + Brier Score + HL 检验 | 高 AUC ≠ 高校准 |
| 这个模型有临床价值吗？ | DCA | 统计显著 ≠ 临床获益 |
| 在哪些风险阈值下应干预？ | DCA 获益范围 | 不同阈值对应不同临床决策 |

### 关键概念

| 概念 | 说明 | 重要性 |
|------|------|--------|
| 预测概率 | 模型的原始输出（0-1 连续值） | 两个模型 AUC 相同但概率分布可能完全不同 |
| 校准度 | 预测概率与实际观测比例的一致性 | 决定医生是否信任模型的概率输出 |
| AUC | 排序能力 | 只关心"顺序"不关心"数值" |
| Brier Score | 预测概率与真实标签的均方误差 | 同时惩罚排序错误和校准错误 |
| 净获益 | 权衡"正确干预"和"错误干预"的临床获益 | 将统计指标翻译为临床决策 |

---

## 代码精读

### Block 1: 模型训练与概率输出

```python
models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', max_iter=5000, random_state=RANDOM_STATE),  # 线性基线
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight='balanced',              # 200棵树
        random_state=RANDOM_STATE, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,                    # 梯度提升
        scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),              # 类别权重
        random_state=RANDOM_STATE, verbosity=0, eval_metric='logloss'),
    'KNN (k=15)': KNeighborsClassifier(n_neighbors=15, n_jobs=-1),           # 15近邻
}

for name, model in models.items():
    model.fit(X_tr_imp, y_tr)                                                # 训练
    y_prob = model.predict_proba(X_te_imp)[:, 1]                             # 预测概率
    auc = roc_auc_score(y_te, y_prob)                                        # AUC
    brier = brier_score_loss(y_te, y_prob)                                   # Brier
    prob_true, prob_pred = calibration_curve(y_te, y_prob, n_bins=10, strategy='uniform')  # 校准曲线
```

**实验结果：**

| 模型 | Brier | AUC | 校准曲线位置 | 校准度 |
|------|-------|-----|------------|--------|
| XGBoost | 0.1153 | 0.9168 | 接近对角线 | 最佳 |
| Random Forest | 0.1164 | 0.9162 | 偏低（过度自信） | 良好 |
| KNN (k=15) | 0.1216 | 0.9048 | 几乎完美匹配 | 中 |
| Logistic Regression | 0.1307 | 0.8944 | 系统性偏差 | 最差 |

### Block 2: 校准曲线绘制

```python
# 校准曲线: 预测概率 vs 实际比例
fig, ax = plt.subplots(figsize=(10, 8))
for idx, (name, data) in enumerate(calibration_data.items()):
    ax.plot(data['prob_pred'], data['prob_true'], 'o-',          # 预测概率 vs 实际比例
            color=colors_cc[idx], linewidth=2.5, markersize=8,
            label=f"{name} (Brier={data['brier']:.4f})")

ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=2,         # 完美校准对角线
        label='Perfect Calibration')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Observed Proportion (True Probability)')
```

**校准曲线解读：**

| 曲线位置 | 含义 | 示例 |
|---------|------|------|
| 在对角线上 | 模型不够自信（预测概率低于实际比例） | 预测 0.3，实际 0.5 |
| 在对角线下 | 模型过度自信（预测概率高于实际比例） | 预测 0.8，实际 0.4 |
| 在对角线上 | 完美校准 | 预测 0.6，实际 0.6 |

### Block 3: Hosmer-Lemeshow 检验

```python
def hosmer_lemeshow_test(y_true, y_prob, n_groups=10):
    """Hosmer-Lemeshow 检验
    H0: 模型校准良好 (预测概率 = 实际比例)
    p < 0.05 → 拒绝 H0 → 校准不良
    """
    df_hl = pd.DataFrame({'y': y_true, 'prob': y_prob})
    df_hl['decile'] = pd.qcut(df_hl['prob'].rank(method='first'),  # 按概率分10组
                               q=n_groups, labels=False)

    obs_pos = df_hl.groupby('decile')['y'].sum().values            # 观测正类数
    exp_pos = df_hl.groupby('decile')['prob'].sum().values         # 期望正类数
    obs_neg = df_hl.groupby('decile')['y'].count().values - obs_pos
    exp_neg = df_hl.groupby('decile')['prob'].count().values - exp_pos

    chi2_stat = np.sum((obs_pos - exp_pos)**2 / exp_pos +          # 卡方统计量
                       (obs_neg - exp_neg)**2 / exp_neg)
    p_value = 1 - chi2.cdf(chi2_stat, n_groups - 2)               # p值
    return chi2_stat, p_value
```

**实验结果：**

| 模型 | χ² | p-value | 校准判断 |
|------|-----|---------|---------|
| KNN (k=15) | 极大 | 0.0000 | 不良 |
| XGBoost | 54.74 | 0.0000 | 不良 |
| Random Forest | 74.64 | 0.0000 | 不良 |
| Logistic Regression | 111.46 | 0.0000 | 不良 |

**HL 检验的局限性：**

| 局限 | 说明 |
|------|------|
| 对样本量敏感 | 大样本下即使微小校准偏差也会被判定为"显著不良" |
| 分组方式影响大 | 等频分组 vs 等距分组可能给出不同结论 |
| 只能整体检验 | 无法定位具体哪个概率区间校准差 |
| 建议 | 校准曲线做可视化诊断 + HL 检验做形式验证 |

### Block 4: Brier Score 的 Murphy 分解

```python
def brier_decomposition(y_true, y_prob):
    """Murphy分解: Brier = 校准度 - 鉴别力 + 不确定性"""
    y_bar = y_true.mean()
    uncertainty = y_bar * (1 - y_bar)                              # 不确定性=ȳ(1-ȳ)

    bin_indices = np.digitize(y_prob, bin_edges) - 1               # 分到10个bin
    for bin_i in range(n_bins):
        mask = bin_indices == bin_i
        n_k = mask.sum()
        o_k = y_true[mask].mean()                                  # 实际正类比例
        r_k = y_prob[mask].mean()                                  # 预测概率均值
        calibration += n_k / len(y_true) * (o_k - r_k)**2         # 校准度分量
        refinement += n_k / len(y_true) * o_k * (1 - o_k)         # 鉴别力分量

    return refinement, calibration, uncertainty
```

**Murphy 分解公式：**

$$\text{Brier} = \underbrace{\text{Calibration}}_{\text{校准度}} - \underbrace{\text{Refinement}}_{\text{鉴别力}} + \underbrace{\text{Uncertainty}}_{\text{不确定性}}$$

| 组件 | 含义 | 可改善性 |
|------|------|---------|
| 鉴别力 (Refinement) | 模型在概率分组内的"纯度" | 由模型本身决定 |
| 校准度 (Calibration) | 预测概率与实际比例的差距 | 可通过 Platt Scaling 等改善 |
| 不确定性 (Uncertainty) | 数据本身的噪声 = ȳ(1-ȳ) | 固定值，无法改变 |

**关键教学发现：** KNN 的校准度分量最好（0.0010），但整体 Brier 不是最低（0.1216），因为 KNN 的鉴别力较高，部分抵消了校准优势。同时 KNN 的 Recall 最低（0.8335）。Brier 单独看会产生误导，必须结合 Recall 和校准曲线综合分析。

### Block 5: DCA 净获益计算

```python
def dca_net_benefit(y_true, y_prob, thresholds):
    """计算 DCA 的净获益
    NB = TP/N - FP/N × (pt / (1-pt))
    """
    N = len(y_true)
    net_benefits = []
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)                        # 按阈值二值化
        TP = ((y_pred == 1) & (y_true == 1)).sum()                 # 真阳性
        FP = ((y_pred == 1) & (y_true == 0)).sum()                 # 假阳性
        nb = TP / N - FP / N * (pt / (1 - pt))                     # 净获益公式
        net_benefits.append(nb)
    return np.array(net_benefits)

def dca_treat_all(y_true, thresholds):
    """Treat All: 所有患者都干预"""
    N = len(y_true)
    n_pos = (y_true == 1).sum()
    for pt in thresholds:
        TP = n_pos                                                  # 所有正类都检出
        FP = N - n_pos                                             # 所有负类都是FP
        nb = TP / N - FP / N * (pt / (1 - pt))
    return net_benefits

def dca_treat_none(y_true, thresholds):
    """Treat None: 都不干预 → 净获益=0"""
    return np.zeros(len(thresholds))
```

**净获益公式解读：**

$$\text{Net Benefit} = \frac{TP}{N} - \frac{FP}{N} \times \frac{p_t}{1-p_t}$$

- 第一项（TP/N）：正确干预带来的获益
- 第二项（FP/N × pt/(1-pt)）：错误干预的代价，权重随阈值增大而增大
- 当 pt 很小时：FP 权重小，模型只需避免 FN（适合癌症筛查）
- 当 pt 很大时：FP 权重大，模型必须避免 FP（适合化疗决策）

### Block 6: DCA 曲线与获益范围分析

```python
thresholds = np.linspace(0.01, 0.99, 99)                           # 99个阈值点

for name, data in calibration_data.items():
    dca_results[name] = dca_net_benefit(y_te, data['y_prob'], thresholds)  # 模型净获益

# 寻找临床获益范围: 模型优于 Treat All 和 Treat None 的阈值区间
for name, nb in dca_results.items():
    better_than_none = nb > nb_treat_none                         # 优于全不治
    better_than_all = nb > nb_treat_all                           # 优于全治
    beneficial = better_than_none & better_than_all              # 同时满足

    idx_beneficial = np.where(beneficial)[0]
    pt_start = thresholds[idx_beneficial[0]]                      # 获益起点
    pt_end = thresholds[idx_beneficial[-1]]                       # 获益终点
    max_nb = nb.max()                                             # 最大净获益
```

**DCA 实验结果：**

| 模型 | 临床获益范围 | 最大净获益 | 最优阈值 | 说明 |
|------|------------|-----------|---------|------|
| Logistic Regression | [0.01, 0.87] | 0.4069 | 0.01 | 范围较窄 |
| Random Forest | [0.01, 0.97] | 0.4072 | 0.01 | 范围宽 |
| XGBoost | [0.01, 0.98] | 0.4088 | 0.01 | 范围最宽、获益最大 |
| KNN (k=15) | [0.01, 0.94] | 0.4073 | 0.01 | 范围宽但获益不突出 |

**DCA 三条基线：**

| 策略 | 含义 | 净获益曲线特征 |
|------|------|-------------|
| Treat All | 所有患者都干预 | 高阈值时急剧下降 |
| Treat None | 都不干预 | 始终为 0 |
| Model | 概率 > 阈值时干预 | 存在"临床获益区间" |

---

## 校准改善方法

| 方法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| Platt Scaling | 在模型输出上拟合 Logistic 回归 | 简单、保序 | 假设 Sigmoid 形状 |
| Isotonic Regression | 非参数保序回归 | 灵活、无假设 | 小样本易过拟合 |
| Temperature Scaling | 除以温度参数后软最大化 | 单一参数、保序 | 最适用于神经网络 |
| Beta Calibration | 专为 [0,1] 区间设计 | 针对概率优化 | 不常见 |

```python
# Platt Scaling 示例
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)  # Sigmoid校准
calibrated.fit(X_train, y_train)
y_prob_calibrated = calibrated.predict_proba(X_test)[:, 1]
```

---

## 模型"临床可用性"评估流程

```
第 1 步: AUC > 0.70?                    → 基本排除能力
第 2 步: 校准曲线接近对角线?              → 概率可信度
第 3 步: Brier < 0.2?                    → 综合误差
第 4 步: DCA 在临床阈值下优于基线?        → 临床价值
第 5 步: 确定模型的"最佳阈值范围"          → 临床操作化
```

---

## 关键收获

1. **AUC 和校准度是独立的质量维度**：一个 AUC=1.0 的模型可能 Brier=0.25（所有概率集中在 0.5 附近），排序完美但概率值不可信。AUC 回答"哪个患者风险更高"，校准回答"风险有多高"，两者缺一不可。
2. **Brier Score 需要 Murphy 分解才能深入理解**：Brier = 校准度 - 鉴别力 + 不确定性。KNN 校准度最好但 Brier 不是最低，因为鉴别力分量较高。单独看 Brier 会产生误导。
3. **HL 检验在大样本下过于敏感**：本实验中所有模型的 HL 检验 p 值均为 0.0000，但这并不意味所有模型都不可用。HL 检验应与校准曲线可视化结合使用。
4. **DCA 将统计指标翻译为临床决策**：DCA 回答"模型在哪些阈值下比全治/全不治更有价值"。本实验中所有模型在 pt ≤ 0.87 时均有临床获益，最大净获益约 0.408。
5. **校准度是可以主动改善的**：通过 Platt Scaling、Isotonic Regression 等后处理方法，可以在不改变模型排序能力的前提下改善概率校准。

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 高 AUC 但 Brier 高 | 排序好但概率值不准 | 使用 Platt Scaling 校准 |
| HL 检验全部不通过 | 大样本下检验过于敏感 | 以校准曲线可视化为主 |
| RF 校准曲线偏低 | 随机森林倾向于过度自信 | 使用 Isotonic Regression |
| DCA 获益范围窄 | 模型在高阈值下 FP 过多 | 考虑调整决策阈值或改用其他模型 |
| KNN 校准好但 Recall 差 | 鉴别力高但排序能力不足 | 不应仅凭校准度选择模型 |
| 校准后 AUC 下降 | 校准方法可能影响排序 | 使用保序方法（Platt/Isotonic） |

---

## 与其他模块的联系

- **前置模块**：Module 09（建模对比）— 本模块深入分析了 Module 09 中各模型的 Brier Score 差异，揭示了 AUC 与校准度的独立性；Module 10（类别不平衡）— 不平衡数据中的概率往往被压缩，影响校准质量。
- **后续模块**：Module 12（可解释性）— SHAP 分析可以解释模型为何给出某个概率值，与校准分析互补，共同构建模型可信度。


---

## 参考资料

- 教程原文：`ml4health-main/jupyter/11_calibration_dca.ipynb`
- 讲义：`ml4health-main/lectures/11_calibration_dca_teaching_doc.md`
- Vickers, A. J., & Elkin, E. B. (2006). *Decision Curve Analysis: A Novel Method for Evaluating Prediction Models.* Medical Decision Making.
- Niculescu-Mizil, A., & Caruana, R. (2005). *Predicting Good Probabilities With Supervised Learning.* ICML.
- Guo, C. et al. (2017). *On Calibration of Modern Neural Networks.* ICML.
- Murphy, A. H. (1973). *A New Vector Partition of the Probability Score.* Journal of Applied Meteorology.
