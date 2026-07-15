# Module 13 笔记: SHAP 特征交互效应分析

> 本模块的核心命题：**特征并非孤立作用于模型预测，两个特征的联合影响可能远超各自贡献的简单叠加，量化与可视化这种交互效应是理解模型深层逻辑的关键。**

---

## 核心概念梳理

### SHAP 交互效应是什么？

在传统可加模型中，各特征对预测的贡献可简单求和。然而在树模型等非线性模型中，特征之间可能存在交互效应（Interaction Effect）：两个特征联合作用时，其总贡献偏离各自单独贡献之和。SHAP 交互分析旨在量化这种偏离，揭示模型决策中的"协同"与"抑制"机制。

在医学预测场景中，交互效应尤为重要。例如，诊断年份（year）与诊断方式（Diagnostic.means）对癌症存活概率的联合影响可能远超两者各自的影响——近年特定诊断技术的进步改善了特定亚群的预后，但这种改善并非均匀分布。理解交互效应，有助于构建更精确的特征工程策略，并为临床解释提供更丰富的维度。

### 交互效应的三种类型

| 类型 | 机制 | 依赖图表现 | 医学示例 |
|------|------|-----------|---------|
| 增强效应 (Synergistic) | 特征 A↑ + 特征 B↑ → 影响大于 A+B 各自之和 | 颜色变化时散点 y 值范围扩大 | year 高 + 特定 Diagnostic.means → 存活概率极高 |
| 抑制效应 (Antagonistic) | 特征 A↑ → 特征 B 的作用被削弱 | 不同斜率的趋势线 | year 的正影响在年龄高时减弱 |
| 调节效应 (Moderation) | 交互特征改变主特征 SHAP 方向 | 颜色区域在 y=0 线两侧翻转 | Extension 低时 Age↑→SHAP↓；Extension 高时 Age↑→SHAP↑ |

### 两种交互强度度量方法

| 方法 | 公式 | 含义 | 优势 |
|------|------|------|------|
| M1: \|corr(SHAP_i, X_j)\| | SHAP 值与交互特征原始值的皮尔逊相关 | 特征 j 的值变化时，特征 i 的贡献是否随之变化 | 捕捉"真正的交互调节效应" |
| M2: \|corr(SHAP_i, SHAP_j)\| | 两个 SHAP 值的皮尔逊相关 | 特征 i 贡献高时，特征 j 的贡献是否也高 | 捕捉"贡献模式的相似性" |

**关键差异**：M1 高代表真正的交互调节，M2 高可能仅反映共同趋势（虚假相关）。两种方法缺一不可——M1 与 M2 同时高的交互对才最值得关注。

---

## 代码精读

### Block 1: 数据加载与预处理

```python
# 加载癌症数据集
df = pd.read_csv(DATA_PATH, low_memory=False, encoding='latin-1')
# 将生存状态映射为二分类目标
df['target'] = df['Status.Vital'].map({'VIVO': 1, 'MORTO': 0})
# 限制样本量至 10000
if len(df) > N_SAMPLES:
    idx = np.random.choice(len(df), N_SAMPLES, replace=False)
    df = df.iloc[idx].copy()

# Boruta 筛选后的 6 个特征
feature_cols = ['Age', 'year', 'Code.Profession', 'Diagnostic.means',
                'Extension', 'Raca.Color']
# 分类变量编码
for col in cat_cols:
    le = LabelEncoder()
    le.fit(df_feat[col].dropna().astype(str))
    df_feat[col] = df_feat[col].apply(encode)

# 标准化处理
X_tr_final = scaler.fit_transform(X_tr_imp)
X_te_final = scaler.transform(X_te_imp)
```

**要点**：本教程使用经 Boruta 筛选后的 6 个特征，而非原始 30+ 特征。筛选后特征冗余度极低，交互强度自然降低（0.10~0.16 范围），这恰恰说明特征选择方法有效去除了共线性冗余。

### Block 2: 模型训练与 SHAP 计算

```python
# 训练随机森林分类器
model = RandomForestClassifier(
    n_estimators=200, max_depth=8,
    class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
model.fit(X_tr_final, y_tr)

# SHAP TreeExplainer 计算
explainer = shap.TreeExplainer(model)
shap_values_full = explainer.shap_values(X_shap)
# 提取类别 1 (VIVO) 的 SHAP 值
sv = shap_values_full[1] if isinstance(shap_values_full, list) else shap_values_full

# 按重要性排序特征
shap_importance = np.abs(sv).mean(0)
feature_order = np.argsort(shap_importance)[::-1]
```

**要点**：使用 500 个测试样本计算 SHAP 值。TreeExplainer 对树模型提供精确的 SHAP 值计算，无需 KernelExplainer 的近似采样。

### Block 3: 交互效应扫描——发现最强交互对

```python
# 构建交互强度矩阵
interaction_matrix = np.zeros((n_features, n_features))

for i in range(n_features):
    for j in range(n_features):
        if i == j: continue
        # M1: SHAP_i 与 X_j 的相关强度
        corr_val, _ = pearsonr(sv[:, i], X_shap[:, j])
        strength_method1 = abs(corr_val)
        # M2: SHAP_i 与 SHAP_j 的相关强度
        corr_shap, _ = pearsonr(sv[:, i], sv[:, j])
        strength_method2 = abs(corr_shap)
        # 取两种方法中更强者
        strength = max(strength_method1, strength_method2)
        interaction_matrix[i, j] = strength
```

**要点**：交互扫描自动为每个特征寻找最强交互对方。实验发现 year 与 Diagnostic.means 彼此互为最强交互对象（强度 0.1563），两种方法一致，构成"互作用核心"。

### Block 4: 交互依赖图（2×2 面板）

```python
# Top 4 特征的交互依赖面板
for p_idx, main_rank in enumerate(range(top_n)):
    main_idx = top_idx_list[main_rank]
    main_values = X_shap[:, main_idx]       # 主特征值
    shap_vals = sv[:, main_idx]              # 主特征 SHAP 值
    interact_values = X_shap[:, interact_idx] # 交互特征值（颜色编码）

    # 散点图：按交互特征值着色
    scatter = ax.scatter(main_values, shap_vals, c=interact_values,
                         cmap='viridis', alpha=0.6, s=45)
    # 二次拟合趋势线
    z = np.polyfit(x_clean, y_clean, 2)
    p = np.poly1d(z)
    ax.plot(x_range, p(x_range), color='darkred', linewidth=2.5)
    # 交互强度标注框
    ax.text(0.05, 0.95, f'Interaction Strength = {strength:.4f}')
```

**读图顺序**：颜色范围 → 颜色与 y 值关联 → 趋势线弯曲 → 强度标注值。交互强度 > 0.10 值得注意，> 0.30 为强交互。

### Block 5: 交互矩阵热图

```python
# 6×6 交互强度矩阵可视化
im = ax.imshow(interaction_matrix, cmap='YlOrRd', aspect='auto', vmin=0)
# 添加数值标注
for i in range(n_features):
    for j in range(n_features):
        if i != j:
            val = interaction_matrix[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center')
```

**要点**：交互矩阵是**非对称**的。Age × year = 0.127 但 year × Age = 0.036，说明 year 的值变化能改变 Age 的 SHAP 贡献，但 Age 的值变化几乎不影响 year 的贡献。year 是"中心调节者"——它影响其他特征的贡献模式，但几乎不被反向调节。

### Block 6: 交互强度排名图

```python
# 收集所有非对角交互对，按强度排序取 Top 10
all_pairs = []
for i in range(n_features):
    for j in range(n_features):
        if i != j:
            all_pairs.append((feature_names[i], feature_names[j],
                              interaction_matrix[i, j]))
all_pairs.sort(key=lambda x: x[2], reverse=True)
top_pairs = all_pairs[:10]
```

**要点**：排名前 3 的交互对均为 year/Diagnostic.means 相关，确认了"互作用核心"的判断。

### Block 7: 深度交互分析——以 year 为例

```python
# year 与所有其他特征的交互形态多面板图
year_idx = list(feature_names).index('year')
for j in other_indices:
    x_vals = X_shap[:, year_idx]  # year 值
    y_vals = sv[:, year_idx]       # year SHAP 值
    color_vals = X_shap[:, j]      # 其他特征值（颜色）
    # 散点 + 二次拟合 + 交互强度标注
    ax.scatter(x_vals, y_vals, c=color_vals, cmap='coolwarm')
    ax.text(0.05, 0.05, f'Strength={interaction_matrix[year_idx, j]:.4f}')
```

**要点**：year × Diagnostic.means (0.156) 最强交互，year × Extension (0.003) 几乎无交互——肿瘤扩展程度不影响年份的贡献模式。

### Block 8: 方法对比——M1 vs M2

```python
# 分别计算两种方法的交互矩阵
method1_matrix = np.zeros((n_features, n_features))  # M1: |corr(SHAP_i, X_j)|
method2_matrix = np.zeros((n_features, n_features))  # M2: |corr(SHAP_i, SHAP_j)|
for i in range(n_features):
    for j in range(n_features):
        if i == j: continue
        corr_m1, _ = pearsonr(sv[:, i], X_shap[:, j])
        method1_matrix[i, j] = abs(corr_m1)
        corr_m2, _ = pearsonr(sv[:, i], sv[:, j])
        method2_matrix[i, j] = abs(corr_m2)
```

**关键发现**：Age × year 的 M1=0.127, M2=0.036，Δ=+0.091。M1 高说明 year 调节了 Age 的贡献（真正的交互）；M2 低说明两者贡献模式不相关。反之，Diagnostic.means × year 的 M2=0.156, M1=0.017，Δ=-0.139，反映的是"共同趋势"而非双向交互。

---

## 关键收获

1. **year × Diagnostic.means 是最强交互对**（强度 0.1563），两者互为最强交互对象，两种度量方法一致
2. **交互矩阵是非对称的**：|corr(SHAP_Age, X_year)| ≠ |corr(SHAP_year, X_Age)|，交互具有方向性
3. **两种度量方法各有侧重**：M1 捕捉真正调节效应，M2 捕捉贡献模式相似性；两者同时高才值得重点关注
4. **Boruta 筛选后交互强度自然降低**（0.10~0.16），因冗余特征被去除，说明特征选择有效
5. **year 是"中心调节者"**：影响其他特征的贡献模式，但几乎不被反向调节——临床可解读为医疗技术进步改变了各年龄组的预后模式
6. **交互分析的扩展方法**：SHAP Interaction Values（精确但计算成本高）与 Friedman's H-statistic（全局平均、模型无关）

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 交互强度仅 0.15，是否意味交互弱？ | Boruta 筛选后特征独立性提高，交互自然降低 | 与未筛选的 30+ 特征交互强度（可达 0.5+）比较 |
| 交互矩阵不对称，如何解释？ | 皮尔逊相关度量的是单向调节关系 | 分别解读两个方向的含义 |
| M1 与 M2 排名不一致 | M1 反映交互调节，M2 反映贡献相似性 | 同时报告两种度量，以 Δ 值区分"真交互"与"共同趋势" |
| 高 R² 二次拟合是否意味交互强？ | 高 R² 可能仅反映非线性主效应 | 结合交互强度标注判断 |
| 仅线性交互？pearsonr 只捕捉线性 | 局限于线性关联 | 同时使用 M1/M2 并考虑 SHAP Interaction Values |

---

## 与其他模块的联系

- **前置模块**：Module 12 SHAP 概述（SHAP 值、依赖图）、Module 12b 高级 SHAP（二次拟合 R²、交互网络图）
- **后续模块**：Module 14 SHAP 依赖图（引入数据密度维度验证交互模式可信度）、Module 15 SHAP 聚类（发现具有不同交互模式的子群）
- **与研究工作的联系**：在 EcMurJ 虚拟筛选中，化合物结合亲和力可能与分子描述符之间存在交互效应（如疏水性与柔性联合影响 docking 评分）；宏基因组研究中，肠道菌群属间的交互效应可能影响长寿相关代谢通路

---

## 参考资料

- 教程原文：`ml4health-main/jupyter/13_shap_interaction_analysis.ipynb`
- 讲义：`ml4health-main/lectures/13_shap_interaction_analysis_teaching_doc.md`
- Lundberg SM, Lee SI. A unified approach to interpreting model predictions. *NeurIPS* 2017.
- Friedman JH, Popescu BE. Predictive learning via rule ensembles. *Ann Appl Stat* 2008. (H-statistic)
