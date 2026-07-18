# Module 02 笔记: 统计检验与特征筛选

> 本模块的核心命题：**如何用统计方法判断特征与目标变量的关联强度，以及统计显著与预测力强之间的本质区别。**

---

## 核心概念梳理

### 统计检验在特征筛选中解决什么问题？

在机器学习流程中，统计检验通常在 EDA 之后、特征工程之前执行，承担**特征筛选第一道关卡**的角色。具体回答三个问题：

1. 该特征在存活组与死亡组之间是否存在差异？（统计显著性）
2. 这种差异的实际大小是多少？（效应量）
3. 差异方向如何？（均值/中位数的组间比较）

### 方法选择决策树

```
特征类型
├── 数值型（Age, year, ...）
│   └── 正态性检验
│       ├── 近似正态 → T 检验 + Levene 方差齐性 → Cohen's d
│       └── 非正态   → Mann-Whitney U 检验 → Rank-biserial r
│
└── 分类型（Gender, Extension, ...）
    └── 构建列联表 → 卡方检验 → Cramér's V
```

### 为什么大样本下统计检验容易"全显著"？

样本量 N 与可检测的最小效应量呈反比。当 N ≈ 210,000 时，即使两组均值差仅 0.05 个标准差，p 值也会 < 0.05。本数据集 22 个特征在 Bonferroni 校正（α' = 0.05/22 ≈ 0.0023）后仍然全部显著，正是大样本"放大镜效应"的体现。

---

## 代码精读

### Block 1: 依赖导入与路径配置

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency
```

新增依赖说明：

| 函数 | 来源 | 用途 |
|------|------|------|
| `mannwhitneyu` | `scipy.stats` | 非参数秩和检验，比较两组独立样本 |
| `chi2_contingency` | `scipy.stats` | 卡方独立性检验，判断两个分类变量是否独立 |
| `stats.normaltest` | `scipy.stats` | D'Agostino-Pearson 正态性检验 |
| `stats.levene` | `scipy.stats` | Levene 方差齐性检验 |

---

### Block 2: 数据加载（与 Module 01 一致）

与 01_EDA 使用相同的数据加载和 target 构建逻辑，不再重复解释。

---

### Block 3: 特征分类

```python
exclude_cols = ['Patient.Code', 'target', 'Status.Vital',
                'Date.of.Birth', 'Date.of.Death', 'Date.of.Last.Contact',
                'Date.of.Diagnostic']

numeric_candidates = ['Age', 'Code.Profession', 'Code.of.Morphology', 'year']

categorical_candidates = [
    'Gender', 'Raca.Color', 'Diagnostic.means', 'Extension',
    'Laterality', 'State.Civil', 'Degree.of.Education',
    'Description.of.Topography', 'Topography.Code',
    'Morphology.Description', 'Description.of.Disease',
    'Illness.Code', 'Child.Illness.Description',
    'Youth.Adult.Illness.Description', 'Type.of.Death',
    'Distant.metastasis', 'Nationality', 'Naturality.State'
]

total_n_features = len(numeric_features) + len(categorical_features)
```

要点：
- 排除日期、患者 ID、原始标签列——这些或是时间元数据，或已被 target 替代
- 数值特征仅 4 个（Age、职业编码、形态学编码、年份），其余均为分类特征
- `total_n_features` 用于 Bonferroni 校正的计算

---

### Block 4: 正态性判断函数

```python
def check_normality_practical(data, sample_limit=5000):
    data_clean = data.dropna()
    if len(data_clean) < 30:
        return False, "样本不足"

    skewness = data_clean.skew()

    if len(data_clean) > sample_limit:
        data_test = data_clean.sample(sample_limit, random_state=42)
    else:
        data_test = data_clean

    if abs(skewness) < 0.5:
        return True, f"近似正态 (偏度={skewness:.3f})"
    elif abs(skewness) < 1.0:
        _, p_value = stats.normaltest(data_test)
        if p_value > 0.05:
            return True, f"正态 (p={p_value:.4f})"
        else:
            return False, f"非正态 (偏度={skewness:.3f}, p={p_value:.4f})"
    else:
        return False, f"非正态 (偏度={skewness:.3f})"
```

三层判断逻辑：

| 偏度范围 | 判断策略 | 理由 |
|---------|---------|------|
| \|偏度\| < 0.5 | 直接判为正态 | 分布几乎对称 |
| 0.5 ≤ \|偏度\| < 1.0 | D'Agostino-Pearson 检验辅助 | 边界情况需要正式检验 |
| \|偏度\| ≥ 1.0 | 直接判为非正态 | 分布明显不对称 |

抽样策略：当样本量超过 5000 时，随机抽取 5000 条进行正式检验，避免大样本下微小的偏度也获得极小的 p 值。

---

### Block 5: 数值特征检验（T 检验 / Mann-Whitney U）

```python
for col in numeric_features:
    data = df_model[[col, 'target']].dropna()
    group_vivo = data.loc[data['target'] == 1, col]
    group_morto = data.loc[data['target'] == 0, col]

    is_normal, norm_note = check_normality_practical(data[col])

    if is_normal:
        # Levene 方差齐性检验
        levene_stat, levene_p = stats.levene(group_vivo, group_morto)
        equal_var = levene_p > 0.05

        # 独立样本 T 检验
        t_stat, p_value = stats.ttest_ind(
            group_vivo, group_morto, equal_var=equal_var
        )
        test_used = "T-test"

        # Cohen's d
        pooled_std = np.sqrt(
            ((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2)
        )
        effect_size = abs(mean_v - mean_m) / pooled_std
        effect_type = "Cohen's d"
    else:
        # Mann-Whitney U 检验
        u_stat, p_value = mannwhitneyu(
            group_vivo, group_morto, alternative='two-sided'
        )
        test_used = "Mann-Whitney U"

        # Rank-biserial r
        effect_size = abs(1 - (2 * u_stat) / (n1 * n2))
        effect_type = "Rank-biserial r"
```

**T 检验的完整流程：**

1. 正态性判断（偏度 + D'Agostino-Pearson）
2. Levene 方差齐性检验 → 决定 `equal_var` 参数
3. 执行 T 检验
4. 计算 Cohen's d 效应量

**Mann-Whitney U 检验的流程：**

1. 不假设正态，直接用秩和检验
2. 计算 Rank-biserial r 效应量

选择逻辑：本数据集的 4 个数值特征（Age、Code.Profession、Code.of.Morphology、year）均为偏态分布，因此全部使用 Mann-Whitney U。

---

### Block 6: 分类特征检验（卡方检验）

```python
for col in categorical_features:
    data = df_model[[col, 'target']].dropna()

    # 过滤频数 < 5 的稀疏类别
    value_counts = data[col].value_counts()
    valid_categories = value_counts[value_counts >= 5].index
    data_filtered = data[data[col].isin(valid_categories)]

    # 构建列联表
    contingency = pd.crosstab(data_filtered[col], data_filtered['target'])

    # 卡方检验
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency)

    # Cramér's V
    n_total = contingency.values.sum()
    phi2 = chi2_stat / n_total
    k = min(contingency.shape) - 1
    cramer_v = np.sqrt(phi2 / k) if k > 0 else 0
```

关键预处理：删除频数 < 5 的类别。如果列联表中存在期望频数 < 5 的单元格，卡方近似精度下降，此时应改用 Fisher 精确检验或合并稀疏类别。

**Cramér's V 公式推导：**
$$V = \sqrt{\frac{\chi^2}{n \cdot (k-1)}}$$
其中 k = min(行数, 列数)，用于将卡方统计量标准化到 [0, 1] 区间。

---

### Block 7: 结果汇总与多重比较校正

```python
# Bonferroni 校正
bonf_threshold = 0.05 / total_n_features

# 双列判断：
# Significant_0.05: p < 0.05
# Significant_Bonf: p < bonf_threshold
```

Bonferroni 校正的数学原理：

$$P(\text{至少一次第 I 类错误}) = 1 - (1 - \alpha)^m$$

当 m = 22, α = 0.05 时：$$P \approx 1 - 0.95^{22} \approx 67.6\%$$

即在所有特征均无效的假设下，仍有 67.6% 的概率获得至少一个"显著"结果。Bonferroni 通过将 α 除以检验次数来控制 Family-Wise Error Rate。

三种常见校正策略对比：

| 方法 | 阈值 | 控制指标 | 严格程度 |
|------|------|---------|---------|
| Bonferroni | α/m | FWER | 最严格 |
| Holm | 逐步递降 | FWER | 中等 |
| Benjamini-Hochberg | FDR | 错误发现率 | 较宽松 |

---

### Block 8: 可视化

三张图的各自职责：

| 图 | 展示内容 | x 轴 | y 轴 |
|----|---------|------|------|
| P-value 柱状图 | 每个特征的统计显著性水平 | 特征名 | -log₁₀(p) |
| 效应量水平条形图 | 特征的实际差异大小 | 效应量值 | 特征名 |
| p 值 vs 效应量散点图 | 统计显著与实际显著的对应关系 | 效应量 | -log₁₀(p) |

散点图是本模块最重要的可视化：

- 右上角的点（高效应量 + 低 p 值）= 真正的强特征，如 Illiness.Code、Morphology.Description
- 左上角的点（低效应量 + 极低 p 值）= 统计显著但实际意义弱，如 Type.of.Death（效应量 0.017）
- 散点图揭示了"统计显著但实际不显著"的典型案例

---

## 实际分析结果

本数据集 22 个特征在 Bonferroni 校正后全部显著，但效应量差异极大：

| 等级 | 效应量范围 | 典型特征 | 预测价值 |
|------|-----------|---------|---------|
| 极大 | >0.5 | Morphology.Description (V=0.70), Illness.Code (V=0.64) | 作为核心特征使用 |
| 大 | 0.3-0.5 | Distant.metastasis (V=0.47), Code.of.Morphology (r=0.47) | 优先级加入模型 |
| 中 | 0.1-0.3 | Age (r=0.21), Raca.Color (V=0.16) | 可考虑保留 |
| 小 | <0.1 | Gender (V=0.09), Type.of.Death (V=0.017) | 对预测贡献极小 |

> 核心发现：所有特征均通过统计显著性检验，但效应量从 0.017 到 0.696 跨度超过 40 倍。仅凭 p 值筛选特征会将 Type.of.Death 与 Morphology.Description 等同视之，而实际上两者的预测价值有天壤之别。

---

## 关键收获

1. **统计检验方法是数据类型的函数。** 数值特征先判断正态性再选 T 检验或 Mann-Whitney U；分类特征构建列联表后用卡方检验。方法选错会导致结论不可靠。

2. **p 值同时衡量"差异大小"和"样本量"，效应量仅反映"差异大小"。** 在百万级数据集中，p 值失去了区分度——所有特征都显著。效应量才是评估特征预测潜力的正确指标。这是连接统计分析与机器学习的桥梁。

3. **多重比较校正是统计严谨性的底线。** 同时检验 22 个特征时，未校正的假阳性概率为 67.6%。在论文或正式分析中，Bonferroni/Holm/FDR 校正不可省略。

4. **统计显著不等于预测力强，但也不是无关。** 正确的策略不是二选一，而是结合两者：统计检验作为初筛（排除完全无关的特征），效应量作为排序依据（优先选择大效应量特征）。

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 所有特征 p 值均 < 0.001 | 大样本放大镜效应（N ≈ 210,000） | 关注效应量而非 p 值，用散点图同时展示两者 |
| 正态性检验对大样本"过于敏感" | Shapiro-Wilk 等检验在 N > 5000 时几乎总是拒绝 H₀ | 本代码的偏度优先策略（\|偏度\| < 0.5 直接接受）是一种务实方案 |
| 卡方检验中稀疏类别导致结果不稳定 | 期望频数 < 5 时卡方近似失效 | 预处理中过滤频数 < 5 的类别，或用 Fisher 精确检验 |
| 效应量解读标准不统一 | Cohen 的标准（0.2/0.5/0.8）针对心理学实验设计，医学文献可能有自己的阈值 | 以领域文献为参考，或使用效应量本身的相对比较 |
| Bonferroni 校正过于保守 | α/m 的调整方式对所有检验一视同仁 | 根据研究阶段选择：探索性分析用 FDR，确证性分析用 Bonferroni |

---

## 与其他模块的联系

- **前置模块**：Module 01 EDA — 提供数据的分布形态（偏度/峰度）信息，直接决定本模块的正态性判断和检验方法选择
- **后续模块**：
  - Module 03 缺失值处理 — 根据统计检验效应量排序，优先修复高效应量特征的缺失值
  - Module 05 特征选择 — 统计检验的效应量排序可作为特征选择的 baseline，后续与模型重要性交叉验证
  - Module 09 建模对比 — 比较"仅统计显著特征"与"全特征"的模型性能差异

---

