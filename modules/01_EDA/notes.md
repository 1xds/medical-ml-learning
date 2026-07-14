# Module 01 笔记: 探索性数据分析 (EDA)

> EDA 的核心目标：**检查数据质量 → 识别缺失模式 → 评估分布特征 → 检测异常值**。四个步骤完成后再进入建模阶段。

---

## 核心概念梳理

### 探索性数据分析 是什么？

在建模之前对原始数据进行系统性检查的过程。不涉及假设检验或预测，仅通过描述性统计和可视化手段理解数据的基本结构、质量问题和潜在模式。

### 为什么重要？（医学场景）

医学数据的特殊性决定了 EDA 不可跳过：

- 目标变量（如生存状态）若存在非随机缺失，模型将系统性低估死亡风险
- 录入错误（如年龄字段出现 `999`）会导致特征分布严重失真
- 类别严重不平衡（如 1000:10）会使模型通过预测多数类获得虚高准确率，但无临床价值

EDA 的质量直接决定后续特征工程和模型选择的可靠性。

---

## 代码精读

### Block 1: 依赖导入与环境配置

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

%matplotlib inline
warnings.filterwarnings('ignore')
```

| 库 | 用途 |
|----|------|
| `pandas` | 表格数据读取、清洗、聚合 |
| `numpy` | 向量化数值计算 |
| `matplotlib` | 底层绑图，控制画布、坐标轴、图例 |
| `seaborn` | 基于 matplotlib 的高级统计可视化封装 |
| `scipy.stats` | 偏度/峰度计算、统计检验 |

`%matplotlib inline` 是 IPython/Jupyter 魔法命令，将图表内嵌在 notebook 中而非弹出独立窗口。

---

### Block 2: 路径配置与数据加载

```python
BASE_DIR = "C:/Users/Lenovo/Downloads/"
DATA_PATH = os.path.join(BASE_DIR, "data", "cancer_data_eng.csv")
IMG_DIR = os.path.join(BASE_DIR, "img")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, low_memory=False, encoding='latin-1')
```

要点：
- `os.path.join` 跨平台拼接路径，避免 Windows/Linux 路径分隔符不一致
- `encoding='latin-1'` 针对意大利语癌症登记数据的特殊字符（UTF-8 无法解析）
- `low_memory=False` 要求 pandas 一次性读取全部数据推断 dtype，适用于需要全量列分析的场景
- `df.shape` 返回 `(样本数, 特征数)`

---

### Block 3: 目标变量构造

```python
df = df.dropna(subset=['Status.Vital']).reset_index(drop=True)
df['target'] = (df['Status.Vital'] == 'VIVO').astype(int)
```

关键细节：
- `dropna(subset=['Status.Vital'])` 只删除目标列缺失的行，其他列缺失不影响
- 必须先删缺失再创建 target：若先执行 `== 'VIVO'`，`NaN == 'VIVO'` 返回 `False`，缺失值会被错误标记为死亡（0）
- `reset_index(drop=True)` 删除缺失行后重建连续索引，避免后续 `iloc` 访问出错

---

### Block 4: 数据概况统计

```python
n_samples = len(df)
n_features = df.shape[1]

target_counts = df['target'].value_counts()
target_props = df['target'].value_counts(normalize=True) * 100

ratio = target_props.get(1, 0) / target_props.get(0, 0)
```

- `value_counts()` 返回各类别的频数
- `normalize=True` 将频数转换为比例（0~1），`* 100` 转为百分比
- `ratio` 用于评估类别平衡程度：≈1 表示平衡，越偏离 1 不平衡越严重

---

### Block 5: 标签分布可视化

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# 左侧：条形图（精确数值比较）
# 右侧：饼图（占比直观展示）
plt.savefig(...)
```

- `plt.subplots(1, 2)` 创建 1 行 × 2 列的子图布局
- 条形图基于长度编码，感知精度优于饼图的面积/角度编码（Cleveland & McGill, 1984）
- 饼图仅在类别 ≤3 且差异明显时适用

---

### Block 6: 缺失值分析

```python
missing_series = df.isnull().sum()
missing_pct = (missing_series / len(df)) * 100

missing_df = pd.DataFrame({
    'Column': missing_series.index,
    'Missing_Count': missing_series.values,
    'Missing_Pct': missing_pct.values
}).sort_values('Missing_Pct', ascending=False)
```

- `df.isnull()` 返回布尔型 DataFrame（True = 缺失），`.sum()` 按列统计缺失数
- 缺失率分档策略：<5% 可直接删除样本，5-20% 考虑插补，20-50% 需结合领域知识判断，>50% 通常直接删除特征
- 热力图的附加价值：识别列间缺失的共现模式——如两列总是一起缺失，暗示它们可能来自同一来源（如某类检查项目）

---

### Block 7: 分布分析

```python
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

skewness = data.skew()      # 偏度
kurtosis = data.kurtosis()  # 峰度
```

| 指标 | 含义 | 判断标准 |
|------|------|---------|
| 偏度 | 分布对称性 | 正偏 = 右尾更长；\|偏度\| > 1 需关注 |
| 峰度 | 尾部厚度 | 高峰度 = 厚尾，离群值风险更高 |
| KDE | 直方图的平滑近似 | 核密度估计，带宽决定平滑程度 |
| Q-Q 图 | 与理论正态分位数对比 | 散点落在对角线上 = 近似正态 |

`select_dtypes(include=[np.number])` 自动筛选数值型列，避免对分类变量执行无意义的统计计算。

---

### Block 8: 离群值检测

**方法一：IQR（四分位距法）**

```python
q1 = data.quantile(0.25)
q3 = data.quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
```

**方法二：Z-score**

```python
z_scores = np.abs((data - data.mean()) / data.std())
# |Z| > 3 视为离群值
```

两种方法的差异：

| | IQR | Z-score |
|----|-----|---------|
| 前提假设 | 无分布假设 | 数据近似正态 |
| 对偏态数据的表现 | 较稳健 | 可能误判（偏态下尾部 Z 值自然偏大） |
| 阈值 | 1.5 × IQR | \|Z\| > 3 |

选择策略：先通过 Block 7 的偏度和 Q-Q 图评估分布形态，偏态数据优先使用 IQR。

---

## 关键收获

1. **EDA 是建模的前提而非可选步骤。** 数据质量问题（缺失、不平衡、录入错误）若在 EDA 阶段未被发现，将传导至整个分析流程，使后续结论失去可靠性。

2. **缺失机制比缺失率更重要。** 医学数据中的缺失通常非随机（MNAR）——重症患者更易失访，死亡导致随访中断。简单删除或均值填充会引入系统性偏差，需结合临床背景判断处理方法。

3. **统计指标需结合业务背景解读。** IQR 与 Z-score 可能对同一数据点给出不同结论；偏度 > 1 不代表数据不可用（真实世界数据常呈长尾分布）。理解数据生成过程是选择分析方法的前提。

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `encoding='latin-1'` 缺失导致读取报错 | 非英文数据集的默认 UTF-8 编码无法解析特殊字符 | 先用 `open(file, 'rb').read(100)` 检查字节内容 |
| `NaN == 'MORTO'` 返回 `False` | `NaN` 与任何值的等值比较均返回 `False` | 使用 `.map()` 或 `pd.isna()` 显式处理缺失值 |
| 删除缺失行后索引不连续 | `dropna` 保留原始行号 | 立即执行 `.reset_index(drop=True)` |
| IQR 与 Z-score 结果矛盾 | Z-score 假设正态，偏态数据尾部点被误判 | 先评估偏度，偏态数据优先用 IQR |
| `plt.savefig` 后未释放画布 | matplotlib 默认保留 figure 对象 | 调用 `plt.close()` 或使用 `with` 上下文管理器 |

---

## 与其他模块的联系

- **前置模块**：无（EDA 是所有后续分析的起点）
- **后续模块**：
  - Module 02 统计检验 → 对 EDA 中观察到的组间差异（如存活 vs 死亡在年龄上的分布差异）进行统计显著性检验
  - Module 03 缺失值处理 → 基于 EDA 的缺失率分档决定删除或插补策略
  - Module 06 降维与聚类 → EDA 的分布特征决定选择 PCA（线性）还是 t-SNE/UMAP（非线性）
- **与研究工作的联系**：该流程适用于任何医学预测建模任务（生存分析、疾病分类、治疗反应预测）。掌握 EDA 方法论后可直接迁移至 EcMurJ 虚拟筛选的特征分析或宏基因组数据的质量评估。

---

## 参考资料

- 教程原文：`ml4health-main/jupyter/01_eda_detail_tutorial.ipynb`
- 数据集：`Downloads/data/cancer_data_eng.csv`（意大利癌症登记数据）
- McKinney, W. (2017). *Python for Data Analysis* (2nd ed.). O'Reilly. 第 5 章.
- Cleveland, W. S. & McGill, R. (1984). Graphical Perception: Theory, Experimentation, and Application to the Development of Graphical Methods. *Journal of the American Statistical Association*, 79(387), 531-554.
