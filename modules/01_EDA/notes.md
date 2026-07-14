# Module 01 笔记: 探索性数据分析 (EDA)

> 不管代码多长，EDA 就是四件事：**看看数据长什么样 → 有没有缺失 → 分布正不正常 → 有没有异常值**。

---

## 核心概念梳理

### 探索性数据分析 是什么？

==「建模之前，先把数据翻一遍看看」的工作。==

就像你去菜市场买菜，不会看也不看就下锅。得先看看菜新不新鲜（缺失值）、有几斤几两（样本量）、有没有烂叶子（异常值）。EDA 就是数据版的"买菜前先挑挑"。

### 为什么重要？（医学场景）

在医学数据里尤其是——你的模型再怎么调参，如果训练数据本身就有问题，结果一定不可信。比如：

- 如果"死亡"标签大量缺失，你训出来的模型就不知道"什么样的人会死"
- 如果年龄里混入了 999 岁的录入错误，模型会被带歪
- 如果数据极度不平衡（1000 个存活 vs 10 个死亡），模型会偷懒直接猜"存活"，准确率 99% 但没有临床价值

所以 EDA 不是可有可无的步骤，而是**决定后续所有工作有没有意义**的起点。

---

## 代码精读

> 下面是整个 notebook 的分段解读。每个 block 我都用最直白的话解释了"这段在干什么"。

### Block 1: 导包和设置

```python
import pandas as pd          # 处理表格数据的"瑞士军刀"
import numpy as np           # 数学计算
import matplotlib.pyplot as plt  # 画图的"基础画笔"
import seaborn as sns        # 比 matplotlib 更好看的高级画图工具
from scipy import stats      # 统计检验用的

%matplotlib inline           # 让图直接显示在 notebook 里，不用弹窗
warnings.filterwarnings('ignore')  # 关掉烦人的警告信息
```

**我的理解**：这些 import 就像一个工具箱开箱——pandas 处理数据，matplotlib/seaborn 画图，numpy 做计算。`%matplotlib inline` 是 jupyter 特有的魔法命令。

---

### Block 2: 数据加载

```python
BASE_DIR = "C:/Users/Lenovo/Downloads/"                             # 数据在哪个文件夹
DATA_PATH = os.path.join(BASE_DIR, "data", "cancer_data_eng.csv")   # 拼接出完整路径
IMG_DIR = os.path.join(BASE_DIR, "img")                             # 图片保存到哪
RESULTS_DIR = os.path.join(BASE_DIR, "results")                     # 结果保存到哪

os.makedirs(IMG_DIR, exist_ok=True)     # 如果文件夹不存在就创建
os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, low_memory=False, encoding='latin-1')  # 读取 CSV
```

**我的理解**：
- `os.path.join` 是用来拼路径的，比直接写 `"C:/.../data/file.csv"` 更安全，不会因为斜杠方向出错
- `encoding='latin-1'` 是因为这个数据集是意大利语癌症登记数据，有些特殊字符 UTF-8 读不了
- `low_memory=False` 告诉 pandas "一次读完，别分批"，因为我们要对所有列做分析
- `df.shape` 返回 `(行数, 列数)`

---

### Block 3: 处理目标变量

```python
# 删除 Status.Vital 为空的样本（不知道生死的病人没法用）
df = df.dropna(subset=['Status.Vital']).reset_index(drop=True)

# 创建一个新列 'target': VIVO(存活)=1, 其他(死亡)=0
df['target'] = (df['Status.Vital'] == 'VIVO').astype(int)
```

**我的理解**：
- `dropna(subset=['Status.Vital'])`：只删除"Status.Vital 这一列为空"的行，其他列有空没关系
- `(df['Status.Vital'] == 'VIVO')` 返回 True/False，`.astype(int)` 把它转成 1/0
- `reset_index(drop=True)`：删完行后索引会乱，重新从 0 排一遍
- 为什么先删缺失再创建 target？因为如果先创建 target，缺失值也会被变成 0（False），等于把"未知"当成了"死亡"，这是错的

---

### Block 4: 数据概况统计

```python
n_samples = len(df)            # 有多少个病人
n_features = df.shape[1]       # 有多少个特征

target_counts = df['target'].value_counts()          # 各有多少人
target_props = df['target'].value_counts(normalize=True) * 100  # 各占多少%

ratio = target_props.get(1, 0) / target_props.get(0, 0)  # 存活/死亡比例
```

**我的理解**：
- `value_counts()` 就是"数一数每个值出现了多少次"——好比点人数
- `normalize=True` 把人数变成比例（0 到 1 之间），`* 100` 变成百分数
- 最后用 ratio 判断数据是否平衡：接近 1 就是差不多，远离 1 就是偏了

---

### Block 5: 画标签分布图（条形图 + 饼图）

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 一行两列的画布
# 左边画条形图，右边画饼图
# ...画图细节省略...
plt.savefig(...)  # 保存为 PNG
```

**我的理解**：
- `plt.subplots(1, 2)` = 创建 1 行 × 2 列的子图（左右并排）
- 画图分三步：创建画布 → 往上画东西 → 保存/显示
- 条形图更适合看精确数字，饼图适合看占比

---

### Block 6: 缺失值分析

```python
missing_series = df.isnull().sum()               # 每列有多少空值
missing_pct = (missing_series / len(df)) * 100    # 转成百分比

missing_df = pd.DataFrame({
    'Column': missing_series.index,      # 列名
    'Missing_Count': missing_series.values,  # 缺失多少个
    'Missing_Pct': missing_pct.values        # 缺失百分比
}).sort_values('Missing_Pct', ascending=False)  # 按缺失率从高到低排
```

**我的理解**：
- `df.isnull()` 把整个表变成 True(空)/False(不空)，`.sum()` 数每一列有几个 True
- 有个非常容易踩坑的点：`.isnull()` 和 `.isna()` 是一回事，pandas 作者故意留了两个名字
- 缺失值分档（<5% / 5-20% / 20-50% / >50%）是为了快速判断哪些列可以直接删，哪些需要想办法填补

**缺失值热力图的关键**：颜色块代表某些列的缺失是否有"同步性"——比如如果 A 列和 B 列总是一起缺失，说明它们可能来自同一份检查单

---

### Block 7: 分布分析（直方图 + KDE + Q-Q 图）

```python
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# 挑出所有数值列，排除 target 和 ID

skewness = data.skew()    # 偏度：分布是不是歪的
kurtosis = data.kurtosis() # 峰度：数据集中在中间还是散在两头
```

**我的理解**：
- `select_dtypes(include=[np.number])`：自动识别哪些列是数字
- **偏度**：正数 = 尾巴拖在右边（多数人聚集在左边），负数 = 尾巴拖在左边。绝对值 > 1 说明歪得厉害
- **KDE（核密度估计）**：可以理解为直方图的"光滑版"，用曲线代替柱子展示分布形状
- **Q-Q 图**：如果数据是正态分布，散点会落在对角线上；偏离对角线 = 不是正态

---

### Block 8: 离群值分析（IQR + Z-score）

```python
q1 = data.quantile(0.25)    # 第 25 百分位
q3 = data.quantile(0.75)    # 第 75 百分位
iqr = q3 - q1               # 四分位距
lower = q1 - 1.5 * iqr       # 下界：比 Q1 小 1.5 倍 IQR 就是异常
upper = q3 + 1.5 * iqr       # 上界：比 Q3 大 1.5 倍 IQR 就是异常
```

```python
z_scores = np.abs((data - data.mean()) / data.std())  # 标准化后取绝对值
# Z > 3 就算异常（偏离均值超过 3 个标准差）
```

**我的理解**：
- **IQR 方法**：不看数据长什么样，纯靠"排名"来判断——适合数据本身就不是正态的情况
- **Z-score 方法**：假设数据是正态的，看每个点离"中心"有多远——适合数据接近正态
- 两个方法结果可能不一样：IQR 对偏态数据更宽容，Z-score 对偏态数据更敏感
- 箱线图里的"小圆点"就是 IQR 判定为异常的点

---

## 今天最大的收获

1. **EDA 不是走过场，是建模的保险丝。** 数据如果本身就有坑（大量缺失、极度不平衡、录入错误），后面调再多参数也没用。就像盖房子地基歪了，装修再好也会塌。

2. **"缺失"比"有值"更值得关注。** 在医学数据里，缺失往往不是随机的——重症患者更容易失访，死亡的人没有随访记录。删缺失数据前要先想清楚"为什么缺"。

3. **不要盲目相信统计指标。** IQR 和 Z-score 可能给出不一样的离群值结论；偏度 > 1 不代表数据不能用，很多真实世界数据天然就是长尾的。理解数据的业务背景比套公式更重要。

---

## 容易踩坑

| 坑 | 为什么容易踩 | 怎么避免 |
|----|-------------|---------|
| **`encoding='latin-1'` 忘了加** | 中文/特殊字符数据集用 UTF-8 能读，但这个意大利癌症数据不行 | 先 `open(file, 'rb').read(100)` 看看有没有乱码字节 |
| **`NaN == 'MORTO'` 返回 False** | 人直觉以为"不等于 VIVO 就是死亡"，但 `NaN == 任何东西` 都是 False | 用 `.map()` 或 `pd.isna()` 显式处理缺失值 |
| **删缺失后忘记 `reset_index`** | 索引断掉后后面用 `df.iloc[i]` 会出错 | 删行后立刻 `.reset_index(drop=True)` |
| **IQR 和 Z-score 混用** | 以为两个方法等价，但实际上 Z-score 假设正态，偏态数据会被误杀 | 先看分布形状，再选方法：偏态用 IQR，正态用 Z-score |
| **`plt.savefig` 后不关图** | Jupyter 内存里积压几百张图没释放 | 用 `plt.close()` 或在不需要显示时注释掉 `plt.show()` |

---

## 与其他模块的联系

- **前置模块**：无（这是第一个模块，所有后续分析都依赖 EDA 的结论）
- **后续模块**：
  - Module 02 统计检验 → EDA 里发现的"存活 vs 死亡在年龄上有差异吗"需要统计检验来确认
  - Module 03 缺失值处理 → EDA 告诉我们哪些列缺失严重，后续要决定删还是补
  - Module 06 降维/聚类 → EDA 的分布分析告诉我们数据大概长什么样，决定用 PCA 还是 t-SNE
- **与我的研究的联系**：做任何医学预测模型（生存预测、疾病分类、治疗反应预测），第一步永远是 EDA。看懂这个 notebook，后面所有项目的数据分析流程都能套用。

---

## 参考资料

- 教程原文：`ml4health-main/jupyter/01_eda_detail_tutorial.ipynb`
- 数据集：`Downloads/data/cancer_data_eng.csv`（意大利癌症登记数据）
- 推荐阅读：《Python for Data Analysis》(Wes McKinney) 第 5 章
- Cleveland & McGill (1984): Graphical Perception 经典论文（解释了为什么条形图比饼图好）
