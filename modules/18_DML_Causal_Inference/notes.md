# Module 18 笔记: 异质性处理效应 (HTE) — 双重机器学习 (DML)

> 本模块的核心命题：**如何在高维混淆变量的医学观察性研究中，利用双重机器学习 (DML) 估计癌症转移对生存的异质性因果效应 (CATE)，识别不同年龄子群体的差异化处理效应。**

---

## 核心概念梳理

### 异质性处理效应 (HTE) 与 CATE

在医学因果推断中，传统的平均处理效应 (ATE) 回答"处理（如癌症转移）对结局（如生存）的平均影响是什么"，这一全局数字掩盖了不同特征人群之间的效应差异。异质性处理效应 (Heterogeneous Treatment Effect, HTE)，亦称条件平均处理效应 (Conditional Average Treatment Effect, CATE)，定义为：

$$\text{CATE}(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]$$

其中 $Y(1)$ 和 $Y(0)$ 分别表示接受处理和未接受处理的潜在结局，$X$ 为异质性特征（如年龄）。CATE 的核心价值在于揭示"谁受益、谁受损、谁不受影响"的差异化图景，构成精准医学和个性化干预决策的理论基础。

### 双重机器学习 (DML) 的原理

在观察性研究中，处理分配非随机导致了经典的混淆问题：年龄同时影响转移概率和生存概率，直接回归会产生有偏估计。DML 通过"正交化 + 交叉拟合"的双重策略解决了这一问题。

**交叉拟合 (Cross-Fitting)** 是 DML 的核心创新。将数据分为 $K$ 折，对每一折用其余 $K-1$ 折数据训练两个骚扰参数模型（结果模型 $\mathbb{E}[Y|X,W]$ 和倾向模型 $P(T=1|X,W)$），然后在留出折上计算残差：

$$Y_{\text{res}} = Y - \hat{\mathbb{E}}[Y|X,W], \quad T_{\text{res}} = T - \hat{P}(T=1|X,W)$$

最后对残差做回归：$Y_{\text{res}} \sim T_{\text{res}}$ 得到 CATE。交叉拟合避免了过拟合偏差，而残差化去除了混淆因素的线性影响，使得 DML 具备半参数有效性——即在理论最优效率下进行估计，且对模型设定错误具有双重稳健性（只要结果模型或倾向模型至少一个正确，效应估计即一致）。

### 方法对比

| 方法 | 高维混淆鲁棒性 | 需指定函数形式 | 理论性质 | 典型应用 |
|------|--------------|-------------|---------|---------|
| 线性回归 | 弱 | 是 | 需线性假设 | 探索性分析 |
| 倾向得分匹配 | 中等 | 否 | 匹配偏差 | 观察性研究 |
| 工具变量 (IV) | 强 | 是 | 需有效工具 | 自然实验 |
| T-Learner | 中等 | 否 | 需额外校正 | 快速原型 |
| **DML** | **强** | **否** | **半参数有效** | **精准医学** |

### 本模块使用的模型

**DROrthoForest (Double Robust Orthogonal Random Forest)**：将正交化思想嵌入随机森林。每棵树在 Bootstrap 样本上训练，叶子节点内进行局部 DML 估计，最终聚合所有树的输出。其双重稳健性意味着即使骚扰参数模型存在偏误，CATE 估计仍保持渐进一致性。

**CausalForestDML (Causal Forest DML)**：因果森林与 DML 的结合。与普通随机森林的关键区别在于：分裂准则最大化处理效应的异质性（而非预测 MSE），叶子节点估计局部平均处理效应 (LATE)，并使用加权平均聚合。CausalForestDML 内置交叉拟合，训练更轻量，并提供置信区间和 p 值推断。

---

## 代码精读

### Block 1: 数据加载与预处理

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from econml.orf import DROrthoForest           # 双重稳健正交森林
from econml.dml import CausalForestDML          # 因果森林 DML
from econml.sklearn_extensions.linear_model import WeightedLasso
```

```python
df = pd.read_csv(DATA_PATH, low_memory=False, encoding='latin-1')
df['target'] = df['Status.Vital'].map({'VIVO': 1, 'MORTO': 0})  # 存活→1, 死亡→0

# 定义处理变量 T: METÁSTASE=1 (转移), LOCALIZADO=0 (局限)
df['T'] = np.nan
df.loc[df['Extension'] == 'LOCALIZADO', 'T'] = 0
df.loc[df['Extension'] == 'METÁSTASE',  'T'] = 1
df = df.dropna(subset=['T'])             # 删除 T 缺失的样本
```

**要点**：处理变量 T 使用二值编码，从原始分类变量 `Extension` 派生。存活状态映射为标准 0/1 编码。这是因果推断的标准数据预处理流程。

### Block 2: 特征定义与标准化

```python
feature_cols_X = ['Age']                    # 异质性特征: 效应随年龄变化
feature_cols_W = [
    'Gender', 'Raca.Color', 'year',         # 控制变量: 需控制其混淆效应
    'Laterality', 'Diagnostic.means'
]

# 类别特征 Label 编码, 数值特征 StandardScaler 标准化
cat_cols = [c for c in all_features if not pd.api.types.is_numeric_dtype(df_model[c])]
for col in cat_cols:
    df_model[col] = LabelEncoder().fit_transform(df_model[col].astype(str))

num_cols = [c for c in all_features if df_model[c].dtype in ['int64', 'float64']]
scaler = StandardScaler()
df_model[num_cols] = scaler.fit_transform(df_model[num_cols])
```

```python
# 子采样至 5000 条记录 (Bootstrap 多轮训练，样本量不能太大)
np.random.seed(RANDOM_STATE)
if len(df_model) > N_SAMPLES:
    idx = np.random.choice(len(df_model), N_SAMPLES, replace=False)
    df_model = df_model.iloc[idx].copy()
```

**要点**：严格区分异质性特征 X（作为 CATE 的自变量）和控制变量 W（作为混淆调整因子）。类别变量经 LabelEncoder 编码，数值变量经 StandardScaler 标准化以适应 L1 正则化模型。

### Block 3: 准备 DML 输入张量

```python
Y = df_model['target'].values               # 结局: 存活 (0/1)
T = df_model['T'].values                    # 处理: 是否转移 (0/1)
X = df_model[available_X].values            # 异质性特征: Age
W = df_model[available_W].values            # 控制变量: 5 个混淆因子

# 测试点: 在 Age 范围内均匀取 50 个点
age_min, age_max = X[:, 0].min(), X[:, 0].max()
X_test = np.linspace(age_min, age_max, 50).reshape(-1, 1)
```

**要点**：X_test 用于在年龄轴上绘制连续 CATE 曲线，而非仅输出单点估计。

### Block 4: DROrthoForest 训练

```python
# 计算自适应正则化参数 lambda
subsample_ratio = 0.3
lambda_reg = np.sqrt(np.log(n_features_W) / (10 * subsample_ratio * n_samples))

est = DROrthoForest(
    n_trees=100,                     # 树的数量
    min_leaf_size=10,                # 叶节点最小样本数
    max_depth=20,                    # 树的最大深度
    subsample_ratio=subsample_ratio, # 每棵树的子采样比例
    propensity_model=LogisticRegression(    # 倾向得分模型 (L1 正则)
        C=1/(X.shape[0]*lambda_reg), penalty='l1', solver='saga', max_iter=1000
    ),
    model_Y=Lasso(alpha=lambda_reg, max_iter=1000),  # 结果模型 (L1 正则)
    propensity_model_final=LogisticRegression(        # DML 最终阶段倾向模型
        C=1/(X.shape[0]*lambda_reg), penalty='l1', solver='saga', max_iter=1000
    ),
    model_Y_final=WeightedLasso(alpha=lambda_reg),    # DML 最终阶段结果模型
    n_jobs=1, random_state=RANDOM_STATE
)
est.fit(Y, T, X=X, W=W)

# 估计 CATE 和 95% 置信区间
treatment_effects = est.effect(X_test)
te_lower, te_upper = est.effect_interval(X_test, alpha=0.05)
```

**要点**：正则化参数 $\lambda$ 按 `sqrt(log(p) / (10 * subsample_ratio * n))` 自适应计算，这是高维稀疏假设下的理论推荐值。倾向模型和结果模型均使用 L1 正则化以处理高维 W。

### Block 5: CausalForestDML 训练

```python
est2 = CausalForestDML(
    model_y=Lasso(alpha=lambda_reg, max_iter=1000),   # 结果模型
    model_t=LogisticRegression(                        # 倾向模型
        C=1/(X.shape[0]*lambda_reg), penalty='l1', solver='saga', max_iter=1000
    ),
    n_estimators=100,               # 树的数量
    min_samples_leaf=10,            # 叶节点最小样本数
    max_depth=20,                   # 最大深度
    max_samples=subsample_ratio,    # 带放回子采样比例
    discrete_treatment=True,        # 二值处理变量
    n_jobs=1, random_state=RANDOM_STATE
)
est2.fit(Y, T, X=X, W=W, cache_values=True)  # cache_values 加速后续推断

treatment_effects2 = est2.effect(X_test)
te_lower2, te_upper2 = est2.effect_interval(X_test, alpha=0.05)
```

**要点**：`discrete_treatment=True` 告知模型 T 为二值变量，内部处理路径与连续 T 不同。`cache_values=True` 缓存中间结果以加速 effect_interval 调用。

### Block 6: 模型比较可视化

```python
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 左图: DROrthoForest
axes[0].plot(X_test[:, 0], treatment_effects, label='DROrthoForest CATE', color='tab:orange')
axes[0].fill_between(X_test[:, 0], te_lower, te_upper, color='tab:orange', alpha=0.25)
axes[0].set_xlabel("Age (standardized)")
axes[0].set_ylabel("Treatment Effect (CATE)")

# 右图: CausalForestDML
axes[1].plot(X_test[:, 0], treatment_effects2, label='CausalForestDML CATE', color='tab:green')
axes[1].fill_between(X_test[:, 0], te_lower2, te_upper2, color='tab:green', alpha=0.25)
axes[1].set_xlabel("Age (standardized)")

# 计算平均处理效应 (ATE) 作为汇总统计
ATE_dro = np.mean(treatment_effects)
ATE_cfd = np.mean(treatment_effects2)
```

**要点**：并排可视化便于直观比较两种模型的 CATE 曲线趋势和置信区间宽度，这是模型选择的重要诊断手段。

### Block 7: 恒定边际效应 vs 异质性效应推断

```python
# 第二个 CausalForestDML 实例，参数更细粒度用于推断
est_inf = CausalForestDML(
    cv=2, criterion='mse', n_estimators=400,
    min_var_fraction_leaf=0.1, min_var_leaf_on_val=True,
    discrete_treatment=False, n_jobs=1, random_state=123
)
est_inf.fit(Y, T, X=X, W=W)

# 4a. 恒定边际效应: 假设效应不随 X 变化
res_me = est_inf.const_marginal_effect_inference(X_test)
point_me = res_me.point_estimate                    # 常数点估计
lb_me, ub_me = res_me.conf_int(alpha=0.01)          # 99% 置信区间

# 4b. 异质性处理效应: CATE(X) 随 X 变化
res_cate = est_inf.effect_inference(X_test, T0=0, T1=1)
point_cate = res_cate.point_estimate                # 随 Age 变化的 CATE
lb_cate, ub_cate = res_cate.conf_int(alpha=0.01)
```

```python
# 可视化: 常数效应 (红色水平线) 叠加异质性效应 (虚线)
plt.plot(X_test[:, 0], point_me * np.ones_like(X_test[:, 0]), label='Const. Marginal Effect')
plt.fill_between(X_test[:, 0], lb_me, ub_me, alpha=0.2, label='99% CI')
plt.plot(X_test[:, 0], treatment_effects2, 'b--', label='CausalForestDML (Heterogeneous)')
```

**要点**：恒定边际效应假设 CATE 为常数（即不存在异质性），结果是一条水平线。而异质性效应允许 CATE 随年龄变化。比较两者可判断异质性是否显著——若异质性曲线落在常数效应的置信区间内，则异质性可能不显著。

### Block 8: K-Means 聚类发现效应子群体

```python
# 对训练集样本计算 CATE 及置信区间
point_orig = est2.effect(X)              # 每个样本的 CATE
lb_orig, ub_orig = est2.effect_interval(X, alpha=0.01)

# 构建聚类特征矩阵: [lb, ub, point, X, W]
total_frame = np.column_stack([lb_orig, ub_orig, point_orig, X, W])

# 肘部法则 (仅用 CATE 点估计)
for k in k_range:
    kmeans_pca = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans_pca.fit(point_orig.reshape(-1, 1))
    wcss.append(kmeans_pca.inertia_)

# CH 分数 (基于全部特征, 用于确定最优 K)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=1, n_init=10)
    kmeans.fit(total_frame)
    score = metrics.calinski_harabasz_score(total_frame, kmeans.labels_)
    hc_metrics.append(score)
```

```python
# 可视化最优聚类结果
for i in np.unique(best_labels):
    spots = np.where(best_labels == i)
    plt.scatter(
        total_frame[spots, 3],    # X = Age (标准化)
        total_frame[spots, 2],    # Y = CATE 点估计
        label=f'Cluster {i}', alpha=0.7, s=30
    )
```

**要点**：聚类特征矩阵包含 CATE 点估计、置信区间上下界和原始特征 X、W。肘部法则（仅用 CATE 估计值）和 Calinski-Harabasz 分数（用全部特征）双指标评估最优 K。最终聚类可视化横轴为年龄、纵轴为 CATE，各簇颜色区分——揭示不同年龄段的差异化处理效应模式。

---

## 关键收获

1. **DML 解决的核心问题**：在观察性研究中，当混淆变量既影响处理又影响结局时，DML 通过"正交化 + 交叉拟合"消除了混淆偏差，同时保持了半参数有效性。这一方法对高维混淆具有内在鲁棒性。

2. **CATE 超越 ATE 的临床价值**：ATE 提供单一数值（"转移使生存降低 X%"），CATE 揭示效应在不同子群体的分布（"年轻患者效应弱，老年患者效应强"），为筛选高危人群和差异化干预提供依据。

3. **双重稳健性 (Double Robustness)**：DROrthoForest 的关键理论保证——只要倾向模型或结果模型中至少一个正确，CATE 估计就是渐近无偏的。这在医学场景中尤为重要，因为模型设定错误几乎是不可避免的。

4. **聚类分析在因果推断中的角色**：基于 CATE 的 K-Means 聚类能够发现天然的子群体结构，将连续的处理效应空间划分为离散的风险组，输出可直接用于临床分层的患者分组方案。

5. **残差化的本质**：$Y_{\text{res}} = Y - \hat{\mathbb{E}}[Y|X,W]$ 和 $T_{\text{res}} = T - \hat{P}(T=1|X,W)$ 实质上通过投影去除了混淆因素 W 的线性效应，使得残差回归不受混淆干扰——这是半参数有效性的核心机制。

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| CATE 置信区间很宽 | 样本量不足或子群体稀疏 | 增大样本量；增大 `min_leaf_size` 减少过拟合；使用 BLB 自助法诚实反映不确定性 |
| ATE 为负但 CI 包含 0 | 平均效应可能为零，但存在子群体异质性 | 重点转向 CATE 分析；检查遗漏的 W 变量 |
| 两种模型 CATE 趋势矛盾 | 模型设定不当或数据噪声过大 | 检查超参数；增加交叉验证折数；评估数据质量 |
| `effect_interval` 不可用 | 部分 DROrthoForest 版本不支持 | 回退到正态近似：`mean ± 1.96 * std` |
| 聚类结果不稳定 | K-Means 对初始化和 K 选择敏感 | 多指标评估最优 K；使用 k-means++ 初始化；增加 `n_init` |

---

## 与其他模块的联系

- **前置模块**：Module 12-17 (SHAP 可解释性系列) — SHAP 回答"模型如何预测"，DML 回答"处理的因果效应是什么"。先理解预测模式再验证因果效应，两者互补。
- **后续模块**：Module 19-20 (影像组学) — 影像组学特征可用于构建更丰富的异质性特征矩阵 X，使 CATE 估计包含影像生物标记信息。
- **与研究工作的联系**：在 EcMurJ 虚拟筛选中，DML 可用于估计"特定分子特征对结合自由能的因果效应"，识别哪些化学子结构对蛋白互作有异质性影响。在宏基因组研究中，可估计"特定菌群丰度对宿主代谢指标的因果效应"，区分关联与因果。

---

## 参考资料

- 教程原文：`ml4health-main/jupyter/18_hte_dml.ipynb`
- 讲义：`ml4health-main/lectures/18_hte_dml_teaching_doc.md`
- Chernozhukov, V., et al. (2018). "Double/Debiased Machine Learning for Treatment and Structural Parameters." *The Econometrics Journal*, 21(1), C1-C68.
- Athey, S., & Imbens, G. (2016). "Recursive Partitioning for Heterogeneous Causal Effects." *PNAS*, 113(27), 7353-7360.
- Wager, S., & Athey, S. (2018). "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests." *JASA*, 113(523), 1228-1242.
- econml 文档：https://econml.azurewebsites.net/
