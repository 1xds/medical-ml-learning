# Module 22 笔记: CGM 连续血糖监测回归 — 分组数据的预测与评估

> 本模块的核心命题：**在带有分组结构（45 名受试者/1665 餐次）的非 i.i.d. 数据上，通过 4 种划分策略正确评估回归模型的泛化能力，并验证从分类到回归任务的完整迁移路径。**

---

## 核心概念梳理

### 从分类到回归的任务迁移

在 Module 01-21 全部为分类任务的基础上，本模块是第一个回归任务综合案例。回归与分类在 ML 流程上的核心差异集中于四点：

| 维度 | 分类 | 回归 |
|------|------|------|
| 目标类型 | 离散类别 (0/1) | 连续值 (iauc_2h ∈ [0, 14341] mg/dL·min) |
| 统计检验 | T检验/MW-U/ANOVA（组间） | Pearson/Spearman 相关分析 + FDR |
| 评估指标 | AUC/Recall/F1/Brier | MAE/RMSE/R²/Pearson r |
| 校准与诊断 | 校准曲线/DCA | 回归校准 (slope/intercept) + 残差诊断 (QQ/同方差) |

任务设定为预测餐后 2 小时增量血糖曲线下面积 (`iauc_2h`)，特征覆盖四个层次：餐次营养 (eff_carbs, eff_protein 等)、餐前 CGM 状态 (baseline_glucose, pre_glucose_slope_60 等)、可穿戴/时间信息 (meal_hour_sin/cos, prev_meal_gap_min) 和临床基线 (A1c, Fasting GLU, Age 等)。M3 主模型包含 31 个特征（29 数值 + 2 分类）。

### 分组数据与 i.i.d. 假设破坏

本案例的数据结构决定了其核心教学价值——1665 餐次并非独立同分布 (i.i.d.)，而是嵌套在 45 名受试者之下（中位数 38 餐/人，范围 17-63）。同一受试者的多餐次因代谢基线、饮食习惯和 CGM 传感器特性而高度相关。这一结构从根本上改变了交叉验证策略的选择：

**朴素 KFold 的泄漏机制**：随机划分会将同一受试者的餐次分配到训练集和测试集，模型通过"认出人"而非"预测血糖"即可获得高 R²。本案例实测显示，朴素 KFold R²=0.507，GroupKFold R²=0.203——R² 虚高 0.305。

### 4 种划分策略与对应问题

| 策略 | 实现 | 回答的问题 | 泄漏风险 | 典型 R² |
|------|------|----------|---------|---------|
| 朴素 KFold | `KFold(5, shuffle=True)` | （反面教材） | 高 | 0.507 (虚高) |
| GroupKFold | `GroupKFold(5, groups=subject_id)` | 能泛化到**新用户**吗？ | 无 | 0.203 (真实) |
| 受试者内时间 80/20 | 每人按 meal_time 排序，前 80% 训练/后 20% 测试 | 能预测某用户**未来餐次**吗？ | 无 | 0.430 |
| Few-shot k=0,1,3,5,10 | 训练=其他受试者+本人前 k 餐，测试=本人剩余 | 新用户需登录**多少餐**？ | 无 | k=10: R²=0.425 |

关键发现：时间划分 R² (0.430) > GroupKFold R² (0.203) — 说明个人基线信息贡献了约一半的预测力。Few-shot 曲线揭示 k=0（纯群体模型）MAE=1872 → k=10（10 餐个性化）MAE=1549，下降 17%。

### 回归评估指标体系

| 指标 | 公式 | 优点 | 缺点 |
|------|------|------|------|
| MAE | mean(\|pred - obs\|) | 抗异常值，与目标同单位 | 不惩罚大误差 |
| RMSE | sqrt(mean((pred-obs)²)) | 惩罚大误差 | 对异常值敏感 |
| R² | 1 - SS_res/SS_tot | 方差解释率 (0-1) | 可为负（模型比均值还差） |
| Pearson r | corr(pred, obs) | 衡量排序能力 | 只捕捉线性关系 |
| MAPE | mean(\|pred-obs\|/\|obs\|) | 无量纲可跨任务对比 | obs≈0 时不稳定 |

### Bootstrap 的不确定性量化

对于分组数据，正确的 Bootstrap 必须**重抽样受试者**而非餐次——重抽样餐次会因同人餐次相关而低估方差。subject-level bootstrap 通过对 45 个受试者做 500 次有放回重抽样，每次重算指标，取 2.5% 和 97.5% 分位数构成 95% 置信区间。

---

## 代码精读

### Block 1: 数据加载与目标 winsorize

```python
df = pd.read_csv(DATA_CSV)
df["meal_time"] = pd.to_datetime(df["meal_time"], errors="coerce")
df = df.dropna(subset=[PRIMARY_TARGET, "subject_id", "meal_time"])

# winsorize 1-99%: 限制极端值影响 (回归特有)
q_lo, q_hi = df[PRIMARY_TARGET].quantile([0.01, 0.99])
df = df[(df[PRIMARY_TARGET] >= q_lo) & (df[PRIMARY_TARGET] <= q_hi)]
df["subject_id"] = df["subject_id"].astype(int)
```

**要点**：winsorize 替代分类中的 SMOTE/重采样——回归中无类别不平衡概念，处理目标分布偏态的方法是将极端值钳制在 1-99% 分位数。

### Block 2: 泄漏审计

```python
FORBIDDEN_EXACT = {
    "calories", "carbs", "protein", "fat", "fiber",       # 原始营养，被 eff_* 替代
    "auc_2h", "iauc_2h", "peak_glucose_2h", ...            # 结果变量（目标!）
}
FORBIDDEN_REGEX = [r"fingerstick", r"contour", r"^auc_", r"^iauc_", ...]

def forbidden_reason(col):
    """判断列是否禁止进入模型"""
    c = str(col).strip().lower()
    if c in FORBIDDEN_EXACT:
        return "explicitly_forbidden"
    for pat in FORBIDDEN_REGEX:
        if re.search(pat, lower):
            return f"regex_forbidden:{pat}"
    return None
```

**要点**：泄漏审计混合了精确匹配和正则匹配双重策略，防止衍生变量（如 `iauc_2h`、`peak_delta_2h`）和同义变量（原始营养 vs eff_* 加工变量）被误输入模型。这是回归任务特有的安全措施——连续目标变量的衍生量极易被建模为"预测因子"。

### Block 3: 回归的防泄漏 Pipeline

```python
def make_pipeline(model, numeric_features, categorical_features):
    """每个 fold 内重新 fit (imputer/scaler 只见本 fold 训练数据)"""
    transformers = []
    if numeric_features:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),   # 中位数插补抗异常值
            ("scaler", StandardScaler())]), numeric_features))
    if categorical_features:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder())]), categorical_features))
    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)
    return Pipeline(steps=[("preprocess", pre), ("model", clone(model))])
```

**要点**：与分类任务完全相同的防泄漏 Pipeline 架构，但数值插补策略从均值改为中位数（`strategy="median"`）——回归目标受异常值影响更大，中位数比均值更鲁棒。

### Block 4: 4 种划分策略实现 (核心)

```python
# ① 朴素 KFold — 反面教材
def split_naive_kfold_oof(df, target, spec, model):
    features = spec["numeric"] + spec["categorical"]
    X, y = df[features], df[target].values
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in kf.split(X):
        pipe = make_pipeline(model, spec["numeric"], spec["categorical"])
        pipe.fit(X.iloc[tr], y[tr])
        pred = pipe.predict(X.iloc[te])
        # 记录每餐次的预测

# ② GroupKFold — 训练/测试受试者不交叉 (诚实泛化)
def split_groupkfold_oof(df, target, spec, model):
    groups = df["subject_id"].values
    gkf = GroupKFold(n_splits=min(n_splits, df["subject_id"].nunique()))
    for tr, te in gkf.split(X, y, groups):
        # tr 和 te 中不含相同 subject_id

# ③ 受试者内时间 holdout — 按 meal_time 排序
def split_temporal_holdout(df, target, spec, model):
    data = df.sort_values(["subject_id", "meal_time"])
    for sid, sub in data.groupby("subject_id"):
        n = len(idx); cut = max(1, min(int(n*0.8), n - 1))
        train_idx.extend(idx[:cut]); test_idx.extend(idx[cut:])

# ④ Few-shot — 训练=其他受试者+本人前k餐, 测试=本人剩余
def split_fewshot(df, target, spec, model, shots_list=(0,1,3,5,10)):
    for k in shots_list:
        for sid, sub in data.groupby("subject_id"):
            adapt, test = sub.iloc[:k], sub.iloc[k:]
            train_base = data[data["subject_id"] != sid]
            train = pd.concat([train_base, adapt]) if k > 0 else train_base
```

**要点**：GroupKFold 通过 `groups=subject_id` 确保同一受试者的所有餐次要么全在训练集、要么全在测试集。时间划分依赖 `meal_time` 排序，模拟真实部署场景（用过去预测未来）。Few-shot 模拟冷启动——新用户只有 k 餐历史时，个性化预测能做到多好。

### Block 5: subject-level Bootstrap

```python
def bootstrap_ci(pred_df, n_bootstrap=500):
    """重抽样受试者 (而非餐次) → 重算指标 → 95% CI"""
    subjects = np.unique(pred_df["subject_id"].values)
    samples = {"MAE": [], "RMSE": [], "R2": [], "Pearson_r": []}
    for _ in range(n_bootstrap):
        resampled = rng.choice(subjects, size=len(subjects), replace=True)
        mask = pred_df["subject_id"].isin(resampled).values
        m = evaluate_predictions(pred_df["y_true"][mask], pred_df["y_pred"][mask])
        for k in samples: samples[k].append(m[k])
    ci = {}
    for k, vals in samples.items():
        vals = np.array(vals)
        ci[k] = (float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)))
    return base, ci
```

**要点**：正确的不确定性量化对于分组数据至关重要——重抽样单元必须是受试者（独立单元），而非餐次（相关单元）。若错误地对餐次做 Bootstrap，置信区间会被严重低估。

### Block 6: LASSO Signature 在回归中的构建

```python
# 在受试者内时间划分的训练集上做 LASSO (防泄漏)
lasso_cv = LassoCV(cv=5, n_alphas=100, max_iter=20000, random_state=RANDOM_STATE)
lasso_cv.fit(X_l, y_l)
sig_features = lasso_coefs[lasso_coefs != 0].sort_values(key=abs, ascending=False)
# 13 个非零系数: Fasting GLU (+460), baseline_glucose (−413), A1c (+365)...
```

**要点**：回归 LASSO 使用 `LassoCV`（而非分类的 `LogisticRegressionCV` 带 `penalty='l1'`），自动选择 alpha。signature 中 `baseline_glucose` 系数为负说明了"回归到均值"效应——餐前血糖越高，餐后增量越小。

### Block 7: 11 Regressor 对比与消融

```python
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1, max_iter=10000),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
    "KNN": KNeighborsRegressor(n_neighbors=15),
    "SVR": SVR(C=1.0, kernel="rbf"),
    "DecisionTree": DecisionTreeRegressor(max_depth=6),
    "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3),
    "HGB": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03),
    "XGBoost": xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05),
    "LightGBM": lgb.LGBMRegressor(n_estimators=400, max_depth=4, learning_rate=0.05),
}
```

```python
# 消融 M0→M4 (HGB, GroupKFold R²)
# M0 (仅餐次营养) R²=0.066 → M1 (+餐前CGM) R²=0.137 → M3 (+临床) R²=0.203
# M4 (+肠道) R²=0.134 — 反而降低: 肠道评分在 GroupKFold 下过拟合
```

**要点**：线性模型在高度共线性的临床特征（LDL/HDL/Cholesterol/Cho-HDL Ratio 同时存在）下严重过拟合——LinearRegression R²=-121, Ridge R²=-59。树模型对共线性天然鲁棒，HGB 为最佳。消融结果揭示了两层意义：M0→M1 R² 翻倍（餐前 CGM 是强信号），M4 降说明受试者级特征（肠道评分）在 GroupKFold（从未见过该受试者）下无预测价值。

### Block 8: 回归校准与残差诊断

```python
# 回归校准: 观测 = intercept + slope × 预测
slope, intercept = np.polyfit(y_pred, y_true, 1)
# slope=0.735 (<1: 预测被压缩, 回归到均值)
# intercept=+743 (>0: 低值高估, 高值低估)
sd_ratio = np.std(y_pred) / np.std(y_true)  # 0.657: 预测方差小于真实方差

# 残差诊断 4-panel:
# (i) 残差 vs 预测 — 漏斗形 → 异方差 (高值预测的误差更大)
# (ii) Q-Q 图 — 两端偏离 → 重尾非正态 (置信区间不可靠)
# (iii) 残差直方图 + KDE — 左偏 (偏度=−0.72)
# (iv) Scale-Location — 趋势上升 → 确认异方差
stats.probplot(resid, dist="norm", plot=axes)
```

**要点**：回归校准的 slope 和 intercept 揭示模型"保守"——对极端高血糖反应预测不足，所有预测被拉向均值。这是小样本 + GroupKFold 的典型现象（训练集未见过极端受试者）。残差非正态意味着报告置信区间时必须用 Bootstrap 而非正态假设。

### Block 9: SHAP 在回归中的单位含义

```python
shap_values = explainer.shap_values(X_shap)
# SHAP 值单位 = mg/dL·min (目标单位, 非 log-odds!)

# 全局重要性
global_imp = np.abs(sv).mean(axis=0)

# SHAP Top 特征:
# 1. eff_carbs (|SHAP|=714.7) — 碳水越高 → iauc 越高
# 2. Fasting GLU (|SHAP|=488.5) — 空腹血糖越高 → iauc 越高
# 3. prev_meal_gap_min (|SHAP|=366.1) — 距上餐越久 → iauc 越高
# 4. A1c (|SHAP|=356.7) — 糖化越高 → iauc 越高
# 5. baseline_glucose (|SHAP|=330.1) — 餐前血糖越高 → 增量越小!
# 8. eff_protein (|SHAP|=218.2) — 蛋白质 → 降低 iauc (延缓吸收)
# 10. eff_fiber (|SHAP|=147.6) — 纤维 → 降低 iauc (延缓吸收)
```

**要点**：回归 SHAP 值的单位就是目标单位（mg/dL·min），而非分类中的 log-odds。这使得 SHAP 值具有直接的生理学解释："eff_carbs 将 iauc_2h 预测推高约 714.7 mg/dL·min"。LASSO signature 与 SHAP Top 特征高度一致——线性与非线性方法指向同一组驱动特征。

---

## 关键收获

1. **分组数据必须用 GroupKFold，朴素 KFold 的 R² 虚高可达 0.3**：这是本模块的核心教训。只要数据存在分组结构（同人多样本/同医院多患者/同设备多测量），随机 KFold 就会将同组样本分散到训练集和测试集，导致模型通过"识别分组"而非"学习任务"来获得虚高的评估指标。

2. **划分策略决定回答什么问题**：GroupKFold 回答"对新用户如何"（最诚实的泛化），时间划分回答"对老用户未来餐次如何"（实际应用价值），few-shot 回答"需要多少个性化数据"（冷启动产品决策）。不同策略回答不同问题，不存在"最佳"划分。

3. **回归校准的 slope < 1 揭示了模型的保守性**：slope=0.735 意味着预测被系统性压缩——高血糖反应预测不足，低血糖反应预测过高。这是模型"回归到均值"的体现，在样本量不足以覆盖全部人群变异时是普遍现象。

4. **消融研究的阶梯式信息增益揭示了特征层次价值**：M0→M1 R² 翻倍（餐前 CGM 状态是强信号），M3 达峰值（临床基线决定跨受试者泛化），M4 反而下降（受试者级特征在 GroupKFold 下过拟合）——这一模式对任何分组数据集的特征工程都具有指导意义。

5. **Bootstrap 的重抽样单元必须是独立单元**：在分组数据中，独立性发生在受试者层而非餐次层。在餐次层做 Bootstrap 会因同一受试者餐次相关而低估方差——这一原则同样适用于住院患者数据（患者是独立单元）、设备测量数据（设备是独立单元）等场景。

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 朴素 KFold R² 远高于 GroupKFold | 同受试者餐次泄漏 | 分组数据必须用 GroupKFold |
| 线性模型 R² 为负 | 共线性（LDL/HDL/Cholesterol 同时存在） | 使用树模型或先做相关性去冗余 |
| log1p 变换未改善 R² | HGB 基于排序的树模型对单调变换不敏感 | 不要默认 log 变换，需实测对比 |
| 残差呈漏斗形 (异方差) | 高值预测的方差更大 | log1p 目标/WLS 加权回归/分区报告 CI |
| SHAP 值过大 (>1000) | 目标值域大 (0-14341 mg/dL·min) | 正常现象，解释时使用相对指标 |

---

## 与其他模块的联系

- **前置模块**：Module 01-12 (分类 ML 流程) — 本模块直接复用其骨架，将 Classifier 替换为 Regressor；Module 20/21 (影像组学/基因组学) — 与本模块并列构成"分类→回归"任务迁移的主线
- **后续模块**：无（本教程体系的终点模块）— 标志着从"学方法"到"用方法"的过渡
- **与研究工作的联系**：CGM 回归的分组数据划分策略可直接迁移至 EcMurJ 虚拟筛选中的"同蛋白多配体"场景——同一靶蛋白的多个配体互相相关，应采用 GroupKFold (按蛋白分组) 评估模型泛化能力。Few-shot 个性化概念可应用于宏基因组研究中"新样本的宿主表型预测"——少量个性化菌群数据能多大程度提升预测精度。

---

## 参考资料

- 教程原文：`ml4health-main/jupyter/22_regression_cgm_pipeline.ipynb`
- 讲义：`ml4health-main/lectures/22_regression_cgm_teaching_doc.md`
- Saeb, S., et al. (2017). "The need to approximate the use-case in clinical machine learning." *GigaScience*, 6(5), 1-9. — 讨论分组数据划分策略的关键论文
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed. — 回归模型与正则化的经典教材
- Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*, 4765-4774.
- Zeevi, D., et al. (2015). "Personalized Nutrition by Prediction of Glycemic Responses." *Cell*, 163(5), 1079-1094. — CGM 个性化血糖预测的开创性工作
