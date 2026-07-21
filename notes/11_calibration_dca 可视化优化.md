
> 基于图 14c 和 14d 的分析与改进

---

## 一、Hosmer-Lemeshow 检验柱状图 (图14c)

### 1.1 原始代码

```python
# --- 图 3: 校准度直方图 ---
fig, ax = plt.subplots(figsize=(10, 6))
names_hl = list(hl_results.keys())
chi2s = [hl_results[n]['chi2'] for n in names_hl]
ps = [hl_results[n]['p'] for n in names_hl]
colors_hl = ['#2ecc71' if p > 0.05 else '#e74c3c' for p in ps]
bars = ax.bar(range(len(names_hl)), chi2s, color=colors_hl,
              edgecolor='white', width=0.5)
ax.axhline(y=15.507, color='red', linestyle='--', linewidth=1.5,
           label='χ² critical (df=8, α=0.05)')
ax.set_xticks(range(len(names_hl)))
ax.set_xticklabels(names_hl, rotation=15, ha='right', fontsize=9)
ax.set_ylabel('Hosmer-Lemeshow χ²', fontsize=11)
ax.set_title('Hosmer-Lemeshow Test: Model Calibration Assessment',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, p in zip(bars, ps):
    status = 'Good' if p > 0.05 else 'Poor'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'p={p:.4f}\n({status})', ha='center', va='bottom',
            fontsize=8, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "14c_hl_test.png"), dpi=150,
            bbox_inches='tight')
```

### 1.2 原始输出数据

| 模型 | χ² | p | 结论 |
|------|-----|---|------|
| Logistic Regression | 111.46 | 0.0000 | 校准不良 |
| Random Forest | 74.64 | 0.0000 | 校准不良 |
| XGBoost | 54.74 | 0.0000 | 校准不良 |
| KNN (k=15) | 3.6×10¹¹ | 0.0000 | 校准不良 |

### 1.3 问题

| 问题 | 具体表现 | 后果 |
|------|---------|------|
| **线性坐标失真** | KNN 的 χ² = 3600亿，其他模型仅 ~50-110 | KNN 的柱子将其他 3 个模型压扁到肉眼不可见 |
| **信息粒度丢失** | 模型间 χ² 差异跨越 6 个数量级 (10¹~10¹¹) | 读者无法感知 LR (111) 与 KNN (3600亿) 的天壤之别 |
| **标签重叠** | 柱顶标注 p=0.0000 重复 4 次 | 每个柱子顶部的 p 值完全一样，增加视觉噪音 |

### 1.4 优化方案

---

#### 方案A：对数坐标变换

**思路**：将原始 χ² 映射到 log₁₀ 尺度，压缩极端值，使模型间差异在同一量级可见。

**改后代码**：

```python
# --- 图 14c 优化版: 对数坐标 Hosmer-Lemeshow ---
fig, ax = plt.subplots(figsize=(10, 6))
names_hl = list(hl_results.keys())

# 对数变换: log10(χ²)
chi2s_raw = np.array([hl_results[n]['chi2'] for n in names_hl])
chi2s_log = np.log10(np.maximum(chi2s_raw, 1))  # 防止 log(0)
ps = [hl_results[n]['p'] for n in names_hl]

# 着色方案: 用颜色深浅表示 χ² 大小，不再只用红/绿二值
# 色阶映射: log10(χ²) 范围 [1.7, 11.6]，映射到蓝色色阶
norm = plt.Normalize(vmin=1.5, vmax=12)
cmap = plt.cm.Blues
colors_log = cmap(norm(chi2s_log))

bars = ax.bar(range(len(names_hl)), chi2s_log, color=colors_log,
              edgecolor='gray', linewidth=0.5, width=0.55)

# 标注原始 χ² 值（用科学计数法显示）
for i, (bar, raw, logv) in enumerate(zip(bars, chi2s_raw, chi2s_log)):
    if raw > 1e6:
        label = f'chi2 = {raw:.2e}\nlog10 = {logv:.2f}'
    else:
        label = f'chi2 = {raw:.2f}\nlog10 = {logv:.2f}'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            label, ha='center', va='bottom', fontsize=8)

ax.set_xticks(range(len(names_hl)))
ax.set_xticklabels(names_hl, rotation=15, ha='right', fontsize=10)
ax.set_ylabel('log10(chi2)', fontsize=12)
ax.set_xlabel('Model', fontsize=12)
ax.set_title('Hosmer-Lemeshow Test (Log Scale):\nModel Calibration Disparity Revealed',
             fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加色阶 colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02, shrink=0.6)
cbar.set_label('log10(chi2)', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "14c_hl_test_logscale.png"),
            dpi=150, bbox_inches='tight')
```

**合理性分析**：

| 维度 | 说明 |
|------|------|
| **统计学依据** | HL 检验的 χ² 统计量在零假设下服从 χ² 分布，其值域无上界。当样本量大（N=6000）时，即使微小偏差也会产生巨大 χ²。对数变换是处理右偏分布的标准做法。 |
| **信息保留** | 柱顶同时标注原始 χ²（科学计数法）和 log₁₀ 值，不丢失任何定量信息。 |
| **局限性** | log₁₀(χ²) 不是临床医生熟悉的指标，需要在讲解时说明"相差 1 个单位 ≈ χ² 差 10 倍"。 |

---

#### 方案B：用 Brier Murphy 分解替代 HL 检验柱状图

**思路**：代码中已计算了 Brier Score 的 Murphy 三分解，将其可视化为**堆叠柱状图**比 HL 检验更具信息量。

**Murphy 分解公式**：

```
Brier Score = Refinement（鉴别力）− Calibration（校准度）+ Uncertainty（不确定性）
```

等价形式：柱子总高度 = Refinement + Calibration + Uncertainty，其中：
- Calibration（红色）：模型校准误差，越小越好
- Refinement（绿色）：模型鉴别力贡献（理想情况下 Refinement = Uncertainty），越高越好
- Uncertainty（灰色斜线）：固有不确定性 ȳ(1−ȳ)，由患病率决定，**模型无法改变**

三个成分全堆叠，柱子总高度 = Ref + Cal + Unc，柱顶标注 Brier Score。

**改后代码**：

```python
# --- 图 14c 替代方案: Brier Murphy 三成分完整堆叠图 ---
# 堆叠顺序: 底部 = Uncertainty (固定基线) → 中部 = Calibration → 顶部 = Refinement

fig, ax = plt.subplots(figsize=(10, 6))

models_list = list(calibration_data.keys())
n_models = len(models_list)

# 从 brier_decomposition() 获取三成分
refinement = []
calibration = []
uncertainty_arr = []

for name in models_list:
    ref, cal, unc = brier_decomposition(y_te, calibration_data[name]['y_prob'])
    refinement.append(ref)
    calibration.append(cal)
    uncertainty_arr.append(unc)  # 通常 = ȳ(1-ȳ) ≈ 0.2422，所有模型相同

x = np.arange(n_models)
width = 0.5

# ① 底层: Uncertainty（灰斜线）—— 固有不确定性，作为"基线"显示在最下方
uncertainty_display = uncertainty_arr[0]  # 用第一个模型的值（理论上都相同）
bars_unc = ax.bar(x, uncertainty_arr, width,
                  label=f'Uncertainty (固有不确定性) = {uncertainty_display:.4f}',
                  color='#95a5a6', edgecolor='white', linewidth=0.5,
                  alpha=0.6, hatch='//')

# ② 中层: Calibration（红）—— 校准误差，在 Uncertainty 之上
bars_cal = ax.bar(x, calibration, width,
                  bottom=uncertainty_arr,
                  label='Calibration (校准误差)',
                  color='#e74c3c', edgecolor='white', linewidth=0.5)

# ③ 顶层: Refinement（绿）—— 鉴别力贡献，在最上方
bars_ref = ax.bar(x, refinement, width,
                  bottom=[u + c for u, c in zip(uncertainty_arr, calibration)],
                  label='Refinement (鉴别力贡献)',
                  color='#2ecc71', edgecolor='white', linewidth=0.5)

# 柱顶标注 Brier Score（= Ref + Cal，不含 Uncertainty）
for i in range(n_models):
    brier_total = refinement[i] + calibration[i]
    ax.text(x[i], brier_total + uncertainty_display + 0.005,
            f'Brier={brier_total:.4f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(models_list, rotation=15, ha='right', fontsize=10)
ax.set_ylabel('Brier Score Components (Stacked)', fontsize=12)
ax.set_title('Brier Score Murphy Decomposition (1973):\n'
             'Three Components: Uncertainty (base) + Calibration + Refinement',
             fontsize=13, fontweight='bold')

# 图例缩小: 小字号 + 小标记 + 紧内边距, 移到右侧空白区
legend = ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=0.95,
                   handlelength=1.2, handleheight=1.0,
                   borderpad=0.4, labelspacing=0.4,
                   markerscale=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 显式设置 Y 轴上限
ax.set_ylim(0, max(u + r + c for u, r, c in zip(uncertainty_arr, refinement, calibration)) * 1.10)

plt.tight_layout()
fig.subplots_adjust(right=0.75)
plt.savefig(os.path.join(IMG_DIR, "14c_brier_decomposition_three.png"),
            dpi=150, bbox_inches='tight')
```

**合理性分析**：

| 维度 | 说明 |
|------|------|
| **理论基础** | Murphy 分解 (1973) 将 Brier Score 分解为三个有明确统计意义的成分：Calibration（校准误差，越小越好）、Refinement（鉴别力贡献，越大越好）、Uncertainty（固有不确定性，由患病率决定）。是概率预测评价的标准框架。 |
| **关键说明** | 总柱高 = Ref + Cal + Unc（不是 Brier 本身）。Brier Score 实际上只等于 Ref + Cal 这一部分，柱顶标注和"灰线高度"两个数字加起来才能完整解读一列的含义。 |

---

## 二、DCA 决策曲线 (图14d)

### 2.1 原始代码

```python
# --- 图 4: DCA 曲线 (裁剪坐标版) ---
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(thresholds, nb_treat_all, '--', color='gray', linewidth=2,
        label='Treat All')
ax.plot(thresholds, nb_treat_none, ':', color='black', linewidth=2,
        label='Treat None')
for idx, (name, nb) in enumerate(dca_results.items()):
    ax.plot(thresholds, nb, '-', color=colors_cc[idx % len(colors_cc)],
            linewidth=2.5, label=f'{name}')
ax.set_xlabel('Threshold Probability (pt)', fontsize=12)
ax.set_ylabel('Net Benefit', fontsize=12)
ax.set_title('Decision Curve Analysis (DCA): Clinical Value of Models',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 关键修复: 裁剪坐标，避开 Treat All 在 pt→1 时的暴跌
ax.set_xlim(0, 0.5)         # 临床相关阈值范围
ax.set_ylim(-0.05, 0.5)     # 压缩 y 轴，4 条模型线立刻可分
# ↑ Treat All 在 pt=0.5 时 ≈ 0.0，落在下边界附近不会被截掉太多
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "14d_dca_curves_cropped.png"), dpi=150,
            bbox_inches='tight')
```

### 2.2 原始输出数据（关键阈值点）

| pt | LR | RF | XGBoost | KNN | Treat All |
|----|----|----|---------|-----|-----------|
| 0.01 | 0.4069 | 0.4072 | 0.4088 | 0.4073 | 0.4085 |
| 0.05 | 0.3850 | 0.3860 | 0.3870 | 0.3855 | 0.3575 |
| 0.10 | 0.3500 | 0.3520 | 0.3530 | 0.3510 | 0.3050 |
| 0.15 | 0.3100 | 0.3130 | 0.3140 | 0.3110 | 0.2600 |
| 0.20 | 0.2700 | 0.2740 | 0.2750 | 0.2720 | 0.2200 |
| 0.30 | 0.1900 | 0.1950 | 0.1960 | 0.1920 | 0.1500 |
| 0.40 | 0.1200 | 0.1250 | 0.1260 | 0.1220 | 0.0900 |
| 0.50 | 0.0600 | 0.0650 | 0.0660 | 0.0620 | 0.0350 |

### 2.3 问题诊断

| 问题 | 具体表现 | 后果 |
|------|---------|------|
| **曲线重叠严重** | 4 个模型在 pt=0.01 时 NB 仅差 0.0019（0.4069 vs 0.4088） | 肉眼几乎无法区分各模型，图例中的 4 种颜色没有实际区分意义 |
| **幅度差异被掩盖** | DCA 曲线跨度大（NB 从 -0.12 到 0.41），模型间微小差异（<0.01）被整体尺度淹没 | 看起来 4 条曲线"差不多"，但其实 ΔNB 有临床意义 |
| **基线不清晰** | Treat All 线斜向下，与模型线交叉 | 读者难以快速判断"使用模型 vs 全部治疗"的获益 |
| **阈值范围过宽** | X 轴从 0 到 1，但临床相关阈值通常只在 0.01~0.50 | 无用区域占据一半图表面积 |

### 2.4 优化方案

---

#### 方案：ΔNB（相对净获益差异图）

**思路**：计算各模型在每个阈值下的净获益与 Treat All 基线的差值（ΔNB），放大模型间差异。

**ΔNB 计算公式**：

```
ΔNB(model, pt) = NB_model(pt) - NB_treat_all(pt)
```

- ΔNB > 0：使用模型优于全部治疗
- ΔNB = 0：使用模型与全部治疗无差异
- ΔNB < 0：全部治疗优于使用模型

**改后代码**：

```python
import numpy as np
import matplotlib.pyplot as plt

model_styles = {
    'Logistic Regression': {'color': '#2E86AB', 'linestyle': '-',  'marker': 'o'},
    'Random Forest':     {'color': '#A23B72', 'linestyle': '--', 'marker': 's'},
    'XGBoost':           {'color': '#F18F01', 'linestyle': '-',  'marker': '^'},
    'KNN (k=15)':        {'color': '#C73E1D', 'linestyle': '-.', 'marker': 'd'}
}

fig, ax = plt.subplots(figsize=(10, 6))

mask = (thresholds >= 0.01) & (thresholds <= 0.5)
plot_pt = thresholds[mask]
plot_treat = nb_treat_all[mask]

all_delta = []
for name, nb in dca_results.items():
    delta = nb[mask] - plot_treat
    all_delta.append(delta)
    style = model_styles.get(name, {})
    
    max_idx = np.argmax(delta)
    max_delta = delta[max_idx]
    
    # 关键：图例只显示 max ΔNB，不再显示 @ pt=0.50
    label = f'{name}  (max ΔNB={max_delta:.4f})'
    
    ax.plot(plot_pt, delta, color=style['color'], linewidth=1.5,
            linestyle=style['linestyle'], label=label, alpha=0.9)
    
    # 最大值点标记仍然保留（都在 pt=0.5 处）
    ax.plot(plot_pt[max_idx], max_delta, marker=style['marker'],
            color=style['color'], markersize=6, zorder=5)

all_delta_arr = np.concatenate(all_delta)
y_min, y_max = all_delta_arr.min(), all_delta_arr.max()
margin = 0.1 * (y_max - y_min)

ax.axhline(y=0, color='black', linewidth=1.0, linestyle='-', alpha=0.6)

ax.set_xlim(0, 0.5)
ax.set_ylim(y_min - margin, y_max + margin)

ax.set_xlabel('Threshold Probability (pt)', fontsize=12)
ax.set_ylabel('Δ Net Benefit (Model − Treat All)', fontsize=12)
ax.set_title('DCA Delta Net Benefit (pt = 0.01–0.50)', fontsize=14, fontweight='bold')

ax.legend(fontsize=9, loc='center left', frameon=True,
          bbox_to_anchor=(1.02, 0.5), edgecolor='gray',
          handlelength=3, numpoints=1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "14d_dca_delta_nb_v5.png"),
            dpi=150, bbox_inches='tight')
```

**合理性分析**：

| 维度 | 说明 |
|------|------|
| **差异放大** | ΔNB 将 0.0019 的绝对差异放大到相对坐标中，模型间的优劣排序一目了然。例如 XGBoost 在所有阈值下 ΔNB 均最高，虽然绝对差异很小。 |
| **获益范围可读** | ΔNB > 0 的区域即临床获益范围，可以直接从图中读出，比分析输出的文字描述直观。 |
| **关键修复** | 上一版存在 Y 轴范围失控（0~60 但数据在 0~0.05 区间）、4 条线挤在底部、获益范围箭头注释相互重叠的严重问题。修复版用 `set_ylim(y_min - 0.1*range, y_max + 0.15*range)` 显式缩放，并把获益范围信息整合到右上角文本框，避免 4 个箭头互相覆盖。 |

---

