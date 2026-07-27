# Module 19 笔记: 影像组学 (Radiomics) 特征提取

> 本模块的核心命题：**如何利用 pyradiomics 将 CT 医学影像转化为结构化定量特征表，完成从"图像"到"表格数据"的模态迁移，为后续机器学习建模奠定基础。**

---

## 核心概念梳理

### 影像组学是什么？

影像组学 (Radiomics) 由 Lambin 等人在 2012 年提出，其核心主张是：医学影像中的像素信息远超人类视觉系统的感知能力，通过高通量定量特征提取，可将影像转化为可挖掘的高维数据矩阵。影像组学的标准流程为：

$$\text{图像获取} \rightarrow \text{ROI分割} \rightarrow \text{特征提取} \rightarrow \text{特征分析} \rightarrow \text{建模}$$

本模块完成前三步：从 DICOM 格式的腹部 CT 图像出发，经 HU 值转换、体部 ROI 自动分割和 pyradiomics 标准化特征提取，输出 `radiomics_features.csv`——一个"一行=一个患者, 列=93 维定量特征"的标准表格，供 Module 20 进行机器学习流水线分析。

临床场景设定为二分类任务：区分增强 CT (Contrast=1) 和平扫 CT (Contrast=0)。增强 CT 注射碘造影剂后血管和脏器密度显著升高，平扫 CT 则不具备此特性。该任务具有天然的物理可解释性基础：密度类特征应当成为核心判别因子，便于验证模型是否学到了正确的生物学信号。

### HU 值 (Hounsfield Unit)

CT 图像的本质不是普通灰度图，而是以 Hounsfield Unit (HU) 为刻度的标准化密度映射系统。HU 是 CT 的绝对定量尺度，定义如下：

| 物质 | 典型 HU 值范围 |
|------|-------------|
| 空气 | -1000 |
| 肺 | -500 ~ -200 |
| 脂肪 | -100 ~ -50 |
| 水 | 0 |
| 软组织 | +40 ~ +80 |
| 骨 | +400 ~ +1000 |

DICOM 文件中的原始像素值通过 RescaleSlope 和 RescaleIntercept 映射到 HU：

$$\text{HU} = \text{pixel} \times \text{slope} + \text{intercept}$$

一切影像组学特征提取均在 HU 空间进行，这是与计算机视觉领域普通 0-255 灰度图像处理的根本区别——HU 赋予了特征物理和生物学含义，使跨扫描仪、跨中心的标准化成为可能。

### ROI 分割策略

真实影像组学研究中，ROI (Region of Interest) 需通过手工勾画或算法分割来确定（如肿瘤边界）。本教学数据集无分割标注，故采用**体部自动分割**作为可复现近似方案：

1. **HU 阈值筛选**：`HU > -300` 区分软组织/骨（体部）与空气/床板（背景），该阈值低于水 (0 HU) 但高于肺 (-500 HU)，确保尽可能保留所有软组织区域。
2. **最大连通域提取**：对二值 mask 进行连通域分析，仅保留面积最大的连通域，剔除零散噪声和床板伪影。

此策略的教学要点在于呈现 ROI 对特征质量的制约——当任务改变（如肿瘤分割）时，ROI 策略必须重新设计。

### pyradiomics 特征类

pyradiomics 是影像组学领域的标准工具库，本模块启用全部六类标准特征及一阶统计，关闭小波和 LoG 滤波以避免特征数爆炸：

| 特征类 | 含义 | 关键特征举例 |
|--------|------|------------|
| **firstorder** | 一阶统计（密度分布） | Mean, Median, Energy, Entropy, Skewness, Kurtosis, RobustMAD, 10/90Percentile |
| **shape** | 形态学（2D） | Perimeter, Area, Compactness, Sphericity |
| **glcm** | 灰度共生矩阵（纹理） | Contrast, Correlation, Homogeneity, Idn, Idm |
| **glrlm** | 灰度游程矩阵 | RunEntropy, RunLengthNonUniformity, ShortRunEmphasis |
| **glszm** | 灰度大小区域矩阵 | SmallAreaEmphasis, ZoneEntropy, LargeAreaEmphasis |
| **gldm** | 灰度依赖矩阵 | DependenceEntropy, LargeDependenceEmphasis |
| **ngtdm** | 灰度邻域差矩阵 | Coarseness, Complexity, Strength |

六类特征从不同角度刻画 ROI 内像素的密度分布和空间纹理模式，共同构成 93 维定量特征空间。各特征向量量级跨度极大（可达 13 个数量级），必须标准化才能进入 ML 流程。

---

## 代码精读

### Block 1: 环境初始化与依赖加载

```python
import pydicom                           # DICOM 文件读写与元数据解析
import SimpleITK as sitk                  # 医学影像处理 (ROI 分割/连通域分析)
from radiomics import featureextractor    # pyradiomics 特征提取核心引擎

# 抑制 pyradiomics 冗余日志输出
logging.getLogger('radiomics').setLevel(logging.ERROR)

BODY_HU_THRESHOLD = -300  # 体部 HU 阈值 (区分软组织与背景空气)
```

**要点**：pydicom 处理 DICOM 格式的元数据和像素映射，SimpleITK 提供医学影像专用的连通域分析和形态学操作，radiomics 是特征提取引擎。三个库分别对应影像组学流程的前三步。

### Block 2: 数据清单读取

```python
overview = pd.read_csv(OVERVIEW_CSV)                     # 读取患者清单
overview['Contrast'] = overview['Contrast'].astype(int)  # True/False → 1/0
print(f"增强 (Contrast=1): {(overview['Contrast']==1).sum()}")  # 50
print(f"平扫 (Contrast=0): {(overview['Contrast']==0).sum()}")  # 50
```

**要点**：100 张 CT 图像，50/50 完全平衡，年龄范围 39~83 岁。平衡标签避免了类别不平衡的特殊处理。

### Block 3: DICOM → HU 转换函数

```python
def dicom_to_hu(dcm_path):
    """读取 DICOM 并转成 HU (Hounsfield Unit) 矩阵"""
    ds = pydicom.dcmread(dcm_path)                     # 读取 DICOM
    arr = ds.pixel_array.astype(np.float32)            # 像素数组 (原始值)
    slope = float(getattr(ds, 'RescaleSlope', 1))      # 获取斜率 (默认 1)
    intercept = float(getattr(ds, 'RescaleIntercept', 0))  # 获取截距 (默认 0)
    hu = arr * slope + intercept                       # HU = pixel * slope + intercept
    return hu, ds
```

**要点**：`getattr` 提供默认值的安全回退。CT 制造商在 DICOM 头文件中写入 RescaleSlope/RescaleIntercept，不同设备和协议下参数不同，直接使用原始值会导致密度偏移。

### Block 4: 体部 ROI 自动分割

```python
def body_mask(hu, hu_threshold=BODY_HU_THRESHOLD):
    """阈值法 + 最大连通域提取体部 ROI mask"""
    img_sitk = sitk.GetImageFromArray(hu.astype(np.float32))        # HU → SimpleITK Image
    mask_arr = (hu > hu_threshold).astype(np.uint8)                 # 二值阈值 mask
    mask_sitk = sitk.GetImageFromArray(mask_arr)
    cc = sitk.ConnectedComponent(mask_sitk)                         # 连通域标记
    cc = sitk.RelabelComponent(cc, sortByObjectSize=True)           # 按面积排序
    largest = sitk.BinaryThreshold(cc, lowerThreshold=1,             # 取最大连通域
                                   upperThreshold=1, insideValue=1, outsideValue=0)
    return img_sitk, largest
```

**要点**：`sortByObjectSize=True` 确保 label=1 对应最大连通域（即体部），`BinaryThreshold(1,1)` 恰好提取该区域。此策略能有效剔除床板、导线架等扫描设备噪声。

### Block 5: 配置 pyradiomics 特征提取器

```python
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.settings['force2D'] = True            # 2D 模式 (单层 CT)
extractor.settings['force2Ddimension'] = 0      # 轴向位
extractor.settings['label'] = 1                 # mask 中 ROI 像素值
extractor.disableAllImageTypes()                # 关闭小波/LoG 滤波
extractor.enableImageTypes(Original={})         # 仅保留原图特征

# 启用七类标准特征
extractor.enableFeatureClassByName('firstorder')
extractor.enableFeatureClassByName('shape')
extractor.enableFeatureClassByName('glcm')
extractor.enableFeatureClassByName('glrlm')
extractor.enableFeatureClassByName('glszm')
extractor.enableFeatureClassByName('gldm')
extractor.enableFeatureClassByName('ngtdm')
```

**要点**：单层 CT 必须设置 `force2D=True`，否则部分 3D 纹理特征（如 3D GLCM）会失败。`disableAllImageTypes` + `enableImageTypes(Original={})` 确保仅提取原图特征，关闭小波 (Wavelet) 和 LoG 滤波——这些滤波每张图产生 8 倍特征，教学情境下会导致特征维度爆炸。

### Block 6: 批量特征提取

```python
rows = []  # 存储所有样本的特征行
fail_ids = []
for i, row in overview.iterrows():
    dcm_path = os.path.join(DICOM_DIR, row['dicom_name'])
    try:
        hu, _ = dicom_to_hu(dcm_path)               # ① DICOM → HU
        img_sitk, mask_sitk = body_mask(hu)          # ② HU → ROI mask
        result = extractor.execute(img_sitk, mask_sitk)  # ③ 特征提取
        feat_row = {'id': int(row['id']), 'Age': int(row['Age']),
                    'Contrast': int(row['Contrast'])}
        for name in FEATURE_NAMES:
            feat_row[name] = float(result[name])     # 标准特征名 → 数值
        rows.append(feat_row)
    except Exception as e:
        fail_ids.append((int(row['id']), str(e)[:80]))

features_df = pd.DataFrame(rows)                      # 构建特征表
features_df.to_csv(FEATURES_CSV, index=False)         # 保存为 CSV
```

**要点**：特征提取循环体对应影像组学三步流程。异常捕获 `try/except` 确保单张失败不影响全局。`FEATURE_NAMES` 通过预跑一个样本获得（动态探明特征数量），避免硬编码特征列表。

### Block 7: 特征质量核查

```python
X = features_df[FEATURE_NAMES]
print(f"含缺失值特征数: {X.isnull().sum().sum()}")         # 应为 0
print(f"含 inf 特征数:   {np.isinf(X.values).sum()}")      # 纹理特征可能 inf
zero_var = [c for c in FEATURE_NAMES if X[c].std() == 0]   # 零方差特征
ranges = X.max() - X.min()                                  # 量级跨度
print(f"特征量级跨度: {ranges.min():.2f} ~ {ranges.max():.2f}")  # 13 个数量级
```

**要点**：纹理特征在均匀区域（如无结构的肝脏实质）可能产出无穷大值 (inf)，零方差特征无区分度需剔除。量级跨度极广说明标准化是必须步骤。

### Block 8: 可视化 — CT/HU/ROI 三联图

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 原图: CT 灰度显示 (窗宽窗位设置 -200~300 HU)
axes[0].imshow(hu_disp, cmap='gray', vmin=-200, vmax=300)

# HU 分布: ROI 内部的密度直方图
axes[1].hist(hu_disp[mask_disp == 1].ravel(), bins=60, color='#3498db')
axes[1].axvline(BODY_HU_THRESHOLD, color='red', ls='--',
                label=f'threshold = {BODY_HU_THRESHOLD} HU')

# ROI 叠加: 绿色高亮掩膜覆盖原图
overlay = np.dstack([hu_disp, hu_disp, hu_disp])
overlay[mask_disp == 1] = [0.2, 0.8, 0.2]  # 绿色标注体部区域
```

**要点**：三联图直观展示"图像→HU→ROI"的完整预处理链路，教学价值在于让使用者理解每一步产出的物理含义。

### Block 9: 特征按类别分布可视化

```python
# 特征类名称 → 计数条形图
classes = list(class_counts.keys())
counts = list(class_counts.values())
bars = axes[0].bar(classes, counts, color=colors_cls, edgecolor='white')

# 标签分布 (增强/平扫各 50)
ct_counts = features_df['Contrast'].value_counts().sort_index()
axes[1].bar(['Non-Contrast\n(0)', 'Contrast\n(1)'], ct_counts.values,
            color=['#2ecc71', '#e74c3c'], width=0.5)
```

**要点**：firstorder 特征最多（~18），glszm 和 glrlm 各约 16 个，所有类别均匀覆盖。

### Block 10: 关键特征分组对比预览

```python
preview_feats = ['original_firstorder_Mean',
                 'original_firstorder_Entropy',
                 'original_glcm_Contrast',
                 'original_glcm_Correlation']
for ax, feat in zip(axes, preview_feats):
    g0 = features_df.loc[features_df['Contrast'] == 0, feat]  # 平扫组
    g1 = features_df.loc[features_df['Contrast'] == 1, feat]  # 增强组
    ax.boxplot([g0, g1], tick_labels=['Non-Contrast', 'Contrast'])
```

**要点**：boxplot 预览提供直觉判断——增强组的 Mean HU 和 Entropy 应系统性地高于平扫组，这一先验期望将在 Module 20 的统计检验中正式验证。

---

## 关键收获

1. **影像组学作为"图像→特征"的标准化桥梁**：DICOM 格式的原始影像无法直接送入机器学习算法，影像组学通过 HU 值转换、ROI 分割和定量特征提取，将非结构化的像素矩阵转化为结构化的特征表——这一转化是实现影像数据端到端 ML 流水线的前提。

2. **HU 值赋予特征物理含义**：与 RGB 灰度的任意性不同，HU 是跨设备、跨中心的标准化密度刻度。这意味着影像组学特征具有物理可解释性——例如 firstorder_Mean 在增强 CT 上系统性升高直接反映造影剂对组织密度的提升。

3. **ROI 分割对特征质量的决定性影响**：ROI 的质量直接影响所有特征的计算基础。当 ROI 包含非目标组织（如本案例中的骨骼外缘、皮下脂肪），特征会被噪声污染。真实研究中的 ROI 策略必须针对具体解剖部位和临床任务定制。

4. **特征维度与样本量的不平衡 (p >> n)**：93 维特征仅 100 样本构成了典型的高维小样本问题，直接使用全部特征将导致严重过拟合。Module 20 的特征选择（LASSO）正是为解决此问题而设计。

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| pyradiomics 安装失败 (缺少 Cython 头文件) | sdist 包不包含生成的 .c/.cpp 文件 | 从 GitHub 源码编译：`git clone + pip install --no-build-isolation .` |
| `force2D` 未设置导致特征提取失败 | 单层 CT 需要 2D 模式 | 设置 `extractor.settings['force2D'] = True` |
| 纹理特征产出 inf 或 NaN | ROI 内部密度过于均匀（如肝脏实质） | 适当扩大 ROI；在切分数据前检测并处理 inf/NaN |
| 小波滤波产生特征爆炸 | 小波滤波默认开启，每张图生成 8 组特征 | `disableAllImageTypes()` + `enableImageTypes(Original={})` |
| HU 映射后出现异常值 | RescaleSlope 或 Intercept 缺失/异常 | 使用 `getattr(ds, 'RescaleSlope', 1)` 提供安全默认值 |

---

## 与其他模块的联系

- **前置模块**：Module 01-12 (表格数据 ML 流程) — Module 19 是"从表格数据到影像数据"的关键桥梁。所有后续 ML 步骤（EDA、统计、特征选择、建模、SHAP）完全复用前 12 个模块的通用流程。
- **后续模块**：Module 20 (影像组学 ML 流水线) — 直接读取本模块输出的 `radiomics_features.csv`，完成从特征分析到建模的完整链路。
- **与研究工作的联系**：若在虚拟筛选研究中引入分子结构的影像化表示（如分子表面静电势图），影像组学方法可直接迁移用于提取分子的"影像特征"，作为传统分子描述符的补充模态。类似的模态迁移思路在宏基因组研究中同样适用：将菌群共丰度热图视为"影像"，用纹理特征捕捉群落结构的空间模式。

---

## 参考资料

- 教程原文：`ml4health-main/jupyter/19_radiomics_feature_extraction.ipynb`
- 讲义：`ml4health-main/lectures/19_20_radiomics_case_study_teaching_doc.md`
- Lambin, P., et al. (2012). "Radiomics: Extracting more information from medical images using advanced feature analysis." *European Journal of Cancer*, 48(4), 441-446.
- Aerts, H. J., et al. (2014). "Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach." *Nature Communications*, 5, 4006.
- van Griethuysen, J. J., et al. (2017). "Computational Radiomics System to Decode the Radiographic Phenotype." *Cancer Research*, 77(21), e104-e107.
- pyradiomics 文档：https://pyradiomics.readthedocs.io/
