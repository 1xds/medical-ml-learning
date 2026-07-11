# Medical Machine Learning Learning Portfolio

> 南京医科大学 · 医学机器学习教程 (ml4health) 学习项目
> Summer 2026 · 2026.7 — 2026.8

---

## 项目简介

本项目是基于南京医科大学 ml4health 教程的 **持续更新学习作品集 (Learning Portfolio)**。

目标：通过 22 个模块的系统学习，掌握医学机器学习的完整工作流程，同时形成可展示的学习成长轨迹。

---

## 总进度

```
Medical Machine Learning
Summer 2026

░░░░░░░░░░░░░░░░░░░░░░

0 / 22 Modules

预计完成时间：2026.7 —— 2026.8
```

---

## 22 个教学模块

| # | 模块 | 英文名 | 阶段 | 状态 |
|---|------|--------|------|------|
| 01 | [探索性数据分析](modules/01_EDA/README.md) | Exploratory Data Analysis | 基础入门 | ⬜ |
| 02 | [统计检验](modules/02_Statistical_Tests/README.md) | Statistical Tests | 基础入门 | ⬜ |
| 03 | [数据预处理](modules/03_Preprocessing/README.md) | Data Preprocessing | 特征工程 | ⬜ |
| 04 | [特征构造](modules/04_Feature_Construction/README.md) | Feature Construction | 特征工程 | ⬜ |
| 05 | [特征选择](modules/05_Feature_Selection/README.md) | Feature Selection | 特征工程 | ⬜ |
| 06 | [降维与聚类](modules/06_Dimensionality_Reduction_Clustering/README.md) | Dimensionality Reduction & Clustering | 建模评估 | ⬜ |
| 07 | [防数据泄漏](modules/07_Data_Leakage_Prevention/README.md) | Data Leakage Prevention | 建模评估 | ⬜ |
| 08 | [交叉验证](modules/08_Cross_Validation/README.md) | Cross Validation | 建模评估 | ⬜ |
| 09 | [建模对比](modules/09_Model_Comparison/README.md) | Model Comparison | 建模评估 | ⬜ |
| 10 | [不平衡数据处理](modules/10_Imbalanced_Data/README.md) | Imbalanced Data Handling | 建模评估 | ⬜ |
| 11 | [校准与决策曲线分析](modules/11_Calibration_DCA/README.md) | Calibration & DCA | 建模评估 | ⬜ |
| 12 | [SHAP 概述](modules/12_SHAP_Overview/README.md) | SHAP Overview | 可解释性与因果 | ⬜ |
| 13 | [SHAP 交互效应](modules/13_SHAP_Interaction/README.md) | SHAP Interaction Effects | 可解释性与因果 | ⬜ |
| 14 | [SHAP 依赖图](modules/14_SHAP_Dependence/README.md) | SHAP Dependence Analysis | 可解释性与因果 | ⬜ |
| 15 | [SHAP 聚类](modules/15_SHAP_Clustering/README.md) | SHAP Clustering | 可解释性与因果 | ⬜ |
| 16 | [SHAP 决策路径](modules/16_SHAP_Decision_Path/README.md) | SHAP Decision Path | 可解释性与因果 | ⬜ |
| 17 | [SHAP Bootstrap](modules/17_SHAP_Bootstrap/README.md) | SHAP Bootstrap Analysis | 可解释性与因果 | ⬜ |
| 18 | [双重机器学习因果推断](modules/18_DML_Causal_Inference/README.md) | Double Machine Learning (DML) | 可解释性与因果 | ⬜ |
| 19 | [影像组学特征提取](modules/19_Radiomics_Feature_Extraction/README.md) | Radiomics Feature Extraction | 综合案例 | ⬜ |
| 20 | [影像组学ML流水线](modules/20_Radiomics_ML_Pipeline/README.md) | Radiomics ML Pipeline | 综合案例 | ⬜ |
| 21 | [基因组学 TCGA 多分类](modules/21_TCGA_Multiclass/README.md) | Genomics TCGA Multiclass | 综合案例 | ⬜ |
| 22 | [CGM 血糖回归](modules/22_CGM_Glucose_Regression/README.md) | CGM Glucose Regression | 综合案例 | ⬜ |

---

## 四阶段学习路径

| 阶段 | 模块范围 | 核心内容 |
|------|----------|----------|
| 1. 基础入门 | 01 — 02 | EDA探索性分析 + 统计检验 |
| 2. 特征工程 | 03 — 05 | 预处理 → 特征构造 → 特征选择 |
| 3. 建模评估 | 06 — 11 | 降维聚类 → 防泄漏 → 交叉验证 → 建模对比 → 不平衡处理 → 校准DCA |
| 4. 可解释性与因果 | 12 — 18 | SHAP全系列 + DML因果推断 |
| 5. 综合案例 | 19 — 22 | 影像组学 / 基因组学 / CGM血糖回归 |

---

## 仓库结构

```
medical-ml-learning/
├── README.md              # 项目首页
├── LEARNING_LOG.md        # 每日/每周学习日志
├── resources.md           # 教程、论文、参考资料
├── modules/
│   ├── 01_EDA/
│   │   ├── README.md      # 模块概述 + 学习目标
│   │   ├── notes.md       # 个人学习笔记
│   │   ├── practice.ipynb # 实践代码
│   │   └── images/        # 运行结果截图
│   ├── 02_Statistical_Tests/
│   └── ...
└── assets/
    └── progress.svg       # 进度条可视化
```

---

## 每个模块的学习流程

1. **阅读教程** → 理解概念框架
2. **写 notes.md** → 用自己的话记录理解（不抄教程）
3. **敲 practice.ipynb** → 重新实现代码（不复制）
4. **保存 images/** → 截图运行结果
5. **写 Reflection** → 记录收获、踩坑、下一步
6. **Git Commit** → `Finish ModuleXX XXX`

---

## 与我的研究的联系

| 研究方向 | 相关模块 | 应用场景 |
|----------|----------|----------|
| EcMurJ 药物设计 | 03, 05, 09, 12-17 | 特征工程 + 建模 + SHAP解释 |
| 长寿人群宏基因组 | 01, 02, 06, 18, 21 | EDA + 统计检验 + 降维聚类 + 因果推断 |
| 菌落图像分析 | 06, 19, 20 | 降维聚类 + 影像组学流水线 |

---

## 推荐环境

- Anaconda (Python 3.10+)
- VSCode + Jupyter 插件
- Git for version control

---

## License

MIT License - 本项目为个人学习作品集，教程内容版权归 ml4health 原作者所有。
