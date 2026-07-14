# Medical Machine Learning Learning Portfolio

> 南京医科大学 · 卓越工程师初阶学习 · Summer 2026
>
> 基于 [ml4health](https://github.com/IILab-Resource/ml4health) 教程，按模块整理的学习笔记。

---

## 学习进度

**Medical Machine Learning · Summer 2026**

- 已完成：0 / 22 模块
- 预计周期：2026.7 — 2026.8

| # | 模块 | 英文名 | 阶段 | 状态 |
|---|------|--------|------|------|
| 01 | [探索性数据分析](modules/01_EDA/) | EDA | 基础入门 | ⬜ |
| 02 | [统计检验](modules/02_Statistical_Tests/) | Statistical Tests | 基础入门 | ⬜ |
| 03 | [预处理与缺失值插补](modules/03_Preprocessing/) | Preprocessing | 特征工程 | ⬜ |
| 04 | [特征工程](modules/04_Feature_Construction/) | Feature Construction | 特征工程 | ⬜ |
| 05 | [特征选择](modules/05_Feature_Selection/) | Feature Selection | 特征工程 | ⬜ |
| 06 | [降维与聚类](modules/06_Dimensionality_Reduction_Clustering/) | Dim. Reduction & Clustering | 建模评估 | ⬜ |
| 07 | [数据泄漏分析](modules/07_Data_Leakage_Prevention/) | Data Leakage Prevention | 建模评估 | ⬜ |
| 08 | [交叉验证](modules/08_Cross_Validation/) | Cross Validation | 建模评估 | ⬜ |
| 09 | [建模对比](modules/09_Model_Comparison/) | Model Comparison | 建模评估 | ⬜ |
| 10 | [类别不平衡处理](modules/10_Imbalanced_Data/) | Imbalanced Data | 建模评估 | ⬜ |
| 11 | [校准分析与DCA](modules/11_Calibration_DCA/) | Calibration & DCA | 建模评估 | ⬜ |
| 12 | [SHAP + LIME](modules/12_SHAP_Overview/) | SHAP Overview | 可解释性 | ⬜ |
| 13 | [SHAP交互效应](modules/13_SHAP_Interaction/) | SHAP Interaction | 可解释性 | ⬜ |
| 14 | [SHAP依赖图](modules/14_SHAP_Dependence/) | SHAP Dependence | 可解释性 | ⬜ |
| 15 | [SHAP聚类](modules/15_SHAP_Clustering/) | SHAP Clustering | 可解释性 | ⬜ |
| 16 | [SHAP决策路径](modules/16_SHAP_Decision_Path/) | SHAP Decision Path | 可解释性 | ⬜ |
| 17 | [SHAP Bootstrap](modules/17_SHAP_Bootstrap/) | SHAP Stability | 可解释性 | ⬜ |
| 18 | [双重机器学习](modules/18_DML_Causal_Inference/) | Double ML (DML) | 因果推断 | ⬜ |
| 19 | [影像组学特征提取](modules/19_Radiomics_Feature_Extraction/) | Radiomics Features | 综合案例 | ⬜ |
| 20 | [影像组学ML流水线](modules/20_Radiomics_ML_Pipeline/) | Radiomics ML Pipeline | 综合案例 | ⬜ |
| 21 | [基因组学ML流水线](modules/21_TCGA_Multiclass/) | TCGA Multiclass | 综合案例 | ⬜ |
| 22 | [CGM血糖回归](modules/22_CGM_Glucose_Regression/) | CGM Glucose Regression | 综合案例 | ⬜ |

---

## 仓库结构

```
medical-ml-learning/
│
├── README.md              # 本文件：项目总览与进度
├── LEARNING_LOG.md        # 学习日志（每日记录 + 每周总结）
├── resources.md           # 参考资料（教程、论文、工具链接）
├── .gitignore
├── assets/
│   └── progress.svg       # 进度可视化
│
└── modules/
    ├── 01_EDA/
    │   ├── README.md      # 模块概述与学习目标
    │   ├── notes.md       # 个人学习笔记
    │   ├── practice.ipynb # 实践代码
    │   └── images/        # 运行结果截图
    ├── 02_Statistical_Tests/
    ├── 03_Preprocessing/
    ├── ...
    └── 22_CGM_Glucose_Regression/
```

每个模块文件夹包含：

| 文件 | 内容 |
|------|------|
| `README.md` | 模块概述、学习目标、知识框架 |
| `notes.md` | 个人笔记（写自己的理解，不抄教程） |
| `practice.ipynb` | 实践代码（自己重新敲，不是复制） |
| `images/` | 运行结果截图（分布图、热力图等） |

---

## 笔记规则

- 每篇笔记对应一个**知识点**，不是日记
- 写自己的理解，不抄教程
- 包含：核心概念 → 代码要点 → 踩坑记录 → 可复用片段
- 学完一个知识点就整理一篇，不赶频率

---

## Commit Message 速查

| 做了什么 | Commit Message |
|----------|---------------|
| 新建仓库 | `Initial commit` |
| 新增笔记 | `Add <topic> notes` |
| 更新笔记 | `Update <topic> notes` |
| 新增代码 | `Add <topic> script` |
| 上传图片 | `Add figures` |
| 修复错误 | `Fix <description>` |
| 更新文档 | `Update documentation` |

---

## 相关仓库

- [ml4health](https://github.com/IILab-Resource/ml4health) — 原始教程
- [MurJ-Drug-Discovery](https://github.com/1xds/MurJ-Drug-Discovery) — 研究项目

## License

MIT
