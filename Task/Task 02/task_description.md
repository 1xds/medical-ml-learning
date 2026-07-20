# 实验任务说明

## 基于乳腺超声影像组学的良恶性分类实验

---

# 一、实验任务

给定**乳腺超声图像、病灶 ROI 掩膜及良恶性标签**，完成超声影像组学分类流程：

- 提取一阶统计、形状和纹理等影像组学特征；
- 选择至少 **2 种分类模型**进行训练与比较；
- 在测试集上评价模型性能；
- 对最优模型进行可解释性分析。

---

# 二、基本要求

至少比较两种模型，其中应包含：

- **1 种线性模型：**LR；
- **1 种非线性模型：**SVM、RF 或 XGBoost。

### 报告以下指标：

- AUC；
- Accuracy；
- Sensitivity；
- Specificity。

### 绘制：

- ROC 曲线；
- 混淆矩阵；
- 前 10 个关键特征的重要性图。

### 简要讨论：

- 哪种模型表现最好；
- 哪些影像组学特征较重要；
- 模型可能存在的局限。

---

# 三、提交材料

## 1. 代码或 Notebook

能够从读取数据开始，完整复现实验结果。

---

## 2. 实验结果

包含模型性能对比表、ROC 曲线、混淆矩阵和特征重要性图。

---

## 3. 简短实验报告

建议 **3～5 页**，内容包括：

> 数据与任务 → 特征与预处理 → 模型设置 → 实验结果 → 可解释性分析 → 总结与思考

---

# 四、拓展任务

- 特征筛选或降维；
- 交叉验证和超参数优化；
- SHAP 个体预测解释；
- 在外部乳腺超声数据集上验证。

---

# 实验数据

**资料/Dataset_BUSI.zip**

---

# 参考文献

1. Romeo V, Cuocolo R, Apolito R, et al. *Clinical value of radiomics and machine learning in breast ultrasound: a multicenter study for differential diagnosis of benign and malignant lesions*. **European Radiology**, 2021, 31(12): 9511–9519.