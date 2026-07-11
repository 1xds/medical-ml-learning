# Module 10: 不平衡数据处理

## Imbalanced Data Handling

> 阶段 3: 建模评估

---

## 为什么医学数据需要这个？

医学数据普遍存在类别不平衡（罕见病、阳性样本少）。错误处理不平衡会导致模型对少数类预测能力极差。

---

## 学习目标

- [ ] 理解类别不平衡对模型的影响
- [ ] 掌握 SMOTE / ADASYN 过采样方法
- [ ] 理解欠采样（RandomUnderSampler / Tomek Links）
- [ ] 学会使用 class_weight / scale_pos_weight 调整
- [ ] 理解不平衡场景下的正确评估指标

---

## 核心概念

`SMOTE` | `ADASYN` | `TomekLinks` | `class_weight` | `PR曲线` | `F1-score` | `采样顺序`

---

## 知识框架

```
本模块知识结构

├── 1. 理论基础
├── 2. 方法实现
├── 3. 医学应用
└── 4. 常见陷阱
```

---

## 文件说明

| 文件 | 内容 |
|------|------|
| `README.md` | 本文件：模块概述与学习目标 |
| `notes.md` | 个人学习笔记（写自己的理解） |
| `practice.ipynb` | 实践代码（自己重新敲） |
| `images/` | 运行结果截图 |

---

## 进度

- [ ] 阅读教程讲义
- [ ] 完成代码笔记
- [ ] 运行实践代码
- [ ] 保存可视化结果
- [ ] 撰写 Reflection
- [ ] 提交 Git Commit
