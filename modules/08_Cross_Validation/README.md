# Module 08: 交叉验证

## Cross Validation

> 阶段 3: 建模评估

---

## 为什么医学数据需要这个？

医学数据常按患者分组，同一患者的多次测量不能分到训练集和测试集。GroupKFold 是防止这种泄漏的关键。

---

## 学习目标

- [ ] 理解 K-Fold / Stratified K-Fold 的区别
- [ ] 掌握 Leave-One-Out 交叉验证的适用场景
- [ ] 理解分组交叉验证（GroupKFold）防止患者级泄漏
- [ ] 学会嵌套交叉验证用于超参数调优
- [ ] 理解重复交叉验证降低结果方差

---

## 核心概念

`K-Fold` | `StratifiedKFold` | `GroupKFold` | `Leave-One-Out` | `嵌套CV` | `RepeatedKFold`

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
