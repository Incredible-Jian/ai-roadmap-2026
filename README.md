# AI 学习与无人机视觉研究记录

记录我的 Python、PyTorch、机器学习基础实验，以及无人机视觉相关论文的阅读过程。

学习方式：理解核心知识 → 手写代码 → 运行实验 → 分析结果 → 记录重点与薄弱点。日常笔记保持简短，以能解释实验、逐步独立编写代码为目标。

## 学习进度

| 阶段 | 内容 | 当前进展 |
| --- | --- | --- |
| Day1–Day7 | Python、PyTorch、神经网络、数据管道、模型评估、优化与正则化 | 已完成相关基础练习，持续复习 |
| Day8 | Early Stopping 对比实验 | 完成随机种子 42、43、44 的对比，理解最佳模型保存与耐心计数 |
| Day9 | 线性二分类 | 完成训练、验证、测试和决策边界可视化 |
| Day10 | 线性模型、多层线性模型与 ReLU MLP | 完成非线性分类对比，理解激活函数的作用 |
| Day11 | IoU、DataLoader 与损失统计 | 完成框重叠实验和 batch 平均损失对比 |

## 目录导航

- [01_python](01_python/)：Python 基础。
- [02_pytorch](02_pytorch/)：张量与 PyTorch 基础。
- [03_neural_network](03_neural_network/)：神经网络基础。
- [04_data_pipeline](04_data_pipeline/)：Dataset、DataLoader 与分批训练。
- [05_model_evaluation](05_model_evaluation/)：模型评估。
- [06_optimization](06_optimization/)：优化方法。
- [07_regularization](07_regularization/)：正则化与相关实验。
- [08_early_stopping_comparison](08_early_stopping_comparison/)：提前停止对比实验。
- [09_binary_classification](09_binary_classification/)：二分类与决策边界。
- Day10：非线性分类实验，脚本为 `nonlinear_comparison.py`，目录沿用本地现有名称。
- [11_detection_basics](11_detection_basics/)：IoU 与批次损失统计实验。
- [notes](notes/)：学习重点与薄弱点。
- [papers](papers/)：论文索引和阅读笔记。

## 代表性实验与结论

### 1. Early Stopping

相同随机种子下，Baseline 与 Early Stopping 使用相同初始模型参数进行对比。Baseline 训练 500 轮，Early Stopping 根据验证集表现停止并恢复所保存的最佳模型。

| 随机种子 | Baseline 测试 MSE | Early Stopping 测试 MSE | Early Stopping 停止轮次 |
| --- | ---: | ---: | ---: |
| 42 | 0.023393 | 0.022991 | 94 |
| 43 | 0.035514 | 0.037517 | 97 |
| 44 | 0.028979 | 0.032923 | 91 |

主要理解：提前停止减少了本实验的训练轮次，但不保证测试误差更低。验证集用于选择模型，有限测试集上的表现仍可能波动。这里仅记录三个种子的结果，不据此作普遍结论。

### 2. 线性二分类

在合成二维数据上，使用 `Linear(2, 1)` 和 `BCEWithLogitsLoss` 训练模型。

- 随机种子：42。
- 数据划分：训练 360、验证 120、测试 120。
- 测试准确率：100.00%。
- 测试损失：0.140789。

主要理解：准确率只反映分类是否正确，交叉熵还与预测概率有关。有限测试集上全部预测正确，不代表模型学到了完全准确的真实边界。

### 3. 非线性分类

合成任务：两个输入特征同号为类别 0，异号为类别 1。

| 模型 | 结构 | 测试准确率 |
| --- | --- | ---: |
| Linear | Linear(2, 1) | 44.37% |
| StackedLinear | Linear(2, 16) → Linear(16, 1) | 49.38% |
| MLP_ReLU | Linear(2, 16) → ReLU → Linear(16, 1) | 97.50% |

主要理解：多层线性变换仍可合并为一个线性变换；ReLU 引入非线性，使网络能够表达更复杂的分类边界。上述数值来自本次运行，不代表所有数据和训练设置下的结果。

### 4. IoU 与 batch 损失统计

- 理解框的 `xyxy` 表示，以及交集、并集和 IoU。
- 修正交集计算：左上角取 `max`，右下角取 `min`。
- 理解候选框过大、过小或偏移都会影响定位质量。
- 在五样本实验中，直接平均三个 batch 的平均损失得到 7.6667，按样本数加权得到 6.0000，与全体样本直接计算一致。
- 理解训练过程中累计的 loss 与训练结束后固定参数重新评估的 loss 不一定相同。

## 论文学习

从基础概念出发阅读无人机视觉相关论文，逐步整理任务、问题、方法与局限之间的联系。

当前正在导读 [PointSR（CVPR 2025）](papers/PointSR.md)，重点理解点标注、候选框和伪框。尚未完成全文精读或复现。

完整索引见 [论文阅读记录](papers/README.md)。

## 下一步

- 学习图像张量、通道与卷积。
- 完成简单 CNN 图像分类实验。
- 继续补齐目标检测基础，再深入阅读 PointSR。
- 逐步提高独立组织训练代码和分析实验结果的能力。

## 说明

本仓库用于个人学习。部分实验与笔记在 AI 辅助下完成，代码由本人逐步练习，结果来自实际运行。论文结论与个人理解分开记录，未完成的阅读和复现不标为完成。
