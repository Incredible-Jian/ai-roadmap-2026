import torch
import torch.nn as nn

# 1. 假设这是对三个样本输出的原始分数
logits = torch.tensor([
    [-2.0],
    [0.0],
    [0.2]
])

# 三个样本的真实标签
# 二元交叉熵使用浮点标签， 形状与Logits一致
labels = torch.tensor([
    [0.0],
    [1.0],
    [1.0]
])

# 2. 将原始分数转成概率
probabilities = torch.sigmoid(logits)

# 3. 按照0.5阈值生成预测类别
# 比较得到True / False， 再转成浮点数1.0 / 0.0
predictions = (probabilities >= 0.5).float()

# 4. 计算每个样本的损失
criterion = nn.BCEWithLogitsLoss(
    reduction="none"
)

losses = criterion(logits, labels)

mean_loss = losses.mean()

# 5. 计算准确率：预测正确的数量 / 总样本数
correct = (predictions == labels).float()

accuracy = correct.mean()

print("Logits:")
print(logits)

print("\nProbabilities:")
print(probabilities)

print("\nPredictions:")
print(predictions)

print("\nLabels:")
print(labels)

print("\nLoss for each sample")
print(losses)

print("\nMean loss:", mean_loss.item())
print("Accuracy:", accuracy.item())
