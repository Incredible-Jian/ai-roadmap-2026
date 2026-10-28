import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset

# 1. 固定随机种子
torch.manual_seed(42)

# 2. 创建合成数据
X = torch.linspace(-3, 3, 300).reshape(-1, 1)

# 真实规律：y = sin(x)
y = torch.sin(X)

# 加入随机噪声
y = y + 0.15 * torch.randn_like(y)

# 3. 划分训练集和验证集
X_train = X[:240]
y_train = y[:240]

X_val = X[240:]
y_val = y[240:]

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)

# 4. 定义神经网络
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
)

# 5. 记录训练过程
train_losses = []
val_losses = []

epochs = 100

for epoch in range(epochs):
    model.train()

    total_train_loss = 0.0
    train_samples = 0

    for X_batch, y_batch in train_loader:
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = X_batch.size(0)
        total_train_loss += loss.item() * batch_size
        train_samples += batch_size

    train_loss = total_train_loss / train_samples

    # 验证阶段：不更新参数
    model.eval()

    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = criterion(
            val_predictions,
            y_val,
        ).item()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch + 1:3d}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

# 6. 绘制损失曲线
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, epochs + 1),
    train_losses,
    label="Train Loss",
)

plt.plot(
    range(1, epochs + 1),
    val_losses,
    label="Validation Loss",
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(
    "05_model_evaluation/loss_curve.png"
)

print("\nLoss curve saved to:")
print("05_model_evaluation/loss_curve.png")
