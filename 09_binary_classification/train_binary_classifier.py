import copy

import torch
import torch.nn as nn
import matplotlib

# 使用不会弹出窗口的绘图方式
matplotlib.use("Agg")

import matplotlib.pyplot as plt


SEED = 42
EPOCHS = 200
LEARNING_RATE = 0.05

torch.manual_seed(SEED)


# 1. 生成 600 个样本，每个样本有两个特征
x = torch.randn(600, 2)

# 如果两个特征之和大于 0，标签为 1，否则为 0
y = (x[:, 0] + x[:, 1] > 0).float()

# 从 [600] 变为 [600, 1]，与模型输出形状一致
y = y.unsqueeze(1)


# 2. 随机划分：训练 360、验证 120、测试 120
indices = torch.randperm(len(x))

train_indices = indices[:360]
val_indices = indices[360:480]
test_indices = indices[480:]

x_train = x[train_indices]
y_train = y[train_indices]

x_val = x[val_indices]
y_val = y[val_indices]

x_test = x[test_indices]
y_test = y[test_indices]


print("Train:", x_train.shape, y_train.shape)
print("Validation:", x_val.shape, y_val.shape)
print("Test:", x_test.shape, y_test.shape)

# 3. 两个输入，一个原始分数输出
model = nn.Linear(2, 1)

# 不在模型后面添加 Sigmoid
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# 4. 同时计算损失与准确率
def evaluate(model, inputs, labels):
    model.eval()

    with torch.no_grad():
        logits = model(inputs)

        # 损失函数接收原始分数
        loss = criterion(logits, labels)

        # 计算预测类别时，再转换成概率
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()

        accuracy = (
            predictions == labels
        ).float().mean()

    return loss.item(), accuracy.item()


# 先看看未经训练的模型表现
initial_val_loss, initial_val_acc = evaluate(
    model, x_val, y_val
)

print(
    f"\nBefore training | "
    f"Val Loss: {initial_val_loss:.4f} | "
    f"Val Acc: {initial_val_acc:.2%}"
)

# 5. 记录训练过程
train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

best_val_loss = float("inf")
best_epoch = 0
best_state = None


for epoch in range(1, EPOCHS + 1):
    model.train()

    optimizer.zero_grad()

    logits = model(x_train)
    loss = criterion(logits, y_train)

    loss.backward()
    optimizer.step()

    # 参数更新结束后，统一评估
    train_loss, train_acc = evaluate(
        model, x_train, y_train
    )

    val_loss, val_acc = evaluate(
        model, x_val, y_val
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    # 根据验证损失选择模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch

        best_state = copy.deepcopy(
            model.state_dict()
        )

    if epoch % 20 == 0:
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.2%} | "
            f"Val Acc: {val_acc:.2%}"
        )


# 6. 恢复验证损失最低的模型
model.load_state_dict(best_state)

# 训练及模型选择完成后，才评估测试集
test_loss, test_acc = evaluate(
    model, x_test, y_test
)

print("\n========== Final Results ==========")
print("Seed:", SEED)
print("Best epoch:", best_epoch)
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"Test loss: {test_loss:.6f}")
print(f"Test accuracy: {test_acc:.2%}")


# 查看模型学到的两个权重和一个偏置
print("\nLearned weights:")
print(model.weight.detach())

print("Learned bias:")
print(model.bias.detach())

# 7. 绘制训练过程
epochs = range(1, EPOCHS + 1)

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 4)
)

axes[0].plot(
    epochs, train_losses,
    label="Train"
)

axes[0].plot(
    epochs, val_losses,
    label="Validation"
)

axes[0].set_title("Binary Cross-Entropy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")


axes[1].plot(
    epochs, train_accuracies,
    label="Train"
)

axes[1].plot(
    epochs, val_accuracies,
    label="Validation"
)

axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1.05)


for ax in axes:
    ax.axvline(
        best_epoch,
        color="black",
        linestyle="--",
        label="Best validation epoch"
    )

    ax.grid(alpha=0.3)
    ax.legend()

plt.tight_layout()

figure_name = f"binary_training_seed_{SEED}.png"

plt.savefig(
    figure_name,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("\nFigure saved:", figure_name)

model.eval()

with torch.no_grad():
    test_logits = model(x_test)

    test_probabilities = torch.sigmoid(
	test_logits
    )

    test_predictions = (
	test_probabilities >= 0.5
    ).float()

true_labels = y_test.squeeze(1)

predicted_labels = test_predictions.squeeze(1)

class_0_mask = (true_labels == 0)
class_1_mask = (true_labels == 1)

wrong_mask = (
    predicted_labels != true_labels
)

print(
    "Misclassified test samples:",
    wrong_mask.sum().item()
)

weights = model.weight.detach()

w1 = weights[0, 0].item()
w2 = weights[0, 1].item()

bias = model.bias.detach().item()

print("Boundary parameters:", w1, w2, bias)

x1_min = x_test[:, 0].min().item() - 0.5
x1_max = x_test[:, 0].max().item() + 0.5

x2_min = x_test[:, 1].min().item() - 0.5
x2_max = x_test[:, 1].max().item() + 0.5

line_x1 = torch.linspace(
    x1_min,
    x1_max,
    200
)

true_line_x2 = -line_x1

fig, ax = plt.subplots(
    figsize=(7, 6)
)

ax.scatter(
    x_test[class_0_mask, 0].numpy(),
    x_test[class_0_mask, 1].numpy(),
    color="royalblue",
    label="True class 0",
    s=35,
    alpha=0.7
)

ax.scatter(
    x_test[class_1_mask, 0].numpy(),
    x_test[class_1_mask, 1].numpy(),
    color="darkorange",
    label="True class 1",
    s=35,
    alpha=0.7
)

# 11. 真实边界
ax.plot(
    line_x1.numpy(),
    true_line_x2.numpy(),
    color="black",
    linestyle="--",
    label="True boundary"
)


# 12. 模型边界
if abs(w2) > 1e-8:
    model_line_x2 = -(
        w1 * line_x1 + bias
    ) / w2

    ax.plot(
        line_x1.numpy(),
        model_line_x2.numpy(),
        color="green",
        label="Learned boundary"
    )

elif abs(w1) > 1e-8:
    # w2 接近 0 时，边界为竖线
    ax.axvline(
        x=-bias / w1,
        color="green",
        label="Learned boundary"
    )

else:
    print("No unique linear boundary to draw.")


# 13. 如果存在分错的样本，用空心圆圈出
if wrong_mask.any().item():
    ax.scatter(
        x_test[wrong_mask, 0].numpy(),
        x_test[wrong_mask, 1].numpy(),
        facecolors="none",
        edgecolors="black",
        s=110,
        linewidths=1.5,
        label="Misclassified"
    )

# 14. 设置显示范围与说明
ax.set_xlim(x1_min, x1_max)
ax.set_ylim(x2_min, x2_max)

ax.set_xlabel("Feature x1")
ax.set_ylabel("Feature x2")

ax.set_title(
    f"Test Samples and Boundaries | Seed {SEED}"
)

# 横轴、纵轴一个单位显示为相同长度
ax.set_aspect("equal", adjustable="box")

ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()


# 15. 保存图片
boundary_figure_name = (
    f"binary_boundary_seed_{SEED}.png"
)

fig.savefig(
    boundary_figure_name,
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

print(
    "Boundary figure saved:",
    boundary_figure_name
)
