import copy

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# 先用 42，之后分别改成 43、44
SEED = 44

MAX_EPOCHS = 500
PATIENCE = 25
MIN_DELTA = 0.0001
LEARNING_RATE = 0.005

torch.manual_seed(SEED)


# 1. 生成 120 个输入，形状从 [120] 变成 [120, 1]
x = torch.linspace(-3, 3, 120).unsqueeze(1)

# 真实关系是 sin(x)，再加一点随机噪声
y = torch.sin(x) + 0.15 * torch.randn_like(x)


# 2. 随机打乱样本编号
indices = torch.randperm(len(x))

# 训练集 72 个、验证集 24 个、测试集 24 个
train_indices = indices[:72]
val_indices = indices[72:96]
test_indices = indices[96:]

x_train = x[train_indices]
y_train = y[train_indices]

x_val = x[val_indices]
y_val = y[val_indices]

x_test = x[test_indices]
y_test = y[test_indices]

print("Seed:", SEED)
print("Train:", x_train.shape)
print("Validation:", x_val.shape)
print("Test:", x_test.shape)
# 3. 定义神经网络
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


# 创建一个模型，复制它的初始参数
initial_model = MLP()

initial_state = copy.deepcopy(
    initial_model.state_dict()
)

criterion = nn.MSELoss()
# 4. 评估指定数据上的 MSE
def evaluate(model, inputs, targets):
    model.eval()

    with torch.no_grad():
        predictions = model(inputs)
        loss = criterion(predictions, targets)

    return loss.item()
# 5. 训练一个模型
def train_model(name, use_early_stopping):
    model = MLP()

    # 两组都从同一份初始参数开始
    model.load_state_dict(initial_state)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    train_losses = []
    val_losses = []

    # 用于保存实际最低验证损失的模型
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None

    # 用于判断是否出现足够明显的改善
    reference_loss = float("inf")
    counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # ---------- 训练 ----------
        model.train()

        optimizer.zero_grad()

        predictions = model(x_train)
        loss = criterion(predictions, y_train)

        loss.backward()
        optimizer.step()

        # ---------- 轮末统一评估 ----------
        train_loss = evaluate(
            model, x_train, y_train
        )

        val_loss = evaluate(
            model, x_val, y_val
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # ---------- 保存实际最好的模型 ----------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

        # ---------- 判断显著改善 ----------
        if val_loss < reference_loss - MIN_DELTA:
            reference_loss = val_loss
            counter = 0
        else:
            counter += 1

        if epoch % 50 == 0:
            print(
                f"{name} | Epoch {epoch:3d} | "
                f"Train {train_loss:.6f} | "
                f"Val {val_loss:.6f} | "
                f"Counter {counter}"
            )

        # ---------- 是否提前停止 ----------
        if use_early_stopping and counter >= PATIENCE:
            print(
                f"{name}: stopped at epoch {epoch}"
            )
            break

    stopped_epoch = len(train_losses)

    # Early Stopping 组恢复最佳权重
    # Baseline 组保留第 500 轮权重
    if use_early_stopping:
        model.load_state_dict(best_state)
        selected_epoch = best_epoch
    else:
        selected_epoch = stopped_epoch

    selected_val_loss = evaluate(
        model, x_val, y_val
    )

    return {
        "name": name,
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "stopped_epoch": stopped_epoch,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "selected_epoch": selected_epoch,
        "selected_val_loss": selected_val_loss
    }
# 6. 分别训练
baseline = train_model(
    name="Baseline",
    use_early_stopping=False
)

early_stopping = train_model(
    name="Early Stopping",
    use_early_stopping=True
)

results = [baseline, early_stopping]


# 7. 两组训练结束后，再评估测试集
print("\n========== Final Results ==========")
print("Seed:", SEED)

for result in results:
    test_loss = evaluate(
        result["model"],
        x_test,
        y_test
    )

    result["test_loss"] = test_loss

    print("\nMethod:", result["name"])
    print("Stopped epoch:", result["stopped_epoch"])
    print("Best validation epoch:", result["best_epoch"])
    print("Selected model epoch:", result["selected_epoch"])

    print(
        "Minimum validation MSE:",
        f"{result['best_val_loss']:.6f}"
    )

    print(
        "Selected model validation MSE:",
        f"{result['selected_val_loss']:.6f}"
    )

    print(
        "Test MSE:",
        f"{result['test_loss']:.6f}"
    )
# 8. 绘制两组损失曲线
fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 4),
    sharey=True
)

for ax, result in zip(axes, results):
    epochs = range(
        1,
        len(result["train_losses"]) + 1
    )

    ax.plot(
        epochs,
        result["train_losses"],
        label="Train MSE"
    )

    ax.plot(
        epochs,
        result["val_losses"],
        label="Validation MSE"
    )

    ax.axvline(
        result["selected_epoch"],
        color="black",
        linestyle="--",
        label="Selected epoch"
    )

    ax.set_title(result["name"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.3)
    ax.legend()

fig.suptitle(f"Early Stopping Comparison | Seed {SEED}")
plt.tight_layout()
figure_name = f"comparison_seed_{SEED}.png"

plt.savefig(
    figure_name,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Figure saved:", figure_name)
