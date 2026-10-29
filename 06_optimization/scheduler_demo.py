import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def run_experiment(use_scheduler):
    torch.manual_seed(7)

    # 构造 y = 3x + 2 的回归数据
    X = torch.linspace(-2, 2, 200).reshape(-1, 1)
    y = 3 * X + 2

    model = nn.Linear(1, 1)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.3,
        )

    criterion = nn.MSELoss()

    losses = []
    learning_rates = []

    for epoch in range(100):
        model.train()

        optimizer.zero_grad()

        predictions = model(X)
        loss = criterion(predictions, y)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
        learning_rates.append(
            optimizer.param_groups[0]["lr"]
        )

    return {
        "losses": losses,
        "learning_rates": learning_rates,
        "weight": model.weight.item(),
        "bias": model.bias.item(),
    }


fixed = run_experiment(use_scheduler=False)
scheduled = run_experiment(use_scheduler=True)

for name, result in [
    ("Fixed LR", fixed),
    ("StepLR", scheduled),
]:
    print(f"\n{name}")
    print(f"Initial loss: {result['losses'][0]:.6f}")
    print(f"Final loss:   {result['losses'][-1]:.6f}")
    print(f"Weight:       {result['weight']:.4f}")
    print(f"Bias:         {result['bias']:.4f}")

# 绘制损失曲线
plt.figure(figsize=(8, 5))

plt.plot(
    fixed["losses"],
    label="Fixed LR",
)

plt.plot(
    scheduled["losses"],
    label="StepLR",
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Fixed Learning Rate vs StepLR")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "06_optimization/scheduler_comparison.png"
)

# 绘制学习率变化
plt.figure(figsize=(8, 5))

plt.plot(
    fixed["learning_rates"],
    label="Fixed LR",
)

plt.plot(
    scheduled["learning_rates"],
    label="StepLR",
)

plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "06_optimization/learning_rate_schedule.png"
)

print("\nPlots saved:")
print("06_optimization/scheduler_comparison.png")
print("06_optimization/learning_rate_schedule.png")
