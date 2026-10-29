import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)

# 1. Construct simple regression data
X = torch.linspace(-2, 2, 200).reshape(-1, 1)
y = 3 * X + 2

# 2. Compare three learning rates
learning_rates = [0.001, 0.01, 0.1]
results = {}

for lr in learning_rates:
	torch.manual_seed(7)

	model = nn.Linear(1, 1)
	optimizer = torch.optim.SGD(
		model.parameters(),
		lr
	)
	criterion = nn.MSELoss()

	losses = []

	for epochs in range(100):
		predictions = model(X)
		loss = criterion(predictions, y)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses.append(loss.item())

	results[lr] = losses

	print(
		f"lr={lr:<6}"
		f"initial_loss={losses[0]:.6f}"
		f"final_loss={losses[-1]:.6f}"
	)

	print(
		" final weight:",
		model.weight.item(),
		" final bias:",
		model.bias.item()
	)

# 3. Draw loss curve
plt.figure(figsize=(8, 5))

for lr, losses in results.items():
	plt.plot(
		range(1, len(losses) + 1),
		losses,
		label=f"lr={lr}",
	)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Learning Rate Comparison")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()

output_path = "06_optimization/learning_rate_comparison.png"
plt.savefig(output_path)

print(f"\nPlot saved to: {output_path}")
