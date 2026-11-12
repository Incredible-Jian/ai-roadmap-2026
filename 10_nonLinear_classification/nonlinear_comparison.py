import copy

import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


SEED = 42
EPOCHS = 500
LEARNING_RATE = 0.02

torch.manual_seed(SEED)

x = torch.rand(800, 2) * 2 - 1

y = (
	x[:, 0] * x[:, 1] < 0
).float().unsqueeze(1)

indices = torch.randperm(len(x))

train_ids = indices[:480]
val_ids = indices[480:640]
test_ids = indices[640:]

x_train = x[train_ids]
y_train = y[train_ids]

x_val = x[val_ids]
y_val = y[val_ids]

x_test = x[test_ids]
y_test = y[test_ids]

print("Train", x_train.shape, y_train.shape)
print("Validation", x_val.shape, y_val.shape)
print("Test", x_test.shape, y_test.shape)

def make_model(name):
	torch.manual_seed(SEED)

	if name == "Linear":
		return nn.Linear(2, 1)

	if name == "StackedLinear":
		return nn.Sequential(
			nn.Linear(2, 16),
			nn.Identity(),
			nn.Linear(16, 1)
		)

	return nn.Sequential(
		nn.Linear(2, 16),
		nn.ReLU(),
		nn.Linear(16, 1)
	)

criterion = nn.BCEWithLogitsLoss()

def evaluate(model, inputs, labels):
	model.eval()

	with torch.no_grad():
		logits = model(inputs)

		loss = criterion(
			logits, labels
		)

		probabilities = torch.sigmoid(logits)

		predictions = (
			probabilities >= 0.5
		).float()

		accuracy = (
			predictions == labels
		).float().mean()

	return loss.item(), accuracy.item()

def train_model(name):
	model = make_model(name)

	optimizer = torch.optim.Adam(
		model.parameters(),
		lr = LEARNING_RATE
	)

	best_val_loss = float("inf")
	best_epoch = 0
	best_state = None

	val_losses = []

	for epoch in range(1, EPOCHS + 1):
		model.train()

		optimizer.zero_grad()

		logits = model(x_train)

		loss = criterion(logits, y_train)

		loss.backward()

		optimizer.step()

		val_loss, val_acc = evaluate(
			model, x_val, y_val
		)

		val_losses.append(val_loss)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_epoch = epoch

			best_state = copy.deepcopy(
				model.state_dict()
			)
		if epoch % 100 == 0:
			print(
				f"{name} | Epoch {epoch} | "
				f"Val loss {val_loss:.4f} | "
				f"Val acc {val_acc:.2%} | "
			)
	model.load_state_dict(best_state)

	train_loss, train_acc = evaluate(
		model, x_train, y_train
	)

	return {
		"name": name,
		"model": model,
		"best_epoch": best_epoch,
		"train_acc": train_acc,
		"val_acc": val_acc,
		"val_losses": val_losses,
	}

names = [
	"Linear",
	"StackedLinear",
	"MLP_ReLU"
]

results = []

for name in names:
	result = train_model(name)
	results.append(result)

print("\n========== Final Results ==========")

for result in results:
    test_loss, test_acc = evaluate(
        result["model"],
        x_test,
        y_test
    )

    print(
        f"{result['name']} | "
        f"Best epoch {result['best_epoch']} | "
        f"Train acc {result['train_acc']:.2%} | "
        f"Val acc {result['val_acc']:.2%} | "
        f"Test acc {test_acc:.2%} | "
        f"Test loss {test_loss:.4f}"
    )

fig, ax = plt.subplots(
    figsize=(8, 4)
)

for result in results:
    ax.plot(
        range(1, EPOCHS + 1),
        result["val_losses"],
        label=result["name"]
    )

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation BCE")
ax.set_title("Nonlinear Classification")
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()

fig.savefig(
    f"day10_loss_seed_{SEED}.png",
    dpi=150
)

plt.close(fig)

print("Loss figure saved.")

axis = torch.linspace(-1, 1, 150)

grid_x1, grid_x2 = torch.meshgrid(axis, axis)
grid_inputs = torch.stack([grid_x1.reshape(-1), grid_x2.reshape(-1)], dim=1)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, result in zip(axes, results):
	model = result["model"]
	model.eval()
	with torch.no_grad():
		grid_predictions = (model(grid_inputs) >= 0).float()

	ax.contourf(
		grid_x1.numpy(), grid_x2.numpy(),
		grid_predictions.reshape(grid_x1.shape).numpy(),
		levels=[-0.5, 0.5, 1.5], cmap="coolwarm", alpha=0.2
	)
	ax.scatter(
		x_test[:, 0].numpy(), x_test[:, 1].numpy(),
		c=y_test.squeeze(1).numpy(), cmap="coolwarm", vmin=0, vmax=1,
		s=12, edgecolors="none"
	)

	ax.axhline(0, color="black", linestyle="--", linewidth=1)
	ax.axvline(0, color="black", linestyle="--", linewidth=1)
	ax.set_title(result["name"])
	ax.set_xlabel("x1")
	ax.set_ylabel("x2")
	ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(f"day10_boundary_seed_{SEED}.png", dpi=150)
plt.close(fig)
print("Saved loss and boundary figure")
