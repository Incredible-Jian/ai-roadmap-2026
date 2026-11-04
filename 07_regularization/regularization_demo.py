import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

# 1. Create noisy training data
x = torch.linspace(-3, 3, 120).unsqueeze(1)
y = torch.sin(x) + 0.15 * torch.randn_like(x)

# split into training set and validation set
x_train, y_train = x[:90], y[:90]
x_val, y_val = x[90:], y[90:]

train_loader = DataLoader(
	TensorDataset(x_train, y_train),
	batch_size = 16,
	shuffle = True
)

# Define model
class MLP(nn.Module):
	def __init__(self, dropout=0.0):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(1, 128),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, 1)
		)

	def forward(self, x):
		return self.net(x)

# Training function
def train_model(name, weight_decay = 0.0, dropout = 0.0):
	model = MLP(dropout)
	optimizer = torch.optim.Adam(
		model.parameters(),
		lr = 0.005,
		weight_decay = weight_decay
	)
	criterion = nn.MSELoss()

	train_losses = []
	val_losses = []

	for epoch in range(200):
		model.train()
		total_loss = 0

		for xb, yb in train_loader:
			optimizer.zero_grad()
			prediction = model(xb)
			loss = criterion(prediction, yb)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()

		train_losses.append(total_loss / len(train_loader))

		model.eval()
		with torch.no_grad():
			val_pred = model(x_val)
			val_loss = criterion(val_pred, y_val)

		val_losses.append(val_loss.item())

	print(
		f"{name}:"
		f"Train Loss={train_losses[-1]:4f},"
		f"Val Loss={val_losses[-1]:.4f}"
	)

	return train_losses, val_losses

# Compare three model
results = {}

results["Baseline"] = train_model("Baseline")
results["L2"] = train_model("L2",weight_decay=0.01)
results["Dropout"] = train_model("Dropout",dropout=0.3)

# plot Val Loss
plt.figure(figsize=(10, 5))

for name, (_, val_losses) in results.items():
	plt.plot(val_losses, label = name)

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Regularization Comparison")
plt.legend()
plt.grid()
plt.savefig("07_regularization/comparison.png")
plt.show()

