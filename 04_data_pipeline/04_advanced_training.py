import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split

# 1. Define random seed
torch.manual_seed(42)

# 2. Construct data

# Generate 200 Input label, from 0 to 10
x = torch.linspace(0,10,200).reshape(-1,1)

# Add random noise
noise = torch.randn_like(x) * 1.5

# Construct label: y = 3x + 2 + noise
y = 3 * x + 2 + noise

print("Total samples:", len(x))

# 3. Define Dataset
class RegressionDataset(Dataset):

	def __init__(self, x, y):
		self.x = x
		self.y = y

		assert len(self.x) == len(self.y),\
			"Input and target sizes do not match!"

	def __len__(self):
		return len(self.x)

	def __getitem__(self, index):
		return self.x[index], self.y[index]

dataset = RegressionDataset(x, y)

# 4. Split into training set, validation set and testing set
train_size = 140
val_size = 30
test_size = 30

train_dataset, val_dataset, test_dataset = random_split(
	dataset,
	[train_size, val_size, test_size]
)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

# Create DataLoader
train_loader = DataLoader(
	train_dataset,
	batch_size = 16,
	shuffle = True,
	drop_last = False
)

val_loader = DataLoader(
        val_dataset,
        batch_size = 30,
        shuffle = False
)

test_loader = DataLoader(
        test_dataset,
        batch_size = 30,
        shuffle = False
)


print("Batches per training epoch:", len(train_loader))

# 6. Define model
class RegressionModel(nn.Module):

	def __init__(self):
		super().__init__()

		self.network = nn.Sequential(
			nn.Linear(1,16),
			nn.ReLU(),
			nn.Linear(16,1)
		)

	def forward(self, x):
		return self.network(x)

model = RegressionModel()

# 7. Define Loss function and optimizer
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
	model.parameters(),
	lr = 0.01
)

# 8. Training
epochs = 200

# Record lowest Loss of validation set
best_val_loss = float("inf")

# 9. Start training
for epoch in range(epochs):
	# Training level

	model.train()

	train_loss_sum = 0.0
	train_sample_count = 0

	for batch_x, batch_y in train_loader:
		predictions = model(batch_x)

		loss = criterion(predictions, batch_y)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		batch_size = batch_x.size(0)

		train_loss_sum += loss.item() * batch_size
		train_sample_count += batch_size

	avg_train_loss = train_loss_sum / train_sample_count


	# Evaling level
	model.eval()

	val_loss_sum = 0.0
	val_sample_count = 0

	with torch.no_grad():

		for batch_x, batch_y in val_loader:

			predictions = model(batch_x)

			loss = criterion(predictions, batch_y)

			batch_size = batch_x.size(0)

			val_loss_sum += loss.item() * batch_size
			val_sample_count += batch_size

	avg_val_loss = val_loss_sum / val_sample_count

	# Save the best model
	if avg_val_loss < best_val_loss:

		best_val_loss = avg_val_loss

		torch.save(
			model.state_dict(),
			"best_model.pth"
		)

		print(
			f"Best model updated: "
			f"Epoch {epoch + 1},"
			f"Val Loss {best_val_loss:.4f}"
		)

	# Print training result
	if (epoch + 1) % 20 == 0:

		print(
			f"Epoch{epoch + 1:3d} / "
			f"Train Loss: {avg_train_loss:.4f} / "
			f"Val Loss: {avg_val_loss:.4f}"
		)

# 10. Load best model

state_dict = torch.load(
	"best_model.pth",
	map_location = "cpu",
	weights_only = True
)

model.load_state_dict(state_dict)

print("\nBest model loaded.")

# 11. Use test set to eval finally
model.eval()

test_loss_sum = 0.0
test_sample_count = 0

with torch.no_grad():

	for batch_x, batch_y in test_loader:

		predictions = model(batch_x)

		loss = criterion(predictions, batch_y)

		batch_size = batch_x.size(0)

		test_loss_sum += loss.item() * batch_size
		test_sample_count += batch_size

avg_test_loss = test_loss_sum / test_sample_count

print(f"Final Test Loss: {avg_test_loss:.4f}")

# 12. Predict new data
new_x = torch.tensor([[2.0], [5.0], [8.0]])

with torch.no_grad():

	new_predictions = model(new_x)

print("\nNew inputs:")
print(new_x)

print("\n Model predictions:")
print(new_predictions)

print("\nIdeal values without noise:")
print(3 * new_x + 2)
