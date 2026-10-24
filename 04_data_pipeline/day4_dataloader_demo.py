import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Prepare data
x_train = torch.tensor([[0.0],[1.0],[2.0],[3.0],[4.0]])

y_train = 3 * x_train + 2

# Define training set
class MyDataset(Dataset):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __len__(self):
		return len(self.x)

	def __getitem__(self, index):
		return self.x[index], self.y[index]

dataset = MyDataset(x_train, y_train)

# Create DataLoader
loader = DataLoader(
	dataset,
	batch_size = 2,
	shuffle = True
)

# Define model
model = nn.Linear(1,1)

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(
	model.parameters(),
	lr = 0.01
)

# Train model
for epoch in range(100):
	for batch_x, batch_y in loader:

		predictions = model(batch_x)

		loss = criterion(predictions, batch_y)

		optimizer.zero_grad()

		loss.backward()

		optimizer.step()

	if((epoch + 1) % 10 == 0):
		print(
			f"Epoch:{epoch + 1},"
			f"Loss:{loss.item():.6f}"
		)

# Test model
model.eval()

test_x = torch.tensor([
	[6.0],
	[7.0]
])

with torch.no_grad():
	test_predictions = model(test_x)

print("Test predictions:")
print(test_predictions)

print("Expected Value:")
print(3 * test_x + 2)

