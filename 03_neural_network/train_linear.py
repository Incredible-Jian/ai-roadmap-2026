import torch
import torch.nn as nn
# 1.Prepare training data
x_train = torch.tensor(
	[[1.0],[2.0],[3.0],[4.0],[5.0]]
)

y_train = torch.tensor(
	[[3.0],[5.0],[7.0],[9.0],[11.0]]
)

# 2. Define the neural network
class SimpleNet(nn.Module):
	def __init__(self):
		super().__init__()

		self.layer1 = nn.Linear(1,8)
		self.relu = nn.ReLU()
		self.layer2 = nn.Linear(8,1)

	def forward(self, x):
		x = self.layer1(x)
		x = self.relu(x)
		x = self.layer2(x)
		return x

# 3. Create the model
model = SimpleNet()

# 4. Define Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(
	model.parameters(),
	lr = 0.01
)

# 5. Train the model
for epoch in range(1000):
	# Forward pass
	predictions = model(x_train)

	# Calculate loss
	loss = criterion(predictions, y_train)

	# Clear the old gradients
	optimizer.zero_grad()

	# Backpropagation
	loss.backward()

	# Update model parameters
	optimizer.step()

	if (epoch % 100 == 0):
		print(
			f"Epoch {epoch + 1},"
			f"Loss: {loss.item():.6f}"
		)

# 6. Test the trained model
model.eval()

with torch.no_grad():
	test_x = torch.tensor([[6.0],[7.0]])
	test_predictions = model(test_x)

print("\nTest inputs:")
print(test_x)

print("Predictions:")
print(test_predictions)

print("Expected outputs:")
print(2 * test_x + 1)
