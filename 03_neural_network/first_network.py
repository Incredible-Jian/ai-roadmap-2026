import torch
import torch.nn as nn

# Define a simple neural network
class SimpleNet(nn.Module):
	def __init__(self):
		super().__init__()

		self.layer1 = nn.Linear(1, 8)
		self.relu = nn.ReLU()
		self.layer2 = nn.Linear(8,1)

	def forward(self, x):
		x = self.layer1(x)
		x = self.relu(x)
		x = self.layer2(x)
		return x

# Create the model
model = SimpleNet()

# Create input data
x = torch.tensor([[1.0],[2.0],[3.0]])

# Make predictions
predictions = model(x)

print("Input:")
print(x)

print("Predictions:")
print(predictions)

print("Model parameters:")
for name, param in model.named_parameters():
	print(name, param.shape)
