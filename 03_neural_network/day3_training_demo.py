import torch
import torch.nn as nn

x_train = torch.tensor([
[0.0], [1.0], [2.0], [3.0], [4.0]
])

y_train = torch.tensor([
[2.0], [5.0], [8.0], [11.0], [14.0]
])

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

model = SimpleNet()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(
	model.parameters(),
	lr = 0.01
)

for epoch in range(1000):
	predictions = model(x_train)

	loss = criterion(predictions, y_train)

	optimizer.zero_grad()

	loss.backward()

	optimizer.step()

	if(epoch % 100 == 0):
		print(
			f"Epoch:{epoch + 1},"
			f"Loss:{loss.item():.6f}"
		)

model.eval()

test_x = torch.tensor([[6.0], [7.0]])

with torch.no_grad():
	test_predictions = model(test_x)

print("test_x:")
print(test_x)

print("Expected predictions:")
print(3 * test_x + 2)

print("test_predictions:")
print(test_predictions)
