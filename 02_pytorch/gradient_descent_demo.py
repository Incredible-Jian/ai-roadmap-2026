import torch

w = torch.tensor(3.0,requires_grad=True)

optimizer = torch.optim.SGD([w], lr=0.1)

for step in range(5):
	optimizer.zero_grad()

	loss = w ** 2

	loss.backward()

	optimizer.step()

	print(
		f"Step {step + 1}: "
		f"w = {w.item():.4f},"
		f"loss = {loss.item():.4f}"
	)
