import torch

weights = torch.tensor(
	[2.0, -3.0, 0.5],
	requires_grad=True
)

lambda_reg = 0.01

print("Initial weights:", weights.detach())

l1_loss = lambda_reg * weights.abs().sum()

l1_loss.backward()

print("L1 Loss:", l1_loss.item())
print("L1 Gradient:", weights.grad)

l1_gradient = weights.grad.clone()

weights.grad.zero_()

l2_loss = lambda_reg / 2 * (weights ** 2).sum()

l2_loss.backward()

print("L2 Loss:", l2_loss.item())
print("L2 Gradient:", weights.grad)


learning_rate = 0.1

l1_new_weights = (
	weights.detach() - learning_rate * l1_gradient
)

l2_gradient = weights.grad.clone()

l2_new_weights = (
	weights.detach() - learning_rate * l2_gradient
)

print("Original:", weights.detach())
print("After L1:", l1_new_weights)
print("After L2:", l2_new_weights)

