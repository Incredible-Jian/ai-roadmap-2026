import torch

# Create a tensor and enable gradient calculation
x = torch.tensor([1.0,2.0,3.0], requires_grad=True)

# Calcultate y = x^2
y = x ** 2

# Sum all elements
z = y.sum()

# Calculate gradients
z.backward()

print("x:", x)
print("z:", z)
print("Gradient of x:", x.grad)
