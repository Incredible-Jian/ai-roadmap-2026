import torch
# 1. Create a tensor
x = torch.tensor([1.0,2.0,3.0])

print("Tensor x:")
print(x)

# 2. Perform basic operations
y = x * 2

print("Tensor y = x * 2:")
print(y)

# 3. Calculate the sum
z = y.sum()

print("Sum of y:")
print(z)

# 4. Check the shape and data type
print("Shape of x:", x.shape)
print("Data type of x:", x.dtype)
