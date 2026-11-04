
import torch

# 1. Initialize parameters
initial = torch.tensor([2.0, -3.0, 0.5])

w_adam = torch.nn.Parameter(initial.clone())
w_adamw = torch.nn.Parameter(initial.clone())

# 2. Set the same data gradient
data_gradient = torch.tensor([-0.1, 0.1, -0.05])

learning_rate = 0.01
weight_decay = 0.1

# 3. Create optimizers
adam = torch.optim.Adam(
    [w_adam],
    lr=learning_rate,
    weight_decay=weight_decay
)

adamw = torch.optim.AdamW(
    [w_adamw],
    lr=learning_rate,
    weight_decay=weight_decay
)

# 4. Assign gradients
w_adam.grad = data_gradient.clone()
w_adamw.grad = data_gradient.clone()

print("===== Before Update =====")
print("Initial:", initial)
print("Data Gradient:", data_gradient)

# Theoretical gradient used by coupled L2 in Adam
effective_gradient = data_gradient + weight_decay * initial

print("Adam effective gradient:", effective_gradient)

# 5. Update parameters
adam.step()
adamw.step()

print("\n===== After Update =====")
print("Adam:", w_adam.detach())
print("AdamW:", w_adamw.detach())

print("\n===== Parameter Changes =====")
print("Adam Change:", w_adam.detach() - initial)
print("AdamW Change:", w_adamw.detach() - initial)
