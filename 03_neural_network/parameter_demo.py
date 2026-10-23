import torch.nn as nn

layer = nn.Linear(1,8)

print("Weight:")
print(layer.weight)

print("Weight shape:")
print(layer.weight.shape)

print("Bias:")
print(layer.bias)

print("Bias shape:")
print(layer.bias.shape)

print("Parameter count:")
print(sum(p.numel() for p in layer.parameters()))
