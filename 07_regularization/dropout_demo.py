import torch
import torch.nn as nn

torch.manual_seed(42)

x = torch.ones(100)

dropout = nn.Dropout(p=0.5)

dropout.train()

output1 = dropout(x)
output2 = dropout(x)

print("======== Training Mode =========")

print("Original:")
print(x)

print("\nFirst Output:")
print(output1)

print("\nSecond Output:")
print(output2)

zero_count1 = (output1 == 0).sum().item()
zero_count2 = (output2 == 0).sum().item()

print("\nFirst zero count:", zero_count1)
print("\nSecond zero count:", zero_count2)

dropout.eval()

output_eval = dropout(x)

print("\n========= Evaluation Mode ========")

print("Evaluation Output:")
print(output_eval)

print(
	"Are outputs equal to original:",
	torch.equal(output_eval, x)
)
