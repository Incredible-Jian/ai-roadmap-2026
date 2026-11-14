import torch

import torch.nn as nn

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# 1. Create a 6*6 Image
image = torch.tensor([
	[1, 1, 1, 0, 0, 0],
	[1, 1, 1, 0, 0, 0],
	[1, 1, 1, 0, 0, 0],
	[1, 1, 1, 0, 0, 0],
	[1, 1, 1, 0, 0, 0],
	[1, 1, 1, 0, 0, 0],
], dtype=torch.float32)

print("Original image shape:", image.shape)

inputs = image.unsqueeze(0).unsqueeze(0)

print("Model input shape:", inputs.shape)

# 2. a Input channel and a Out channel
conv = nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    stride=1,
    padding=0,
    bias=False,
)

kernel = torch.tensor([
	[1, 0, -1],
	[1, 0, -1],
	[1, 0, -1],
], dtype=torch.float32)

with torch.no_grad():
	conv.weight.copy_(
		kernel.unsqueeze(0).unsqueeze(0)
	)

print("Kernel weight shape:", conv.weight.shape)

# 3. Calculation
conv.eval()

with torch.no_grad():
	features = conv(inputs)
	activated = torch.relu(features)

print("\nFeature shape:", features.shape)

print("\nFeature map:")
print(features[0, 0])

print("\nAfter RuLU:")
print(activated[0, 0])

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

axes[0].imshow(
	image.numpy(),
	cmap="gray",
	vmin=0,
	vmax=1,
)
axes[0].set_title("Input image")

feature_plot = axes[1].imshow(
	features[0, 0].numpy(),
	cmap="coolwarm",
	vmin=3,
	vmax=3,
)
axes[1].set_title("Convolution output")

axes[2].imshow(
	activated[0, 0].numpy(),
	cmap="gray",
	vmin=0,
	vmax=3,
)
axes[2].set_title("After ReLU")

for ax in axes:
	ax.set_xticks([])
	ax.set_yticks([])

fig.colorbar(feature_plot, ax=axes[1], shrink=0.7)

fig.tight_layout()

fig.savefig("convolution_demo.png", dpi=150)

plt.close(fig)

print("\nFigure saved: convolution_demo.png")
