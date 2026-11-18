import torch


images = torch.zeros(3, 1, 6, 6)

images[0, 0, :, 2:4] = 1.0

images[1, 0, 1:3, :] = 1.0

images[2, 0, 3:5, :] = 1.0

labels = torch.tensor([
	[0.0],
	[1.0],
	[1.0],
])

print("Image shape:", images.shape)

print("\nImage 2:")
print(images[2, 0])

print("\nLabels:")
print(labels)

from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(images, labels)
loader = DataLoader(dataset, batch_size = 2, shuffle = False)

for batch_images, batch_labels in loader:
	print(batch_images.shape, batch_labels.flatten().tolist())

