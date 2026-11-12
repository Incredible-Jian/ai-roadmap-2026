import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

x = torch.tensor([
	[0.0],
	[1.0],
	[2.0],
	[3.0],
	[4.0],
])

y = torch.zeros_like(x)

dataset = TensorDataset(x, y)

loader = DataLoader(
	dataset,
	batch_size = 2,
	shuffle=False,
	drop_last=False,
)

criterion = nn.MSELoss()

batch_loss_sum = 0.0
weighted_loss_sum = 0.0
sample_count = 0

for batch_index, (batch_x, batch_y) in enumerate(loader):
	predictions = batch_x

	loss = criterion(predictions, batch_y)
	current_batch_size = batch_x.size(0)

	batch_loss_sum += loss.item()

	weighted_loss_sum += loss.item() * current_batch_size
	sample_count += current_batch_size

	print(
		f"Batch {batch_index + 1} | "
		f"Samples: {current_batch_size}"
		f"Mean loss: {loss.item():.4f}"
	)

simple_average = batch_loss_sum / len(loader)

weighted_average = weighted_loss_sum / sample_count

direct_loss = criterion(x, y).item()

print(f"\nSimple batch average: {simple_average:.4f}")
print(f"Weighted sample average: {weighted_average:.4f}")
print(f"Direct full-data loss: {direct_loss:.4f}")

