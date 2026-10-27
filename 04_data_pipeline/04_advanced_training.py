import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split

# 1. Define random seed
torch.manual_seed(42)

# 2. Construct data

# Generate 200 Input label, from 0 to 10
x = torch.linspace(0,10,200).reshape(-1,1)

# Add random noise
noise = torch.randn_like(x) * 1.5

# Construct label: y = 3x + 2 + noise
y = 3 * x + 2 + noise

print("Total samples:", len(x))

# 3. Define Dataset
class RegressionDataset(Dataset):

	def __init__(self, x, y):
		self.x = x
		self.y = y

		assert len(self.x) == len(self.y),\
			"Input and target sizes do not match!"

	def __len__(self):
		return len(self.x)

	def __getitem__(self, index):
		return self.x[index], self.y[index]

dataset = RegressionDataset(x, y)

# 4. Split into training set, validation set and testing set
train_size = 140
val_size = 30
test_size = 30

train_dataset, val_dataset, test_dataset = 
random_split(
	dataset,
	[train_size, val_size, test_size]
)
