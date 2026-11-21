import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


torch.manual_seed(42)
class_names = ["vertical", "horizontal", "diagonal"]

# 1. Draw three kinds of stripes with varied positions and add noise.
images = torch.zeros(600, 1, 16, 16)
labels = torch.arange(len(images)) % len(class_names)
rows = torch.arange(16).reshape(16, 1)
columns = torch.arange(16).reshape(1, 16)

for index, label in enumerate(labels.tolist()):
    position = torch.randint(3, 12, (1,)).item()
    if label == 0:
        images[index, 0, :, position:position + 2] = 1
    elif label == 1:
        images[index, 0, position:position + 2, :] = 1
    else:
        offset = position - 7
        images[index, 0] = ((columns - rows - offset).abs() <= 1).float()

images = (images + 0.15 * torch.randn_like(images)).clamp(0, 1)
dataset = TensorDataset(images, labels)
train_set, val_set, test_set = random_split(
    dataset, [360, 120, 120], generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=32)

# 2. Keep the feature extractor and output one logit per class.
model = nn.Sequential(
    nn.Conv2d(1, 4, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(4 * 8 * 8, len(class_names)),
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)


def evaluate(loader):
    model.eval()
    total_loss = 0.0
    total = 0
    matrix = torch.zeros(len(class_names), len(class_names), dtype=torch.long)

    with torch.no_grad():
        for batch_images, batch_labels in loader:
            logits = model(batch_images)
            predictions = logits.argmax(dim=1)
            count = batch_labels.size(0)
            total_loss += criterion(logits, batch_labels).item() * count
            total += count
            for actual, predicted in zip(batch_labels.tolist(), predictions.tolist()):
                matrix[actual, predicted] += 1

    accuracy = matrix.diag().sum().item() / total
    return total_loss / total, accuracy, matrix


# 3. Train on batches and save the parameters with the lowest validation loss.
best_val_loss = float("inf")
best_state = None
best_epoch = 0

for epoch in range(1, 16):
    model.train()
    for batch_index, (batch_images, batch_labels) in enumerate(train_loader):
        optimizer.zero_grad()
        logits = model(batch_images)
        if epoch == 1 and batch_index == 0:
            print("Images:", batch_images.shape)
            print("Logits:", logits.shape)
            print("Labels:", batch_labels.shape, batch_labels.dtype)
            print("Label IDs in first batch:", torch.unique(batch_labels).tolist())
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

    val_loss, val_acc, _ = evaluate(val_loader)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
    if epoch == 1 or epoch % 5 == 0:
        print(f"Epoch {epoch:2d}/15 | Val loss {val_loss:.4f} | Val acc {val_acc:.2%}")

# 4. Evaluate the selected model on the test set once.
model.load_state_dict(best_state)
test_loss, test_acc, confusion = evaluate(test_loader)
print(f"Best epoch: {best_epoch}")
print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.2%}")
print("Class order:", class_names)
print("Confusion matrix (rows=true, columns=predicted):")
print(confusion)
print("Test samples:", confusion.sum().item())
