import copy

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 1. Reproducibility: CPU experiment
SEED = 42
torch.manual_seed(SEED)


# 2. Make balanced synthetic images: 0=vertical, 1=horizontal
sample_count = 600
images = torch.zeros(sample_count, 1, 16, 16)
labels = (torch.arange(sample_count) % 2).float().reshape(-1, 1)

for index in range(sample_count):
    position = torch.randint(2, 13, (1,)).item()
    if labels[index].item() == 0:
        images[index, 0, :, position:position + 2] = 1.0
    else:
        images[index, 0, position:position + 2, :] = 1.0

images = (images + 0.15 * torch.randn_like(images)).clamp(0.0, 1.0)


# 3. Split samples, then construct loaders
dataset = TensorDataset(images, labels)
split_generator = torch.Generator().manual_seed(SEED)
train_set, val_set, test_set = random_split(
    dataset, [360, 120, 120], generator=split_generator
)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
print("Train / Validation / Test:", len(train_set), len(val_set), len(test_set))


# 4. Conv weights are learned, rather than copied from a hand-written kernel
model = nn.Sequential(
    nn.Conv2d(1, 4, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Flatten(),
    nn.Linear(4 * 8 * 8, 1),
)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

example = images[:2]
print("\nShape changes:", example.shape)
with torch.no_grad():
    for layer in model:
        example = layer(example)
        print(type(layer).__name__, "->", example.shape)


# 5. Evaluate a fixed model; no parameter updates
def evaluate(loader):
    model.eval()
    loss_sum = 0.0
    correct_count = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            predictions = (logits >= 0).float()
            size = batch_x.size(0)
            loss_sum += loss.item() * size
            correct_count += (predictions == batch_y).sum().item()
            total += size
    return loss_sum / total, correct_count / total


# 6. Train and select a checkpoint using validation loss
best_val_loss = float("inf")
best_state = None
best_epoch = 0
initial_val_loss, initial_val_acc = evaluate(val_loader)
print(f"\nBefore training | Val loss {initial_val_loss:.4f} | Val acc {initial_val_acc:.2%}")

for epoch in range(1, 21):
    model.train()
    loss_sum = 0.0
    sample_total = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        size = batch_x.size(0)
        loss_sum += loss.item() * size
        sample_total += size

    val_loss, val_acc = evaluate(val_loader)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
    if epoch == 1 or epoch % 5 == 0:
        print(
            f"Epoch {epoch:2d}/20 | Train process loss {loss_sum / sample_total:.4f} | "
            f"Val loss {val_loss:.4f} | Val acc {val_acc:.2%}"
        )


# 7. Restore the selected model; recompute all final metrics
model.load_state_dict(best_state)
train_loss, train_acc = evaluate(train_loader)
val_loss, val_acc = evaluate(val_loader)
test_loss, test_acc = evaluate(test_loader)
print(f"\nSeed: {SEED} | Best epoch: {best_epoch}")
print(f"Train | Loss {train_loss:.4f} | Acc {train_acc:.2%}")
print(f"Val   | Loss {val_loss:.4f} | Acc {val_acc:.2%}")
print(f"Test  | Loss {test_loss:.4f} | Acc {test_acc:.2%}")


# 8. Show 8 test samples; sigmoid is only needed to display probabilities
sample_images = torch.stack([test_set[i][0] for i in range(8)])
with torch.no_grad():
    probabilities = torch.sigmoid(model(sample_images))
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for index, ax in enumerate(axes.flat):
    true_label = int(test_set[index][1].item())
    probability = probabilities[index].item()
    predicted_label = int(probability >= 0.5)
    ax.imshow(sample_images[index, 0].numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"True {true_label} / Pred {predicted_label}\nP(horizontal)={probability:.3f}")
    ax.axis("off")
fig.tight_layout()
figure_path = f"stripe_predictions_seed_{SEED}.png"
fig.savefig(figure_path, dpi=150)
plt.close(fig)
print("Figure saved:", figure_path)
