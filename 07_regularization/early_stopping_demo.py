
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ==================================
# 1. Create Dataset
# ==================================

x = torch.linspace(-3, 3, 120).unsqueeze(1)

y = torch.sin(x) + 0.15 * torch.randn_like(x)

# Random split
indices = torch.randperm(
    len(x),
    generator=torch.Generator().manual_seed(42)
)

train_indices = indices[:90]
val_indices = indices[90:]

x_train = x[train_indices]
y_train = y[train_indices]

x_val = x[val_indices]
y_val = y[val_indices]


# ==================================
# 2. Define Model
# ==================================

class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


model = MLP()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.005
)


# ==================================
# 3. Early Stopping Settings
# ==================================

epochs = 500

patience = 25
min_delta = 0.0001

best_val_loss = float("inf")
best_epoch = 0

counter = 0

best_model_state = None

train_losses = []
val_losses = []


# ==================================
# 4. Training Loop
# ==================================

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    prediction = model(x_train)

    loss = criterion(prediction, y_train)

    loss.backward()

    optimizer.step()

    # Evaluate
    model.eval()

    with torch.no_grad():

        train_prediction = model(x_train)
        train_loss = criterion(
            train_prediction,
            y_train
        )

        val_prediction = model(x_val)
        val_loss = criterion(
            val_prediction,
            y_val
        )

    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    current_val_loss = val_loss.item()

    # ==================================
    # 5. Early Stopping Logic
    # ==================================

    if current_val_loss < best_val_loss - min_delta:

        best_val_loss = current_val_loss
        best_epoch = epoch + 1

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        counter = 0

    else:

        counter += 1

    if (epoch + 1) % 20 == 0:

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss.item():.4f} "
            f"Val Loss: {current_val_loss:.4f} "
            f"Counter: {counter}"
        )

    if counter >= patience:

        print("\nEarly Stopping Triggered!")

        break


# ==================================
# 6. Restore Best Model
# ==================================

model.load_state_dict(best_model_state)

print("\n===== Final Result =====")

print("Best Epoch:", best_epoch)
print("Best Validation Loss:", best_val_loss)
print("Stopped Epoch:", len(train_losses))

# ==================================
# 7. Save Model
# ==================================

torch.save(
    model.state_dict(),
    "07_regularization/best_regularization_model.pth"
)

# ==================================
# 8. Plot Loss Curves
# ==================================

plt.figure(figsize=(10, 5))

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.axvline(
    best_epoch - 1,
    linestyle="--",
    label="Best Epoch"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Early Stopping Experiment")

plt.legend()
plt.grid()

plt.savefig(
    "07_regularization/early_stopping.png"
)

print("\nPlot saved successfully.")
