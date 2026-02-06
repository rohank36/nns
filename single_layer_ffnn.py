import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Config ────────────────────────────────────────────────────
INPUT_DIM = 4
OUTPUT_DIM = 2
HIDDEN_DIM = 32          # width of hidden layers
NUM_HIDDEN = 0            # 0 = single linear layer (no hidden layers)
ACTIVATION = "relu"       # relu, tanh, gelu
DROPOUT = 0.0             # 0.0 = no dropout
BATCH_NORM = False        # batch normalization between layers
WEIGHT_DECAY = 0.0        # L2 regularization (set > 0 to enable)
LR = 0.1
EPOCHS = 20
BATCH_SIZE = 32
# ──────────────────────────────────────────────────────────────

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}


class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden,
                 activation="relu", dropout=0.0, batch_norm=False):
        super().__init__()
        act_fn = ACTIVATIONS[activation]
        layers = []

        if num_hidden == 0:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # input -> first hidden
            layers.append(nn.Linear(input_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            # additional hidden layers
            for _ in range(num_hidden - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(act_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

            # last hidden -> output
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Data ──────────────────────────────────────────────────────
torch.manual_seed(42)
X = torch.randn(200, INPUT_DIM)
y = (X[:, 0] + X[:, 1] > 0).long()

X_train, X_test = X[:160], X[160:]
y_train, y_test = y[:160], y[160:]
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

# ── Model / Optimizer / Loss ──────────────────────────────────
model = FFNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_HIDDEN,
             activation=ACTIVATION, dropout=DROPOUT, batch_norm=BATCH_NORM)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.CrossEntropyLoss()

print(model)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# ── Train ─────────────────────────────────────────────────────
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        loss = loss_fn(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d} | Loss: {total_loss / len(train_loader):.4f}")

# ── Evaluate ──────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    preds = model(X_test).argmax(dim=1)
    acc = (preds == y_test).float().mean()
    print(f"\nTest Accuracy: {acc:.2%}")
