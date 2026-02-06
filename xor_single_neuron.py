import torch
import torch.nn as nn
import torch.optim as optim

# Set seed for reproducibility
torch.manual_seed(27)

# Simple feedforward network with 1 hidden neuron
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.hidden = nn.Linear(2, 2)  # 2 inputs -> 1 hidden neuron
        self.output = nn.Linear(2, 1)  # 1 hidden -> 1 output

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x

def print_model_parameters(model):
    print("\nModel parameters:")
    print("\nHidden layer:")
    print(f"  Weights: {model.hidden.weight.data}")
    print(f"  Bias: {model.hidden.bias.data}")
    print("\nOutput layer:")
    print(f"  Weights: {model.output.weight.data}")
    print(f"  Bias: {model.output.bias.data}")

def kl_divergence(model1, model2, X):
    """
    Calculate KL divergence between output distributions of two models.
    KL(model1 || model2) = sum(P * log(P/Q))
    where P is model1's output and Q is model2's output.
    """
    with torch.no_grad():
        p = model1(X)
        q = model2(X)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        p = torch.clamp(p, eps, 1 - eps)
        q = torch.clamp(q, eps, 1 - eps)

        # KL divergence for binary classification
        kl = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))

    return kl.mean().item()

def train(model,criterion,optimizer,X,y,epochs):
    for epoch in range(epochs):
        # Forward pass
        predictions = model(X)
        loss = criterion(predictions, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    return model

def get_final_predictions(model,X,y):
    with torch.no_grad():
        predictions = model(X)
        for i in range(len(X)):
            print(f"Input: {X[i].numpy()}, Target: {y[i].item()}, Prediction: {predictions[i].item():.4f}")

# XOR dataset
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Initialize model, loss, and optimizer
criterion = nn.BCELoss()

m1 = XORNet()
o1 = optim.Adam(m1.parameters(), lr=0.1)

#moptimal = XORNet()
#o2 = optim.Adam(moptimal.parameters(), lr=0.01)

print("TRAINED\n\n")
m1 = train(m1,criterion,o1,X,y,1000)
print_model_parameters(m1)

#moptimal = train(moptimal,criterion,o2,X,y,10000)
#print_model_parameters(moptimal)

print("==============================\n\n")

get_final_predictions(m1,X,y)
print("\n")
#get_final_predictions(moptimal,X,y)

#print(kl_divergence(m1,moptimal,X))