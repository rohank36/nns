"""
Toy example: writing a custom PyTorch optimizer.

To create your own optimizer, subclass `torch.optim.Optimizer` and implement `step()`.
This file demonstrates with a simple SGD-with-momentum from scratch.
"""

import torch
from torch.optim import Optimizer


class MyAdamW(Optimizer):
    """AdamW optimizer (Adam with decoupled weight decay) written from scratch.

    Implements the algorithm from 'Decoupled Weight Decay Regularization'
    (Loshchilov & Hutter, 2019). Unlike L2 regularization in standard Adam,
    weight decay is applied directly to the parameters, not to the gradient.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state on first step
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)  # first moment
                    state["v"] = torch.zeros_like(p)  # second moment

                state["step"] += 1
                m, v = state["m"], state["v"]

                # Update biased moments
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                m_hat = m / (1 - beta1 ** state["step"])
                v_hat = v / (1 - beta2 ** state["step"])

                # Decoupled weight decay (applied to params, not gradient)
                p.mul_(1 - lr * weight_decay)

                # Adam update
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

        return loss


class MySGD(Optimizer):
    """A minimal SGD + momentum optimizer written from scratch."""

    def __init__(self, params, lr=0.01, momentum=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure: A callable that re-evaluates the model and returns the loss.
                     Optional for most optimizers.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Momentum: keep a running velocity in optimizer state
                if momentum != 0.0:
                    state = self.state[p]
                    if "velocity" not in state:
                        state["velocity"] = torch.zeros_like(p)
                    v = state["velocity"]
                    v.mul_(momentum).add_(grad)
                    p.add_(v, alpha=-lr)
                else:
                    p.add_(grad, alpha=-lr)

        return loss


# ---------------------------------------------------------------------------
# Quick demo: train a tiny network with our custom optimizer
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    # Toy dataset: learn y = 2x + 1
    x = torch.randn(100, 1)
    y = 2 * x + 1 + 0.1 * torch.randn(100, 1)

    model = torch.nn.Linear(1, 1)
    criterion = torch.nn.MSELoss()

    # Use our custom optimizer exactly like any built-in one
    optimizer = MySGD(model.parameters(), lr=0.05, momentum=0.9)

    for epoch in range(200):
        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"epoch {epoch+1:>3d}  loss={loss.item():.4f}")

    w = model.weight.item()
    b = model.bias.item()
    print(f"\nLearned: y = {w:.3f}x + {b:.3f}  (target: y = 2x + 1)")
