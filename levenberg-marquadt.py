from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# Step #1

# Parameters
R = 4
k = 8
N = R
S = k * N
Nt = S * N
# Initialization
x0 = 0
y0 = 0
X = torch.zeros(N, S)
Y = torch.zeros(N, S)
regions = torch.zeros(N, S, dtype=torch.long)
# Spirals coordinates and regions
for s in range(1, S + 1):
    r = torch.floor((torch.tensor(s) - 1) * R / S + 1)
    for n in range(1, N + 1):
        a0 = n / N * 2 * torch.pi / k
        a = torch.tensor(a0 + s * 2 * torch.pi / S)
        b = a0
        X[n - 1, s - 1] = b * torch.cos(a) + x0
        Y[n - 1, s - 1] = b * torch.sin(a) + y0
        regions[n - 1, s - 1] = r
coordinates = torch.stack([X.ravel(), Y.ravel()], dim=1)
regions = regions.ravel()
# Figure
colors = ["magenta", "lime", "cyan", "yellow"]
plt.figure(figsize=(7, 7))
for r in range(1, R + 1):
    i = regions == r
    plt.scatter(
        coordinates[i, 0],
        coordinates[i, 1],
        c=colors[r - 1],
        edgecolors="k",
        s=60,
    )
# Plot
plt.xlim(X.min().item(), X.max().item())
plt.ylim(Y.min().item(), Y.max().item())
plt.axis("equal")
# plt.show()
# Probabilities
probabilities = torch.zeros(R, torch.numel(regions))
for n in range(torch.numel(regions)):
    probabilities[regions[n] - 1][n] = 1
# Multi-layer perceptron
MLP = nn.Sequential(
    nn.Linear(2, 16),
    nn.Tanh(),
    nn.Linear(16, 8),
    nn.Tanh(),
    nn.Linear(8, R),
    nn.Tanh(),
    nn.Linear(R, R),
    nn.Softmax(dim=1),
)
