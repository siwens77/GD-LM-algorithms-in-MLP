from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Step #1

# Parameter
N = 24
# Coordinates
V = torch.linspace(-2, +2, N)
[X, Y] = torch.meshgrid(V, V, indexing="xy")
coordinates = torch.stack([X.ravel(), Y.ravel()], dim=1)
# Regions
cond1 = Y >= X
cond2 = (Y > -X) | (torch.abs(X + Y) < 1e-2)
regions = 1 * cond1 + 2 * cond2 + 1
regions = regions.ravel()
# Figure
plt.figure(facecolor="w")
plt.axis("off")
colors = ["magenta", "lime", "cyan", "yellow"]
# Plot
for r in range(1, 5):
    i = regions == r
    plt.scatter(
        coordinates[i, 0],
        coordinates[i, 1],
        c=colors[r - 1],
        edgecolors="k",
        s=60,
    )
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.axis("equal")

# Step #2

# Probabilities
probabilities = torch.tensor([[0.0] * 4 for i in range(len(regions))])
for n in range(len(regions)):
    probabilities[n][regions[n] - 1] = 1.0
# Multi-layer perceptron
MLP = nn.Sequential(
    nn.Linear(2, 4),
    nn.Linear(4, 4),
    nn.Linear(4, 4),
    nn.Softmax(dim=1),
)
# Options
epochs = 50
trainingAlgorithm = optim.SGD(MLP.parameters(), lr=0.1, momentum=0.9)
lossFunction = nn.MSELoss()

fullData = TensorDataset(coordinates, probabilities)
sampleSize = len(coordinates)
trainSize = int(0.8 * sampleSize)
testSize = sampleSize - trainSize

trainData, testData = random_split(fullData, [trainSize, testSize])
trainLoader = DataLoader(trainData, batch_size=10, shuffle=True)
testLoader = DataLoader(testData, shuffle=True)

# Training
for epoch in range(epochs):
    for batchCoord, batchProb in trainLoader:
        trainingAlgorithm.zero_grad()
        y_pred = MLP(batchCoord)
        loss = lossFunction(y_pred, batchProb)
        loss.backward()
        trainingAlgorithm.step()
    inputs, outputs = next(iter(testLoader))
    testY = MLP(inputs)
    testloss = lossFunction(testY, outputs)
    if loss.item() < 0.01:
        break

# Step #3

# Map colors
Colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
# Coordinates of the coloration points
N2 = 300
V = torch.linspace(-2, +2, N2)
[X, Y] = torch.meshgrid(V, V, indexing="xy")
coordinates2 = torch.stack([X.ravel(), Y.ravel()], dim=1)
# Most propable regions
with torch.no_grad():
    probabilities2 = MLP(coordinates2)
_, regions2 = torch.max(probabilities2, dim=1)
# Coloration
gridLabels = regions2.reshape(300, 300).numpy()
plt.figure(figsize=(8, 8))
plt.imshow(gridLabels, extent=[-2, 2, -2, 2], origin="lower", alpha=0.25)
plt.scatter(coordinates[:, 0], coordinates[:, 1], c=regions, edgecolors="k", s=60)

# ----------------------------------------------------

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
    nn.Linear(18, R),
    nn.Tanh(),
    nn.Linear(R, R),
    nn.Softmax(),
)
