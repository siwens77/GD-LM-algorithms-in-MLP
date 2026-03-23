#!/Library/Frameworks/Python.framework/Versions/3.12/bin/python3

from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import Trainer

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
plt.show()

# Step #2

# Probabilities
probabilities = torch.zeros(R, torch.numel(regions))
for n in range(torch.numel(regions)):
    probabilities[regions[n] - 1][n] = 1
probabilities = probabilities.t()
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
optimizer = torch.optim.LBFGS(MLP.parameters(), lr=0.01)
loss_function = nn.MSELoss()
epochs = 100

fullData = TensorDataset(coordinates, probabilities)
sampleSize = len(coordinates)
trainSize = int(sampleSize*0.8)
validationSize = sampleSize-trainSize

trainData, validationData = random_split(fullData, [trainSize, validationSize])
trainLoader = DataLoader(trainData, batch_size=trainSize, shuffle=True)
validationLoader = DataLoader(validationData, batch_size=validationSize, shuffle=True)

# Training
lossHistory = []
lossHistoryVal = []
for epoch in range(epochs):
    MLP.train()
    batchLoss = 0
    for coordTrain, probTrain in trainLoader:
        def closure():
            optimizer.zero_grad()
            yhat = MLP(coordTrain)
            loss = loss_function(yhat,probTrain)
            loss.backward()
            return loss
        trainLoss = optimizer.step(closure).item()
        batchLoss+=trainLoss
    lossHistory.append(batchLoss/len(trainLoader))
    MLP.eval()
    with torch.no_grad():
        for coordVal, probVal in validationLoader:
            yhat = MLP(coordVal)
            valLoss = loss_function(yhat, probVal).item()
            avgValLoss = valLoss
            lossHistoryVal.append(avgValLoss)
    if avgValLoss< 0.01:
        break
plt.plot(lossHistoryVal)
plt.plot(lossHistory)
plt.axhline(y=0.01, color="red")
plt.scatter(range(len(lossHistory)), lossHistory, c="blue", s=10) 
plt.title("Training Algorithm: LBFGS")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.show()

# Step #3

# Map colors
Colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
# Parameters
M = (1+1/N/2)/k*2*torch.pi
N2 = 300
V = torch.linspace(-M, +M, N2)
[X, Y] = torch.meshgrid(V+x0, V+y0, indexing="xy")
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
plt.show()
