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
plt.show()

# Step #2

# Probabilities
probabilities = torch.zeros(torch.numel(regions), 4)
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
trainingAlgorithm = optim.SGD(MLP.parameters(), lr=0.01, momentum=0.9)
lossFunction = nn.MSELoss()

fullData = TensorDataset(coordinates, probabilities)
sampleSize = len(coordinates)
trainSize = int(0.8 * sampleSize)
testSize = sampleSize - trainSize

trainData, testData = random_split(fullData, [trainSize, testSize])
trainLoader = DataLoader(trainData, batch_size=10, shuffle=True)
testLoader = DataLoader(testData, batch_size=116,shuffle=True)

lossHistory = []
# Training
for epoch in range(epochs):
    sum_loss = 0
    for batchCoord, batchProb in trainLoader:
        trainingAlgorithm.zero_grad()
        y_pred = MLP(batchCoord)
        loss = lossFunction(y_pred, batchProb)
        sum_loss+=loss.item()
        loss.backward()
        trainingAlgorithm.step()
    lossHistory.append(sum_loss)
    inputs, outputs = next(iter(testLoader))
    testY = MLP(inputs)
    testloss = lossFunction(testY, outputs).item()
    if sum_loss < 0.01:
        break

plt.plot(lossHistory)
plt.scatter(range(len(lossHistory)), lossHistory, c="blue", s=10) 
plt.title("Training Algorithm: gradient descent")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.show()
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
plt.show()
