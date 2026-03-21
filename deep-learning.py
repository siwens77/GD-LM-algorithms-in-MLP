import numpy as np
import matplotlib.pyplot as plt

# Step #1

# Parameter
N = 24
# Coordinates
V = np.linspace(-2, +2, N)
[X, Y] = np.meshgrid(V, V)
coordinates = np.vstack([X.ravel(), Y.ravel()]).T
print(coordinates)
# Regions
cond1 = Y >= X
cond2 = np.logical_or((Y > -X), np.abs(X + Y) < 1e-2)
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
# plt.show()

# Step #2

# Probabilities
probabilities = [[0] * 4 for i in range(len(regions))]
for n in range(len(regions)):
    probabilities[n][regions[n] - 1] = 1
# Multi-layer perceptron
MLP = None

