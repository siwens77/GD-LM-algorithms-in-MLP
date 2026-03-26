from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch_levenberg_marquardt as tlm
import torch.nn.functional as F
import numpy as np
import matplotlib.colors as mcolors

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
Regions = torch.zeros(N, S) 

# Spirals coordinates and regions
for s in range(1,S+1):
    r = np.floor((s-1)*R/S+1)
    for n in range(1,N+1):
        a0=(n/N)*2*torch.pi/k
        a = torch.tensor(a0+s*2*torch.pi/S)
        b=a0
        X[n-1,s-1] = b*torch.cos(a)+x0
        Y[n-1,s-1] = b*torch.sin(a)+y0
        Regions[n-1,s-1] = r

X_flat = X.flatten()
Y_flat = Y.flatten()
Regions_flat = Regions.flatten().long()
Coordinates = torch.stack([X_flat,Y_flat], dim=1)

# Step #2 

# Probabilities
targets_onehot = F.one_hot(Regions_flat-1, num_classes=4).float()
dataset = TensorDataset(Coordinates, targets_onehot)
sampleSize = len(Coordinates)
trainSize = int(0.8 * sampleSize)
testSize = sampleSize - trainSize

trainData, testData = random_split(dataset, [trainSize, testSize])
trainLoader = DataLoader(trainData, batch_size=464, shuffle=True)
testLoader = DataLoader(testData, batch_size=116,shuffle=True)


# Multi-layer perceptron
class SpiralMLP(nn.Module):
    def __init__(self):
        super(SpiralMLP, self).__init__()
        self.hidden = nn.Linear(2,16)
        self.hidden1 = nn.Linear(16,8)
        self.hidden2 = nn.Linear(8,4)
        self.output = nn.Linear(4,4)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x = self.tanh(self.hidden(x))
        x = self.tanh(self.hidden1(x))
        x = self.tanh(self.hidden2(x))
        x = self.softmax(self.output(x))
        return x

model = SpiralMLP()

lm_module = tlm.training.LevenbergMarquardtModule(
        model = model,
        loss_fn = tlm.loss.MSELoss(),
        learning_rate = 1.0,
        attempts_per_step = 10,
        solve_method = 'qr',
        )

# Training
tlm.utils.fit(lm_module, trainLoader,epochs=200)

# Step #3

# Parameters
M = (1 + 1 / N / 2) / k * 2 *torch.pi
N2 = 300
V2 = torch.linspace(-M, M, N2)

X2, Y2 = torch.meshgrid(V2 + x0, V2 + y0, indexing='ij')
features2 = torch.stack([X2.flatten(), Y2.flatten()], dim=1)

# Most probable regions
model.eval()
with torch.no_grad():
    probabilities2 = model(features2)
    Regions2_flat = torch.argmax(probabilities2, dim=1) + 1

# Coloration
Regions2_grid = Regions2_flat.reshape(N2, N2).numpy()
plt.figure(facecolor='w', figsize=(7, 7))
plt.axis('off')
cmap = mcolors.ListedColormap(['m', 'g', 'c', 'y'])
plt.pcolormesh(X2.numpy(), Y2.numpy(), Regions2_grid, cmap=cmap, alpha=0.25, shading='auto')

colors = ['m', 'g', 'c', 'y']
for r in range(1, R + 1):
    mask = (Regions_flat.numpy()== r)
    plt.plot(X_flat.numpy()[mask], Y_flat.numpy()[mask], 'o',
             markeredgecolor='k',
             markerfacecolor=colors[r-1],
             markersize=6)

plt.axis('equal')
plt.show()
