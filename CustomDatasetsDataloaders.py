import torch
import torchvision
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader

class WineDataset(Dataset):
    def __init__(self):
        #data loading
        xy = np.loadtxt("data/wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]


    def __getitem__(self, index):
        #dataset and indexing _> dataset[0]
        return self.x[index], self.y[index]
    
    def __len__(self):
        #len(dataset)
        return self.n_samples
    
dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

#training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        #forward
        #backard
        #update
        if (i+1) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')

