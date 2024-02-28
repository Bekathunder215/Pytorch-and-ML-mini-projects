import torch.nn as nn
import torch.nn.functional as F

#imgsize 4, 3, 32, 32

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,3)  #out -> 30*30*6
        self.pool = nn.MaxPool2d(2,2)  #out -> 15*15*6
        self.conv2 = nn.Conv2d(6,12,5,2) #out -> 6*6*12  after pool 3*3*12
        self.fc1 = nn.Linear(3*3*12, 224)
        self.fc2 = nn.Linear(224, 112)
        self.fc3 = nn.Linear(112, 56)
        self.fc4 = nn.Linear(56, 10)
        
    def forward(self, x):

        # for testing the shape and analysing every input and output of the layers
        # x = self.conv1(x)
        # print(x.shape)
        # x = self.pool(x)
        # print(x.shape)
        # x = self.conv2(x)
        # print(x.shape)
        # x = self.pool(x)
        # print(x.shape)

        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(-1, 3*3*12)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x
    

