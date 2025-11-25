import torch
from torch import nn


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 10)
    
    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = self.fc1(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        ### To do 4
        self.conv1=nn.Conv2d(3, 32, 5, 1,2)
        self.conv2=nn.Conv2d(32,64,5,1,2)
        self.conv3=nn.Conv2d(64,128,5,1,2)
        self.activation=nn.ReLU()
        self.pool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(2048,512)
        self.fc2=nn.Linear(512,10)
    def forward(self, x):
        ### To do 4
        x=self.activation(self.conv1(x))
        x=self.pool(x)
        x=self.activation(self.conv2(x))
        x=self.pool(x)
        x=self.activation(self.conv3(x))
        x=self.pool(x)
        x=x.view(x.size(0),-1)
        x=self.activation(self.fc1(x))
        x=self.fc2(x)
        return x

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        ## To do 7
        # self.resnet.fc = 

    def forward(self, x):
        return self.resnet(x)



