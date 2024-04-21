import torch
import torch.nn as nn

# 目前是两层fc 也可以加到三层
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(784,256)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(256,64)
        self.layer3 = nn.Linear(64,10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):             #网络传播的结构
        x = x.reshape(-1, 28*28)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)

        return x
