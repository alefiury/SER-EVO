import torch
from torch import nn
import torch.nn.functional as F

def swish(x):
    return x * torch.sigmoid(x)

class FullyConnectedEvo(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FullyConnectedEvo, self).__init__()
        self.tc1 = nn.Linear(input_size, 100)
        # self.tc2 = nn.Linear(1000, 100)
        self.last_layer = nn.Linear(100, num_classes)

        self.dropout1 = nn.Dropout(p=0.5)
        # self.dropout2 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.dropout1(F.relu(self.tc1(x)))
        # x = self.dropout1(F.relu(self.tc1(x)))
        x = self.last_layer(x)

        return x