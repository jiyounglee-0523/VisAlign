import torch
from torch import nn
import torch.nn.functional as F

class CNNNet(nn.Module):
    def __init__(self,
                 model_name,
                 num_classes
                 ):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=1)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1)
        # self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.25)
        self.dropout4 = nn.Dropout(0.25)
        self.dropout5 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(21632, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = F.relu(self.conv5(x))
        # x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout3(x)

        # x = F.relu(self.conv7(x))
        # x = F.max_pool2d(x, 2)
        # x = self.dropout4(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        output = self.fc2(self.dropout5(x))
        return output