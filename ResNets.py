import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=(3,3),
                            stride=stride,
                            padding=1
                        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3,3),
                               stride=stride,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x # the residual stream
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn1(self.conv2(x))
        x += self.shortcut(residual)
        x = F.relu(x)
        return x
    

class ResNet(nn.Module):

    def __init__(self, in_channels, num_blocks, num_classes):
        super().__init__()

        self.in_channels = 16

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Make the network structure.
        self.layer1 = self.make_layer(16, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(32, num_blocks[1], stride=1)
        self.layer3 = self.make_layer(128, num_blocks[2], stride=1)

        self.avg_pool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(2048, num_classes)

    def make_layer(self, out_channels, num_block, stride):
        strides = [stride for _ in range(num_block)]
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # print("Input: ", x.shape) # torch.Size([1, 3, 32, 32])
        x = self.bn1(self.conv1(x))
        x = F.relu(x)

        # print("Before L1: ", x.shape) # torch.Size([1, 16, 32, 32])
        x = self.layer1(x)
        
        # print("Before L2: ", x.shape) # torch.Size([1, 32, 32, 32])
        x = self.layer2(x)

        # print("Before L3: ", x.shape) # torch.Size([1, 64, 32, 32])
        x = self.layer3(x)

        # print("Output: ", x.shape) # torch.Size([1, 128, 32, 32])
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1 )

        return self.fc(x)