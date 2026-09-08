import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.GELU()

    def forward(self, x):
        return self.act2(x + self.bn2(self.conv2(self.act1(self.bn1(self.conv1(x))))))

class GeneratedModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        base = 32
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.GELU()
        )
        self.block1 = ResBlock(base)
        self.down1 = nn.Sequential(
            nn.Conv2d(base, base * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.GELU()
        )
        self.block2 = ResBlock(base * 2)
        self.down2 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.GELU()
        )
        self.block3 = ResBlock(base * 4)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base * 4, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        x = self.block3(x)
        return self.classifier(self.pool(x))