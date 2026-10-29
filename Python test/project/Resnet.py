import torch.nn as nn
import torchvision.models as models

class ResNet50FineTune(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.resnet50(weights="IMAGENET1K_V1")

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        for param in self.model.fc.parameters():
            param.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),  # 2048
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.model(x)