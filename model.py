import torch
import torch.nn as nn
from torchvision.models import convnext_tiny

class MMCModel(nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()

        self.rgb = convnext_tiny(pretrained=True)
        self.dep = convnext_tiny(pretrained=True)
        self.ir  = convnext_tiny(pretrained=True)

        dim = self.rgb.classifier[2].in_features
        self.rgb.classifier = nn.Identity()
        self.dep.classifier = nn.Identity()
        self.ir.classifier  = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(dim * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x1, x2, x3):
        f1 = self.rgb(x1)
        f2 = self.dep(x2)
        f3 = self.ir(x3)
        return self.head(torch.cat([f1, f2, f3], dim=1))
