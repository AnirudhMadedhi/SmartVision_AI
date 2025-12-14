
import torch.nn as nn
from torchvision import models

def build_vgg16(num_classes):
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model
