
import os
from torchvision import datasets, transforms

def get_dataloaders(data_dir, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    return datasets.ImageFolder(data_dir, transform=transform)
