import os
import copy
import time
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# -------------------------------
# CONFIG
# -------------------------------
DATA_DIR = "data/smartvision_dataset/classification"
MODEL_DIR = "models"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 2
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# DATA TRANSFORMS
# -------------------------------
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# -------------------------------
# DATA LOADERS
# -------------------------------
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_test_transforms)
test_dataset  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_test_transforms)

num_classes = len(train_dataset.classes)

print(f"\n📂 Dataset loaded")
print(f"Classes ({num_classes}): {train_dataset.classes}")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}\n")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

dataloaders = {
    "train": train_loader,
    "val": val_loader
}

# -------------------------------
# TRAINING LOOP
# -------------------------------
def train_model(model, criterion, optimizer, scheduler, model_name):
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 40)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase.upper()} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())
                torch.save(best_weights, f"{MODEL_DIR}/{model_name}_best.pth")
                print(f"✅ Saved best {model_name} model")

        if scheduler:
            scheduler.step()

    model.load_state_dict(best_weights)
    print(f"\n🏆 Best Val Accuracy ({model_name}): {best_acc:.4f}")
    return model

# -------------------------------
# MODEL BUILDERS
# -------------------------------
def build_vgg16():
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model

def build_resnet50():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def build_mobilenetv2():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

def build_efficientnet():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

# -------------------------------
# TRAIN ALL MODELS
# -------------------------------
def run_training():
    models_to_train = {
        "vgg16": build_vgg16,
        "resnet50": build_resnet50,
        "mobilenetv2": build_mobilenetv2,
        "efficientnetb0": build_efficientnet,
    }

    for name, builder in models_to_train.items():
        print("\n" + "="*60)
        print(f"🚀 Training {name.upper()}")
        print("="*60)

        model = builder().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        train_model(model, criterion, optimizer, scheduler, name)

    print("\n🎉 ALL MODELS TRAINED AND SAVED SUCCESSFULLY")

# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    run_training()
