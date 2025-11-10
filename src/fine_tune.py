# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import models
# from dataset_loader import get_dataloaders

# def main():
#     # ----------------------------
#     # CONFIG
#     # ----------------------------
#     batch_size = 32
#     epochs = 10
#     lr = 1e-4
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # ----------------------------
#     # LOAD DATA
#     # ----------------------------
#     train_loader, val_loader, num_classes = get_dataloaders(batch_size=batch_size)

#     # ----------------------------
#     # LOAD MODEL
#     # ----------------------------
#     model = models.resnet50(pretrained=True)
#     for param in model.parameters():
#         param.requires_grad = False  # Freeze base layers

#     # Replace classifier (fully connected layer)
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     model = model.to(device)

#     # ----------------------------
#     # TRAINING SETUP
#     # ----------------------------
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.fc.parameters(), lr=lr)

#     # ----------------------------
#     # TRAINING LOOP
#     # ----------------------------
#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#         correct, total = 0, 0

#         for imgs, labels in train_loader:
#             imgs, labels = imgs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(imgs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#         train_acc = 100 * correct / total
#         print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")

#     # ----------------------------
#     # VALIDATION
#     # ----------------------------
#     model.eval()
#     correct, total = 0, 0
#     with torch.no_grad():
#         for imgs, labels in val_loader:
#             imgs, labels = imgs.to(device), labels.to(device)
#             outputs = model(imgs)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     val_acc = 100 * correct / total
#     print(f"\n✅ Validation Accuracy: {val_acc:.2f}%")

#     # ----------------------------
#     # SAVE MODEL
#     # ----------------------------
#     torch.save(model.state_dict(), "model/resnet50_finetuned.pth")
#     print("💾 Fine-tuned model saved as resnet50_finetuned.pth")

# if __name__ == "__main__":
#     main()

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataset_loader import get_dataloaders
from tqdm import tqdm
import numpy as np
import os

MODEL_SAVE_PATH = "model/resnet50_finetuned_20epochs.pth"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # -------------------------
    # 1️⃣ Load Datasets
    # -------------------------
    batch_size = 32
    train_loader, val_loader, num_classes = get_dataloaders(batch_size=batch_size)
    print(f"✅ Loaded Caltech-101: {len(train_loader.dataset)+len(val_loader.dataset)} images")

    # -------------------------
    # 2️⃣ Load Pretrained Model
    # -------------------------
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # -------------------------
    # 3️⃣ Training Setup
    # -------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 20
    best_val_acc = 0.0
    patience = 5  # stop if val acc doesn’t improve for 5 epochs
    wait = 0

    # -------------------------
    # 4️⃣ Training Loop
    # -------------------------
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total

        # -------------------------
        # 5️⃣ Validation
        # -------------------------
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"🧠 Epoch [{epoch+1}/{num_epochs}] - Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # -------------------------
        # 6️⃣ Early Stopping
        # -------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Model saved (val acc improved to {val_acc:.4f})")
            wait = 0
        else:
            wait += 1
            print(f"⚠️ No improvement for {wait} epochs.")
            if wait >= patience:
                print("⏹️ Early stopping triggered.")
                break

    print(f"🎯 Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"✅ Model saved at: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
