# import torch
# from tqdm import tqdm
# import numpy as np
# import os
# from dataset_loader import get_dataloaders
# from model import EmbeddingModel
# from torchvision import datasets

# def extract_embeddings(output_dir="embeddings", batch_size=32):
#     os.makedirs(output_dir, exist_ok=True)

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = EmbeddingModel().to(device)
#     model.eval()

#     train_loader, val_loader = get_dataloaders(batch_size=batch_size)
#     all_loaders = [("train", train_loader), ("val", val_loader)]

#     # Point to your Caltech-101 training folder
#     train_dir = "data/caltech101/101_ObjectCategories"

#     # Load dataset (same transforms used for embeddings)
#     dataset = datasets.ImageFolder(root=train_dir)

#     # Get all image paths (in order)
#     image_paths = [path for path, _ in dataset.samples]

#     # Save them
#     np.save("embeddings/train_image_paths.npy", np.array(image_paths))

#     with torch.no_grad():
#         for split, loader in all_loaders:
#             embeddings_list, labels_list = [], []
#             for images, labels in tqdm(loader, desc=f"Extracting {split} embeddings"):
#                 images = images.to(device)
#                 embs = model(images)
#                 embeddings_list.append(embs.cpu().numpy())
#                 labels_list.append(labels.numpy())

#             embeddings = np.concatenate(embeddings_list, axis=0)
#             labels = np.concatenate(labels_list, axis=0)

#             np.save(os.path.join(output_dir, f"{split}_embeddings.npy"), embeddings)
#             np.save(os.path.join(output_dir, f"{split}_labels.npy"), labels)
#             print(f"✅ Saved {split} embeddings: {embeddings.shape}")

# if __name__ == "__main__":
#     extract_embeddings()
import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

def extract_embeddings():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === CONFIG ===
    DATA_DIR = "data/caltech101/101_ObjectCategories"
    MODEL_PATH = "model/resnet50_finetuned_20epochs.pth"
    SAVE_DIR = "embeddings_20epochs"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # === LOAD MODEL ===
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 102)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.to(device).eval()

    # === TRANSFORMS ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # === LOAD DATA ===
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    embeddings = []
    image_paths = []

    print("Extracting train embeddings...")
    for imgs, _ in tqdm(loader):
        imgs = imgs.to(device)
        with torch.no_grad():
            emb = model(imgs).squeeze(-1).squeeze(-1).cpu().numpy()
        embeddings.append(emb)

    embeddings = np.concatenate(embeddings)
    image_paths = [path for path, _ in dataset.samples]

    np.save(os.path.join(SAVE_DIR, "train_embeddings.npy"), embeddings)
    np.save(os.path.join(SAVE_DIR, "train_image_paths.npy"), np.array(image_paths))

    print(f"✅ Saved train embeddings: {embeddings.shape}")

if __name__ == "__main__":
    extract_embeddings()