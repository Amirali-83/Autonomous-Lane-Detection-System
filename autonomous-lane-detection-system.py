#%%
!pip install segmentation-models-pytorch==0.3.3 albumentations==1.3.1 --quiet
!pip install opencv-python matplotlib --quiet
#%%
import os
import glob
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

#%%
DATA_DIR = "/kaggle/input/tusimple/TUSimple"

train_folder = os.path.join(DATA_DIR, "train_set")
test_folder  = os.path.join(DATA_DIR, "test_set")

# Combine all training JSON label files
label_files = glob.glob(os.path.join(train_folder, "label_data_*.json"))
labels = []
for file in label_files:
    with open(file, 'r') as f:
        labels.extend([json.loads(line) for line in f.readlines()])

print(f"Total labeled samples: {len(labels)}")

#%%
def create_mask(sample, base_path):
    img_path = os.path.join(base_path, sample['raw_file'])
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    
    mask = np.zeros((h, w), dtype=np.uint8)
    for lane in sample['lanes']:
        for x, y in zip(lane, sample['h_samples']):
            if x > 0:
                cv2.circle(mask, (x, y), 5, 255, -1)
    return mask

#%%
class LaneDataset(Dataset):
    def __init__(self, labels, base_path, transform=None):
        self.labels = labels
        self.base_path = base_path
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = self.labels[idx]
        img_path = os.path.join(self.base_path, sample['raw_file'])
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = create_mask(sample, self.base_path)
        
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        mask = (mask > 0).float().unsqueeze(0)
        return img, mask

transform = A.Compose([
    A.Resize(256, 512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(),
    ToTensorV2()
])

train_dataset = LaneDataset(labels, train_folder, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

print(f"Total batches: {len(train_loader)}")

#%%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(DEVICE)

loss_fn = smp.losses.DiceLoss(mode='binary')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print(f"Model ready on {DEVICE}")

#%%
def train_model(model, loader, optimizer, loss_fn, epochs=30):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for imgs, masks in tqdm(loader, leave=False):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(loader):.4f}")

train_model(model, train_loader, optimizer, loss_fn, epochs=30)

#%%
torch.save(model.state_dict(), "unet_tusimple_30epochs.pth")
print("Model saved as unet_tusimple_30epochs.pth")

#%%
model.eval()
imgs, masks = next(iter(train_loader))
imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

with torch.no_grad():
    preds = model(imgs)
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

plt.figure(figsize=(12, 8))
for i in range(3):
    plt.subplot(3, 3, i*3 + 1)
    plt.imshow(imgs[i].permute(1,2,0).cpu())
    plt.title("Image")
    plt.axis("off")
    
    plt.subplot(3, 3, i*3 + 2)
    plt.imshow(masks[i].squeeze().cpu(), cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")
    
    plt.subplot(3, 3, i*3 + 3)
    plt.imshow(preds[i].squeeze().cpu(), cmap="gray")
    plt.title("Prediction")
    plt.axis("off")

plt.show()

#%%
from sklearn.metrics import jaccard_score

model.eval()
iou_scores = []
with torch.no_grad():
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model(imgs)
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()
        
        for p, m in zip(preds, masks):
            p_flat = p.cpu().numpy().flatten()
            m_flat = m.cpu().numpy().flatten()
            score = jaccard_score(m_flat, p_flat)
            iou_scores.append(score)

print(f"Mean IoU: {np.mean(iou_scores):.4f}")

#%%

#%%
# Main dataset folder
DATA_DIR = "/kaggle/input/tusimple/TUSimple"

# Test set folder (images)
test_folder = os.path.join(DATA_DIR, "test_set")

# Path to test_label.json
test_labels_path = os.path.join(DATA_DIR, "test_label.json")

# Load test labels
import json
with open(test_labels_path, 'r') as f:
    test_labels = [json.loads(line) for line in f.readlines()]

print(f"Total test samples: {len(test_labels)}")

#%%
test_dataset = LaneDataset(test_labels, test_folder, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

#%%
model.eval()
imgs, masks = next(iter(test_loader))
imgs = imgs.to(DEVICE)

with torch.no_grad():
    preds = model(imgs)
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

# Show predictions
plt.figure(figsize=(12, 8))
for i in range(3):  # show first 3 examples
    plt.subplot(3, 3, i*3 + 1)
    plt.imshow(imgs[i].permute(1,2,0).cpu())
    plt.title("Test Image")
    plt.axis("off")
    
    plt.subplot(3, 3, i*3 + 2)
    plt.imshow(masks[i].squeeze().cpu(), cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")
    
    plt.subplot(3, 3, i*3 + 3)
    plt.imshow(preds[i].squeeze().cpu(), cmap="gray")
    plt.title("Prediction")
    plt.axis("off")

plt.show()

#%%
!jupyter nbconvert --to script "Autonomous Lane Detection System.ipynb"
