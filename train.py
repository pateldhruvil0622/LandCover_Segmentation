import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import albumentations as albu  # Library for Data Augmentation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# --- CONFIGURATION ---
ENCODER = 'efficientnet-b3'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['urban_land', 'agriculture_land', 'rangeland', 'forest_land', 'water', 'barren_land', 'unknown']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4
LEARNING_RATE = 0.0001
EPOCHS = 15  # Recommended: 15 epochs for mixed data


# --- AUGMENTATION FUNCTION ---
# This makes your 10 Google Maps images look like 100 different images

def get_training_augmentation():
    train_transform = [
        # 1. GEOMETRY (Flip/Rotate)
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),

        # 2. COLOR JITTER (Simulate different lighting)
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),

        # 3. TEXTURE BOOSTERS (The Fix for Canals vs Forests)
        # CLAHE: Makes texture pop out (trees look bumpier, water looks flatter)
        albu.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),

        # Sharpen: Highlights the straight edges of canals
        albu.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),

        # Blur: Simulates low-res water
        albu.GaussianBlur(blur_limit=(3, 7), p=0.2),
    ]
    return albu.Compose(train_transform)


# --- DATASET CLASS ---
class LandCoverDataset(Dataset):
    def __init__(self, images_dir, masks_dir, classes, augmentation=None):
        # Find all satellite images
        all_files = os.listdir(images_dir)
        self.ids = [f for f in all_files if f.endswith('_sat.jpg')]
        self.ids.sort()

        # Create full file paths
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id.replace('_sat.jpg', '_mask.png')) for image_id in self.ids]

        # Convert class names to RGB values
        self.class_values = [self.get_class_rgb(cls) for cls in classes]
        self.augmentation = augmentation

    def get_class_rgb(self, name):
        mapping = {
            'urban_land': (0, 255, 255),  # Cyan
            'agriculture_land': (255, 255, 0),  # Yellow
            'rangeland': (255, 0, 255),  # Purple
            'forest_land': (0, 255, 0),  # Green
            'water': (0, 0, 255),  # Blue
            'barren_land': (255, 255, 255),  # White
            'unknown': (0, 0, 0)  # Black
        }
        return mapping.get(name, (0, 0, 0))

    def __getitem__(self, i):
        # 1. Read Image
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Read Mask
        mask = cv2.imread(self.masks_fps[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # 3. Resize (Critical to prevent Out Of Memory errors)
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # 4. Apply Augmentation
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # 5. Preprocessing for Model
        # Create One-Hot encoded masks (Layer per class)
        masks = [(mask == v).all(axis=-1) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float32')

        # Normalize Image and Transpose to (Channels, Height, Width)
        image = image.transpose(2, 0, 1).astype('float32') / 255.0
        mask = mask.transpose(2, 0, 1)

        return torch.tensor(image), torch.tensor(mask)

    def __len__(self):
        return len(self.ids)


# --- NEW EVALUATION FUNCTION ---
def evaluate_model(model, loader):
    print("\n📊 Starting Final Evaluation...")
    model.eval()
    y_true_all = []
    y_pred_all = []

    # Check first 50 batches to save time
    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            images = images.to(DEVICE)

            # Predict
            output = model(images)
            pred_indices = torch.argmax(output, dim=1).cpu().numpy()

            # Ground Truth
            true_indices = torch.argmax(masks, dim=1).cpu().numpy()

            # Filter 'Unknown' class (Index 6)
            valid_pixels = true_indices != 6
            y_true_all.extend(true_indices[valid_pixels].flatten())
            y_pred_all.extend(pred_indices[valid_pixels].flatten())

            if i > 50: break

    print("✅ Calculating Metrics...")

    # 1. Text Report
    print(classification_report(y_true_all, y_pred_all, target_names=CLASSES[:-1]))

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y_true_all, y_pred_all)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', xticklabels=CLASSES[:-1], yticklabels=CLASSES[:-1])
    plt.title('Final Confusion Matrix (%)')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.show()


# --- TRAINING LOOP ---
if __name__ == '__main__':
    # 1. Setup Directories
    # This points to your PyCharm project folder
    BASE_DIR = r"C:\Users\Dhruvil\PycharmProjects\LandCover_Segmentation\Dataset"

    x_train_dir = os.path.join(BASE_DIR, 'train_image')
    y_train_dir = os.path.join(BASE_DIR, 'train_masks')

    print(f"--- CONFIGURATION ---")
    print(f"Images: {x_train_dir}")
    print(f"Masks:  {y_train_dir}")
    print(f"Device: {DEVICE}")
    print("-" * 30)

    # 2. Initialize Dataset & Dataloader
    dataset = LandCoverDataset(
        x_train_dir,
        y_train_dir,
        CLASSES,
        augmentation=get_training_augmentation()  # Activate Augmentation
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # 3. Create Model
    print("Creating U-Net++ model...")
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=len(CLASSES),
        activation=None
    )
    model.to(DEVICE)

    # 4. Optimizer & Loss
    loss_fn = smp.losses.DiceLoss(mode='multilabel', from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Execution
    print("Starting Training...")
    os.makedirs('./models', exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for images, masks in loop:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            # Forward Pass
            outputs = model(images)

            # Loss Calculation
            loss = loss_fn(outputs, masks)

            # Backward Pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Save model after every epoch
        torch.save(model, './models/land_cover_model.pth')
        print(f"Epoch {epoch + 1} Complete. Model Saved.")