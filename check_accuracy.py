import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# --- CONFIGURATION ---
# The folder where your images actually are (based on your previous errors)
REAL_IMAGE_FOLDER = r"C:\Users\Dhruvil\PycharmProjects\LandCover_Segmentation\Dataset\archive\train"
MODEL_PATH = r"C:\Users\Dhruvil\PycharmProjects\LandCover_Segmentation\models\land_cover_model.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- AUTOMATIC IMAGE SELECTION ---
if not os.path.exists(REAL_IMAGE_FOLDER):
    print(f"❌ Error: Could not find folder: {REAL_IMAGE_FOLDER}")
    exit()

# Find all satellite images
all_files = os.listdir(REAL_IMAGE_FOLDER)
sat_images = [f for f in all_files if f.endswith('_sat.jpg')]

if not sat_images:
    print("❌ Error: No satellite images found in that folder!")
    exit()

# Pick a random one automatically
selected_file = random.choice(sat_images)
IMAGE_ID = selected_file.replace('_sat.jpg', '')

print(f"✅ Found {len(sat_images)} images.")
print(f"🔍 Testing on random image ID: {IMAGE_ID}")

# Define Full Paths
IMG_PATH = os.path.join(REAL_IMAGE_FOLDER, f'{IMAGE_ID}_sat.jpg')
MASK_PATH = os.path.join(REAL_IMAGE_FOLDER, f'{IMAGE_ID}_mask.png')

# --- LOAD MODEL ---
print("Loading model...")
try:
    model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.eval()
except Exception as e:
    print(f"❌ Model Error: {e}")
    exit()

# --- PREPROCESS IMAGE ---
image = cv2.imread(IMG_PATH)
if image is None:
    print("❌ Error: CV2 could not read the image. Check permissions.")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
orig_image = image.copy()

# Resize for Model
image = cv2.resize(image, (256, 256))
input_tensor = image.transpose(2, 0, 1).astype('float32') / 255.0
input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(DEVICE)

# --- PREDICT ---
with torch.no_grad():
    output = model(input_tensor)
    pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

# --- PREPROCESS GROUND TRUTH ---
gt_mask = cv2.imread(MASK_PATH)
if gt_mask is None:
    print("❌ Error: Found the image but could not find the matching mask!")
    exit()

gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2RGB)
gt_mask_resized = cv2.resize(gt_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

# Helper: Convert RGB Mask to Class IDs
def rgb_to_2d_label(label):
    label_seg = np.zeros(label.shape[:2], dtype=np.uint8)
    label_seg[np.all(label == (0, 255, 255), axis=-1)] = 0  # Urban
    label_seg[np.all(label == (255, 255, 0), axis=-1)] = 1  # Agri
    label_seg[np.all(label == (255, 0, 255), axis=-1)] = 2  # Range
    label_seg[np.all(label == (0, 255, 0), axis=-1)] = 3    # Forest
    label_seg[np.all(label == (0, 0, 255), axis=-1)] = 4    # Water
    label_seg[np.all(label == (255, 255, 255), axis=-1)] = 5 # Barren
    return label_seg

gt_mask_indices = rgb_to_2d_label(gt_mask_resized)

# --- CALCULATE ACCURACY ---
correct_pixels = np.sum(pred_mask == gt_mask_indices)
total_pixels = pred_mask.size
accuracy = (correct_pixels / total_pixels) * 100

print(f"\n🎯 Accuracy Result: {accuracy:.2f}%")

# --- PLOT ---
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(orig_image)
ax[0].set_title(f"ID: {IMAGE_ID}")
ax[1].imshow(gt_mask)
ax[1].set_title("Ground Truth Mask")
ax[2].imshow(pred_mask, cmap='jet')
ax[2].set_title(f"AI Prediction ({accuracy:.1f}%)")
plt.show()