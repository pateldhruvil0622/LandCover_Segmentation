import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Define Paths
IMG_DIR = r"C:\Users\Dhruvil\PycharmProjects\LandCover_Segmentation\Dataset\train_image"
MASK_DIR = r"C:\Users\Dhruvil\PycharmProjects\LandCover_Segmentation\Dataset\train_masks"

# Get list of files
img_files = os.listdir(IMG_DIR)
mask_files = os.listdir(MASK_DIR)

print(f"Found {len(img_files)} images and {len(mask_files)} masks.")

# Pick the first image to test
test_id = img_files[0]
# Assuming mask has same name but .png extension (or _mask.png depending on dataset)
# Adjust this replace logic if your downloaded files are named differently
mask_id = test_id.replace('.jpg', '.png').replace('_sat', '_mask')

img_path = os.path.join(IMG_DIR, test_id)
mask_path = os.path.join(MASK_DIR, mask_id)

if not os.path.exists(mask_path):
    print(f"ERROR: Could not find matching mask for {test_id}")
    print(f"Expected: {mask_path}")
else:
    print(f"Successfully matched: {test_id} -> {mask_id}")

    # Load and Display
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    # Plot side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)
    ax[0].set_title("Satellite Image")
    ax[1].imshow(mask)
    ax[1].set_title("Ground Truth Mask")
    plt.show()