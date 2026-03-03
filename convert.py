import os
import cv2
import numpy as np

# --- 1. MAIN ARCHIVE PATH (The folder holding Tile 1, Tile 2...) ---
# Make sure this points to the folder that CONTAINS the "Tile X" folders
ARCHIVE_PATH = r"C:\Users\Dhruvil\Downloads\archive\Semantic segmentation dataset"

# --- 2. DESTINATION ---
PROJECT_DIR = r"C:\Users\Dhruvil\PycharmProjects\LandCover_Segmentation\dataset"
DEST_IMAGES = os.path.join(PROJECT_DIR, "train_image")
DEST_MASKS = os.path.join(PROJECT_DIR, "train_masks")

COLOR_MAP = {
    (152, 16, 60): (0, 0, 255),  # Water -> Blue
    (246, 41, 132): (0, 255, 255),  # Building -> Cyan
    (228, 193, 110): (0, 255, 255),  # Road -> Cyan
    (58, 221, 254): (0, 255, 0),  # Vegetation -> Green
    (155, 155, 155): (255, 255, 255)  # Land -> White
}


def import_all_tiles_safely():
    # Force create output folders
    if not os.path.exists(DEST_IMAGES): os.makedirs(DEST_IMAGES)
    if not os.path.exists(DEST_MASKS): os.makedirs(DEST_MASKS)

    # Find all folders inside the archive
    tile_folders = [f for f in os.listdir(ARCHIVE_PATH) if os.path.isdir(os.path.join(ARCHIVE_PATH, f))]

    grand_total = 0

    for tile_name in tile_folders:
        print(f"\n📦 Processing folder: {tile_name}...")

        src_img_dir = os.path.join(ARCHIVE_PATH, tile_name, "images")
        src_mask_dir = os.path.join(ARCHIVE_PATH, tile_name, "masks")

        if not os.path.exists(src_img_dir):
            print(f"   ⚠️ Skipping {tile_name} (No 'images' folder found)")
            continue

        # Get files
        images = sorted([f for f in os.listdir(src_img_dir) if f.lower().endswith(('.jpg', '.png'))])
        masks = sorted([f for f in os.listdir(src_mask_dir) if f.lower().endswith(('.jpg', '.png'))])

        count = 0
        limit = min(len(images), len(masks))

        for i in range(limit):
            # Read
            img = cv2.imread(os.path.join(src_img_dir, images[i]))
            mask = cv2.imread(os.path.join(src_mask_dir, masks[i]))

            if img is None or mask is None: continue

            # Resize
            img = cv2.resize(img, (512, 512))
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

            # Convert Colors
            new_mask = np.zeros_like(mask)
            for dubai_color, my_color in COLOR_MAP.items():
                diff = np.abs(mask.astype(int) - np.array(dubai_color))
                mask_diff = np.sum(diff, axis=2)
                new_mask[mask_diff < 40] = my_color

            # --- THE FIX IS HERE ---
            # We include the {tile_name} in the filename so they don't overwrite!
            # Example: dubai_Tile 1_005_sat.jpg
            save_id = f"dubai_{tile_name}_{i}"

            cv2.imwrite(os.path.join(DEST_IMAGES, f"{save_id}_sat.jpg"), img)
            cv2.imwrite(os.path.join(DEST_MASKS, f"{save_id}_mask.png"), new_mask)

            count += 1
            grand_total += 1
            if count % 10 == 0: print(f"   -> Saved {count}...", end='\r')

        print(f"   ✅ Finished {tile_name}: Added {count} images.")

    print(f"\n🎉 GRAND TOTAL: You now have {grand_total} images ready for training!")


if __name__ == "__main__":
    import_all_tiles_safely()