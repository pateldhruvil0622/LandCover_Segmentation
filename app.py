import os
import torch
import numpy as np
import cv2
from flask import Flask, request, render_template, url_for
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# LOAD MODEL
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Note: In production, load state_dict, but for simplicity we load full object here
# Ensure you run train.py first to generate the .pth file
# --- PASTE THIS EXACTLY ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'land_cover_model.pth')

print(f"🔍 Looking for model at: {MODEL_PATH}")

if os.path.exists(MODEL_PATH):
    try:
        model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
else:
    print("❌ ERROR: File not found at the calculated path.")
    model = None
# ---------------------------

CLASSES = ['Urban', 'Agriculture', 'Rangeland', 'Forest', 'Water', 'Barren', 'Unknown']
# Colors for visualization (R, G, B)
COLORS = np.array([
    [0, 255, 255],  # Urban (Cyan)
    [255, 255, 0],  # Agri (Yellow)
    [255, 0, 255],  # Range (Purple)
    [0, 255, 0],  # Forest (Green)
    [0, 0, 255],  # Water (Blue)
    [255, 255, 255],  # Barren (White)
    [0, 0, 0]  # Unknown (Black)
])


def process_image(image_path):
    # 1. Preprocess
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_size = img.shape[:2]

    # Resize for model (needs to be divisible by 32 for EfficientNet)
    input_img = cv2.resize(img, (256, 256))
    input_tensor = input_img.transpose(2, 0, 1).astype('float32') / 255.0
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(DEVICE)

    # 2. Predict
    with torch.no_grad():
        prediction = model(input_tensor)  # Shape: [1, 7, 256, 256]
        mask = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()  # Shape: [256, 256]

    # 3. Analyze Distribution
    total_pixels = mask.size
    stats = {}
    for idx, class_name in enumerate(CLASSES):
        count = np.sum(mask == idx)
        if count > 0:
            stats[class_name] = round((count / total_pixels) * 100, 2)

    # 4. Colorize Mask
    # Map class indices to RGB colors
    seg_img = COLORS[mask]
    seg_img = cv2.resize(seg_img.astype('uint8'), (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)

    # Save Result
    result_filename = 'seg_' + os.path.basename(image_path)
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))

    return result_filename, stats


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            seg_image, stats = process_image(filepath)

            return render_template('index.html',
                                   original=file.filename,
                                   segmentation=seg_image,
                                   stats=stats)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)