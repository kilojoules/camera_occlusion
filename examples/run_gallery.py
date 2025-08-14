import matplotlib.pyplot as plt
from camera_occlusion.camera_noise import Rain, Dust
import imageio.v2 as imageio
import cv2
import sys

# 1. SETUP & IMAGE LOADING
# -------------------------
IMAGE_PATH = 'GTSRB/Final_Training/Images/00001/00000_00000.ppm'

try:
    original_image = imageio.imread(IMAGE_PATH)
    if original_image.ndim == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    elif original_image.shape[2] == 4:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
except FileNotFoundError:
    print(f"Error: The image file was not found at '{IMAGE_PATH}'")
    sys.exit(1)

# 2. DEFINE EFFECT LEVELS
# -----------------------
effect_presets = {
    "Light Rain": Rain(num_drops=15, radius_range=(2, 4), magnification=1.03),
    "Moderate Rain": Rain(num_drops=40, radius_range=(3, 6), magnification=1.08),
    "Heavy Rain": Rain(num_drops=100, radius_range=(4, 8), magnification=1.15),
    "Light Dust": Dust(num_specks=100, num_scratches=0, splotch_opacity=0.06),
    "Moderate Dust": Dust(num_specks=1000, num_scratches=2, splotch_opacity=0.15),
    "Heavy Dust": Dust(num_specks=5000, num_scratches=3, splotch_opacity=0.25)
}

# 3. GENERATE AND PLOT
# --------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Parameterized Camera Occlusion Gallery (Corrected Import)', fontsize=18)

preset_list = list(effect_presets.items())

for i, ax in enumerate(axes.flat):
    if i < len(preset_list):
        title, effect_instance = preset_list[i]
        print(f"Applying effect: {title}...")
        augmented_image = effect_instance(original_image)
        ax.imshow(augmented_image)
        ax.set_title(title, fontsize=14)
    ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

