import sys
import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from camera_occlusion.camera_noise import Rain, Dust, Glare


# -------- Helpers --------
def load_rgb(path):
    """Load image and ensure RGB."""
    try:
        img = imageio.imread(path)
    except FileNotFoundError:
        print(f"Error: The image file was not found at '{path}'")
        sys.exit(1)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


# -------- Config --------
IMAGE_PATH = "../GTSRB_dataset/GTSRB/Final_Training/Images/00001/00000_00000.ppm"

effect_presets = {
    # Row 1: Rain Progression
    "Light Rain": Rain(num_drops=15, radius_range=(2, 4), magnification=1.03),
    "Moderate Rain": Rain(num_drops=40, radius_range=(3, 6), magnification=1.08),
    "Heavy Rain": Rain(num_drops=100, radius_range=(4, 8), magnification=1.15),

    # Row 2: Dust Progression
    "Light Dust": Dust(num_specks=100, num_scratches=0, splotch_opacity=0.06),
    "Moderate Dust": Dust(num_specks=1000, num_scratches=2, splotch_opacity=0.15),
    "Heavy Dust": Dust(num_specks=5000, num_scratches=3, splotch_opacity=0.25),

    # Row 3: Glare Adversary Progression
    "Light Glare": Glare(
        source_threshold=0.9,
        intensity=0.5,
        num_streaks=6,
        ghost_count=2,
        chromatic_aberration=1.0
    ),
    "Moderate Glare": Glare(
        source_threshold=0.85,
        intensity=0.85,
        num_streaks=8,
        ghost_count=4,
        chromatic_aberration=1.5
    ),
    "Heavy Glare": Glare(
        source_threshold=0.75,         # Stays low to be more sensitive
        intensity=0.95,                # Reduced below 1.0 to prevent blowout
        num_streaks=8,
        streak_length_factor=2.5,      # Increased to make streaks longer
        ghost_count=6,                 # Stays high for complexity
        chromatic_aberration=1.8       # Reduced for more natural color fringing
    ),
}


# -------- Run --------
def main():
    img = load_rgb(IMAGE_PATH)

    # Force a 3x3 grid
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()

    for i, (title, preset) in enumerate(effect_presets.items()):
        if i >= len(axes): break # Stop if we have more presets than axes
        
        ax = axes[i]
        augmented = preset(img)
        ax.imshow(augmented)
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    fig.suptitle("Camera Occlusion Gallery", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('examples/example_noise.png', dpi=125)
    plt.clf()


if __name__ == "__main__":
    main()
