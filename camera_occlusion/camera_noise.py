import cv2
import numpy as np
import random
import argparse
import imageio.v2 as imageio
from abc import ABC, abstractmethod

# ====================================================================================
#  1. CLASS-BASED AUGMENTATION ENGINE
# ====================================================================================

class Effect(ABC):
    """Abstract Base Class for all augmentation effects."""
    @abstractmethod
    def apply(self, image):
        """Applies the effect to an image."""
        pass

    def __call__(self, image):
        """Allows the class instance to be called as a function."""
        return self.apply(image)

class Rain(Effect):
    """
    Simulates raindrops on a camera lens with distortion, shading, and highlights.
    
    This effect models raindrops as small, distorted circular regions on the image,
    mimicking the refractive effect of water droplets.
    """
    def __init__(self,
                 num_drops: int = 20,
                 radius_range: tuple = (4, 8),
                 magnification: float = 1.1,
                 highlight_thickness: int = 1,
                 shade_thickness: int = 2,
                 highlight_alpha: float = 0.8,
                 shade_alpha: float = 0.1):
        """
        Initializes the Rain effect with detailed parameters.

        Args:
            num_drops (int): The number of raindrops to generate.
            radius_range (tuple): A (min, max) tuple for the radius of each drop.
            magnification (float): The magnification factor for the lens distortion effect (>1.0).
            highlight_thickness (int): The thickness of the top highlight on the drop.
            shade_thickness (int): The thickness of the bottom shadow on the drop.
            highlight_alpha (float): Opacity of the highlight (0.0 to 1.0).
            shade_alpha (float): Opacity of the shadow (0.0 to 1.0).
        """
        self.num_drops = num_drops
        self.radius_range = radius_range
        self.magnification = max(1.01, magnification) # Must be > 1.0
        self.highlight_thickness = highlight_thickness
        self.shade_thickness = shade_thickness
        self.highlight_color = (255, 255, 255, highlight_alpha)
        self.shade_color = (0, 0, 0, shade_alpha)

    def apply(self, image):
        img_out = image.copy()
        h, w, _ = image.shape

        for _ in range(self.num_drops):
            center_x = random.randint(0, w - 1)
            center_y = random.randint(0, h - 1)
            radius = random.randint(*self.radius_range)
            
            x1, y1 = max(0, center_x - radius), max(0, center_y - radius)
            x2, y2 = min(w, center_x + radius), min(h, center_y + radius)

            patch = image[y1:y2, x1:x2]
            if patch.size == 0: continue

            # 1. Distortion
            magnified_patch = cv2.resize(patch, (0, 0), fx=self.magnification, fy=self.magnification, interpolation=cv2.INTER_LINEAR)
            mx, my, _ = magnified_patch.shape
            crop_x = max(0, (my - patch.shape[1]) // 2)
            crop_y = max(0, (mx - patch.shape[0]) // 2)
            distorted_patch = magnified_patch[crop_y:crop_y + patch.shape[0], crop_x:crop_x + patch.shape[1]]

            if distorted_patch.shape != patch.shape: continue

            mask = np.zeros_like(patch, dtype=np.uint8)
            cv2.circle(mask, (radius, radius), radius, (255, 255, 255), -1)

            img_out[y1:y2, x1:x2] = np.where(mask > 0, distorted_patch, img_out[y1:y2, x1:x2])

            # 2. Shading and Highlight
            cv2.ellipse(img_out, (center_x, center_y), (radius, radius), 0, 45, 135, self.shade_color, self.shade_thickness)
            cv2.ellipse(img_out, (center_x, center_y), (radius, radius), 0, 225, 315, self.highlight_color, self.highlight_thickness)
            
        return img_out

class Dust(Effect):
    """
    Simulates dust particles, scratches, and semi-transparent grime splotches.
    """
    def __init__(self,
                 num_specks: int = 5000,
                 speck_color: tuple = (50, 50, 50),
                 num_scratches: int = 5,
                 scratch_length_range: tuple = (-50, 50),
                 scratch_color_range: tuple = (50, 100),
                 num_splotches: int = 5,
                 splotch_size_range_factor: tuple = (0.05, 0.1),
                 splotch_opacity: float = 0.15):
        """
        Initializes the Dust effect with detailed parameters.

        Args:
            num_specks (int): Number of fine dust specks.
            speck_color (tuple): RGB color of the specks.
            num_scratches (int): Number of scratches or fibers.
            scratch_length_range (tuple): Range for random length of scratches.
            scratch_color_range (tuple): (min, max) brightness for scratch color.
            num_splotches (int): Number of larger grime splotches.
            splotch_size_range_factor (tuple): (min, max) factor of image width for splotch size.
            splotch_opacity (float): Opacity of the grime splotches.
        """
        self.num_specks = num_specks
        self.speck_color = speck_color
        self.num_scratches = num_scratches
        self.scratch_length_range = scratch_length_range
        self.scratch_color_range = scratch_color_range
        self.num_splotches = num_splotches
        self.splotch_size_range_factor = splotch_size_range_factor
        self.splotch_opacity = splotch_opacity

    def apply(self, image):
        img_out = image.copy()
        h, w, _ = image.shape

        # 1. Add fine dust specks
        for _ in range(self.num_specks):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            img_out[y, x] = self.speck_color

        # 2. Add scratches/fibers
        for _ in range(self.num_scratches):
            x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
            dx = random.randint(*self.scratch_length_range)
            dy = random.randint(*self.scratch_length_range)
            x2, y2 = x1 + dx, y1 + dy
            color_val = random.randint(*self.scratch_color_range)
            color = (color_val,) * 3
            cv2.line(img_out, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)
            
        # 3. Add larger, semi-transparent splotches (grime)
        splotches = np.zeros((h, w, 1), dtype=np.float32)
        min_splotch_size = int(w * self.splotch_size_range_factor[0])
        max_splotch_size = int(w * self.splotch_size_range_factor[1])

        for _ in range(self.num_splotches):
            x, y = random.randint(0, w - 1), random.randint(0, h - 1)
            size = random.randint(min_splotch_size, max_splotch_size)
            cv2.circle(splotches, (x, y), size, (1,), -1, lineType=cv2.LINE_AA)
        
        kernel_size = int(min(h, w) * 0.2)
        if kernel_size % 2 == 0: kernel_size += 1
        splotches = cv2.GaussianBlur(splotches, (kernel_size, kernel_size), 0)
        
        splotch_max = splotches.max()
        if splotch_max > 0:
            splotches = (splotches / splotch_max) * self.splotch_opacity
        
        # Ensure splotches is broadcastable
        splotches = splotches[..., np.newaxis]

        img_out_float = img_out.astype(np.float32) / 255.0
        # Blend using screen-like logic for darkening
        img_out_float = 1 - (1 - img_out_float) * (1 - splotches)
        img_out = np.clip(img_out_float * 255, 0, 255).astype(np.uint8)

        return img_out

# ====================================================================================
#  2. COMMAND-LINE INTERFACE and EXECUTION
# ====================================================================================

def main():
    """Main function to run augmentations from the command line."""
    parser = argparse.ArgumentParser(description="Apply realistic camera occlusion effects to an image.")
    parser.add_argument("input_image", type=str, help="Path to the input image.")
    parser.add_argument("output_image", type=str, help="Path to save the output image.")
    parser.add_argument("--effect", type=str, choices=['rain', 'dust'], required=True, help="The effect to apply.")
    
    # --- Add specific numerical parameters for CLI control ---
    parser.add_argument("--num-drops", type=int, default=20, help="Number of raindrops (for rain effect).")
    parser.add_argument("--num-specks", type=int, default=5000, help="Number of dust specks (for dust effect).")
    parser.add_argument("--splotch-opacity", type=float, default=0.15, help="Opacity of grime splotches (for dust effect).")
    
    args = parser.parse_args()

    # Load image
    try:
        original_image = imageio.imread(args.input_image)
        # Ensure image is RGB
        if original_image.ndim == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        elif original_image.shape[2] == 4:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
            
    except FileNotFoundError:
        print(f"Error: The file was not found at '{args.input_image}'")
        return

    # Apply the selected effect
    if args.effect == 'rain':
        # Use CLI parameters to initialize the class
        effect = Rain(num_drops=args.num_drops)
        print(f"Applying rain effect with {args.num_drops} drops...")
    elif args.effect == 'dust':
        effect = Dust(num_specks=args.num_specks, splotch_opacity=args.splotch_opacity)
        print(f"Applying dust effect with {args.num_specks} specks and opacity {args.splotch_opacity}...")

    transformed_image = effect.apply(original_image)
    
    # Save the output image
    imageio.imwrite(args.output_image, transformed_image)
    print(f"Successfully saved augmented image to '{args.output_image}'")


if __name__ == "__main__":
    # This block allows the script to be run from the command line
    main()
