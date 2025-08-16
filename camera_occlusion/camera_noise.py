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

# add to camera_occlusion/camera_noise.py
class Glare(Effect):
    """
    Simulates camera glare with procedural, physically-inspired effects.

    This effect models several phenomena:
      - Starburst: Diffraction spikes caused by the camera's aperture blades.
      - Haze/Bloom: A soft glow around the light source.
      - Ghosts: Faint, colored copies of the aperture shape reflected inside the lens.
      - Chromatic Aberration: Color fringing on the glare artifacts.
    """
    def __init__(self,
                 source_threshold: float = 0.9,
                 intensity: float = 0.75,
                 num_streaks: int = 6,
                 streak_angle_offset: float = 0.0,
                 streak_length_factor: float = 2.0,
                 streak_thin_factor: float = 2.5,
                 ghost_count: int = 4,
                 ghost_spacing: float = 0.4,
                 ghost_size_decay: float = 0.85,
                 chromatic_aberration: float = 1.5,
                 manual_source: tuple | None = None
                 ):
        """
        Initializes the Glare effect.

        Args:
            source_threshold (float): Luminance value (0-1) to consider a pixel a light source.
            intensity (float): Overall strength of the glare effect.
            num_streaks (int): Number of primary streaks in the starburst (e.g., 6 for a 6-blade aperture).
            streak_angle_offset (float): Rotation angle (degrees) for the entire starburst pattern.
            streak_length_factor (float): How long the streaks are relative to the image diagonal.
            streak_thin_factor (float): Controls the thinness of the streaks.
            ghost_count (int): Number of lens ghosts to generate.
            ghost_spacing (float): Spacing of ghosts along the source-center axis.
            ghost_size_decay (float): How much smaller each successive ghost becomes.
            chromatic_aberration (float): Amount of color fringing on artifacts.
            manual_source (tuple, optional): A (x, y) tuple to manually set the light source.
        """
        self.source_threshold = source_threshold
        self.intensity = intensity
        self.num_streaks = num_streaks
        self.streak_angle_offset = streak_angle_offset
        self.streak_length_factor = streak_length_factor
        self.streak_thin_factor = streak_thin_factor
        self.ghost_count = ghost_count
        self.ghost_spacing = ghost_spacing
        self.ghost_size_decay = ghost_size_decay
        self.chromatic_aberration = chromatic_aberration
        self.manual_source = manual_source

    @staticmethod
    def _to_rgb_float(img):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img.astype(np.float32) / 255.0

    def _luminance(self, img):
        return 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0] # BGR order

    def _estimate_source(self, bright_mask):
        y, x = np.nonzero(bright_mask > 0.0)
        if len(x) == 0:
            return None
        weights = bright_mask[y, x]
        center_x = int(np.average(x, weights=weights))
        center_y = int(np.average(y, weights=weights))
        return (center_x, center_y)

    def _screen_blend(self, base, layer):
        return 1.0 - (1.0 - base) * (1.0 - layer)

    def _generate_ghosts(self, shape, source, center):
        h, w = shape
        ghosts_layer = np.zeros((h, w, 3), dtype=np.float32)
        if self.ghost_count == 0:
            return ghosts_layer

        vec_x, vec_y = center[0] - source[0], center[1] - source[1]
        base_size = int(min(w, h) * 0.03)

        points = []
        for i in range(self.num_streaks):
            angle = 2 * np.pi * i / self.num_streaks
            points.append([np.cos(angle), np.sin(angle)])
        aperture_poly = np.array(points)

        for i in range(1, self.ghost_count + 1):
            pos_x = int(source[0] + vec_x * i * self.ghost_spacing)
            pos_y = int(source[1] + vec_y * i * self.ghost_spacing)
            size = int(base_size * (self.ghost_size_decay ** i))
            if size < 2: continue
            
            # FIX: Use self.ghost_size_decay and apply opacity to the color
            opacity = 0.5 * (self.ghost_size_decay ** i)
            scaled_poly = (aperture_poly * size + np.array([pos_x, pos_y])).astype(np.int32)
            
            # Draw polygon with chromatic aberration
            for ch, shift in [(0, -1), (2, 1)]: # B, R channels
                color = [0,0,0]
                color[ch] = opacity
                offset = int(self.chromatic_aberration * i)
                shifted_poly = (aperture_poly * size + np.array([pos_x + shift * offset, pos_y])).astype(np.int32)
                cv2.fillPoly(ghosts_layer, [shifted_poly], color, lineType=cv2.LINE_AA)

        k_size = int(min(w, h) * 0.1) | 1
        return cv2.GaussianBlur(ghosts_layer, (k_size, k_size), 0)

    def _generate_streaks(self, shape, source):
        h, w = shape
        streaks_layer = np.zeros((h, w, 3), dtype=np.float32)
        diag = int(np.sqrt(h**2 + w**2))
        length = int(diag * self.streak_length_factor)

        for i in range(self.num_streaks):
            angle_rad = np.deg2rad(self.streak_angle_offset + (360.0 / self.num_streaks) * i)
            end_x = int(source[0] + length * np.cos(angle_rad))
            end_y = int(source[1] + length * np.sin(angle_rad))
            start_x = int(source[0] - length * np.cos(angle_rad))
            start_y = int(source[1] - length * np.sin(angle_rad))

            for ch, _, shift in [(0, (0.8, 0.8, 1), -1), (1, (1,1,1), 0), (2, (1, 0.8, 0.8), 1)]:
                offset = int(self.chromatic_aberration * self.intensity * 2 * shift)
                
                # FIX: Create a contiguous copy of the channel view to draw on
                channel_data = np.ascontiguousarray(streaks_layer[..., ch])
                cv2.line(channel_data, (start_x + offset, start_y), (end_x + offset, end_y), 1.0, 1, lineType=cv2.LINE_AA)
                streaks_layer[..., ch] = channel_data

        k_size = max(3, int(diag / (150 / self.streak_thin_factor))) | 1
        blur_kernel = np.zeros((k_size, k_size), dtype=np.float32)
        blur_kernel[k_size // 2, :] = 1
        streaks_layer = cv2.filter2D(streaks_layer, -1, blur_kernel)
        
        return streaks_layer / streaks_layer.max() if streaks_layer.max() > 0 else streaks_layer

    def _generate_haze(self, shape, source):
        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.sqrt((xx - source[0])**2 + (yy - source[1])**2)
        max_dist = np.sqrt(max(source[0], w - source[0])**2 + max(source[1], h - source[1])**2)
        
        falloff = (1 - (dist / max_dist))**3
        haze = np.expand_dims(falloff, axis=-1)
        
        color = np.array([[[0.8, 0.9, 1.0]]]) # Cool white haze
        return haze * color

    def apply(self, image):
        img_float = self._to_rgb_float(image)
        h, w, _ = img_float.shape

        lum = self._luminance(img_float)
        bright_mask = np.clip((lum - self.source_threshold) / (1 - self.source_threshold + 1e-6), 0, 1)

        source = self.manual_source or self._estimate_source(bright_mask)
        if source is None:
            return image

        image_center = (w // 2, h // 2)

        streaks = self._generate_streaks((h, w), source)
        ghosts = self._generate_ghosts((h, w), source, image_center)
        haze = self._generate_haze((h, w), source)

        glare_artifacts = (streaks + ghosts + haze) * self.intensity
        glare_artifacts = np.clip(glare_artifacts, 0, 1)

        out_float = self._screen_blend(img_float, glare_artifacts)
        out_float = self._screen_blend(out_float, bright_mask[..., None] * self.intensity * 0.3)

        return (np.clip(out_float, 0, 1) * 255).astype(np.uint8)

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
    parser.add_argument("--effect", type=str, choices=['rain', 'glare', 'dust'], required=True, help="The effect to apply.")
    
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
    elif args.effect == 'glare':
        effect = Glare()
        print("Applying glare effect...")
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
