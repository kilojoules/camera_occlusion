"""Adversarial fingerprint smudge model: parametric oily whorl patterns on camera lens."""

import cv2
import numpy as np

from adversarial_dust.config import FingerprintConfig

# Parameters per fingerprint: cx, cy, angle, scale, ridge_freq, ridge_amp, opacity, smear_angle
PARAMS_PER_PRINT = 8


class FingerprintSmudgeModel:
    """Parametric fingerprint smudge for camera occlusion.

    Each fingerprint has 8 optimizable parameters:
        cx, cy         - center position (normalized 0-1)
        angle          - rotation of whorl pattern (radians, 0 to 2*pi)
        scale          - overall size (config.scale_range)
        ridge_freq     - frequency of ridge lines (config.freq_range)
        ridge_amp      - amplitude/visibility of ridges (0 to 1)
        opacity        - peak opacity of the smudge (0 to max_opacity)
        smear_angle    - directional smear/swipe angle (radians, 0 to pi)

    The rendering pipeline:
      1. Generate concentric whorl pattern modulated by ridges
      2. Apply directional Gaussian blur (smear)
      3. Mask with smooth elliptical falloff
      4. Alpha blend with oily/greasy color onto image
    """

    def __init__(
        self,
        config: FingerprintConfig,
        image_shape: tuple,
        budget_level: float,
    ):
        self.config = config
        self.image_h, self.image_w = image_shape[:2]
        self.budget_level = budget_level
        self.num_prints = config.num_prints
        self.n_params = self.num_prints * PARAMS_PER_PRINT

        self.smudge_color = np.array(config.smudge_color, dtype=np.float32) / 255.0

        # Pre-compute normalized coordinate grids
        ys = np.linspace(0, 1, self.image_h, dtype=np.float64)
        xs = np.linspace(0, 1, self.image_w, dtype=np.float64)
        self.grid_x, self.grid_y = np.meshgrid(xs, ys)

    def parse_params(self, params: np.ndarray) -> list:
        """Parse flat param vector into list of fingerprint dicts."""
        params = np.asarray(params, dtype=np.float64)
        prints = []
        for i in range(self.num_prints):
            offset = i * PARAMS_PER_PRINT
            p = params[offset:offset + PARAMS_PER_PRINT]
            prints.append({
                "cx": np.clip(p[0], 0.0, 1.0),
                "cy": np.clip(p[1], 0.0, 1.0),
                "angle": p[2] % (2 * np.pi),
                "scale": np.clip(p[3], self.config.scale_range[0], self.config.scale_range[1]),
                "ridge_freq": np.clip(p[4], self.config.freq_range[0], self.config.freq_range[1]),
                "ridge_amp": np.clip(p[5], 0.0, 1.0),
                "opacity": np.clip(p[6], 0.0, self.config.max_opacity),
                "smear_angle": p[7] % np.pi,
            })
        return prints

    def _render_single_print(self, fp: dict) -> np.ndarray:
        """Render a single fingerprint smudge as an alpha mask."""
        cx, cy = fp["cx"], fp["cy"]
        angle = fp["angle"]
        scale = fp["scale"]
        ridge_freq = fp["ridge_freq"]
        ridge_amp = fp["ridge_amp"]
        opacity = fp["opacity"]
        smear_angle = fp["smear_angle"]

        # Translate coordinates relative to fingerprint center
        dx = self.grid_x - cx
        dy = self.grid_y - cy

        # Rotate coordinate system by the whorl angle
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rx = cos_a * dx + sin_a * dy
        ry = -sin_a * dx + cos_a * dy

        # Distance from center (in normalized coords, scaled by print size)
        dist = np.sqrt(rx ** 2 + ry ** 2) / max(scale, 1e-6)

        # Whorl pattern: concentric rings modulated by angular position
        theta = np.arctan2(ry, rx)
        # Spiral component: ridges follow r + theta/freq pattern
        spiral = np.sin(2 * np.pi * ridge_freq * (dist + theta / (2 * np.pi)))
        # Convert to 0-1 range and modulate by ridge amplitude
        ridges = 0.5 + 0.5 * ridge_amp * spiral

        # Smooth elliptical envelope (falloff from center)
        envelope = np.exp(-0.5 * (dist ** 2))
        envelope = np.clip(envelope, 0.0, 1.0)

        # Combine: ridges visible within envelope
        mask = opacity * envelope * ridges

        # Apply directional smear (anisotropic Gaussian blur)
        if scale > 0.02:
            smear_px = max(3, int(scale * min(self.image_h, self.image_w) * 0.3))
            if smear_px % 2 == 0:
                smear_px += 1
            # Create directional kernel
            kernel_size = smear_px
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            center_k = kernel_size // 2
            cos_s, sin_s = np.cos(smear_angle), np.sin(smear_angle)
            for k in range(kernel_size):
                offset = k - center_k
                kx = int(center_k + offset * cos_s)
                ky = int(center_k + offset * sin_s)
                if 0 <= kx < kernel_size and 0 <= ky < kernel_size:
                    kernel[ky, kx] = 1.0
            kernel_sum = kernel.sum()
            if kernel_sum > 0:
                kernel /= kernel_sum
            mask_f32 = mask.astype(np.float32)
            mask = cv2.filter2D(mask_f32, -1, kernel).astype(np.float64)

        return np.clip(mask, 0.0, self.config.max_opacity)

    def render_alpha_mask(self, prints: list, timestep: int = 0) -> np.ndarray:
        """Render all fingerprints into a combined alpha mask.

        Fingerprints are static, so timestep is ignored (API compat).
        """
        mask = np.zeros((self.image_h, self.image_w), dtype=np.float64)
        for fp in prints:
            fp_mask = self._render_single_print(fp)
            mask = np.maximum(mask, fp_mask)
        return np.clip(mask, 0.0, self.config.max_opacity)

    def compute_coverage(self, alpha_mask: np.ndarray) -> float:
        """Fraction of pixels with opacity above the dirty threshold."""
        return float(np.mean(alpha_mask > self.config.dirty_threshold))

    def project_to_budget(self, prints: list, timestep: int = 0) -> np.ndarray:
        """Scale smudge opacities so coverage <= budget_level."""
        alpha_mask = self.render_alpha_mask(prints, timestep)

        if self.budget_level >= 1.0:
            return alpha_mask

        coverage = self.compute_coverage(alpha_mask)
        if coverage <= self.budget_level:
            return alpha_mask

        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2.0
            scaled = alpha_mask * mid
            cov = self.compute_coverage(scaled)
            if cov <= self.budget_level:
                lo = mid
            else:
                hi = mid

        return alpha_mask * lo

    def get_alpha_mask(self, params: np.ndarray, timestep: int = 0) -> np.ndarray:
        """Parse params and return budget-projected alpha mask."""
        prints = self.parse_params(params)
        return self.project_to_budget(prints, timestep)

    def apply(
        self,
        image: np.ndarray,
        params: np.ndarray,
        timestep: int = 0,
    ) -> np.ndarray:
        """Full pipeline: params -> fingerprints -> alpha mask -> blended image.

        The smudge color simulates oily/greasy residue (slightly translucent,
        warmer tone than dust).
        """
        input_dtype = image.dtype
        if input_dtype == np.uint8:
            img_float = image.astype(np.float32) / 255.0
        else:
            img_float = image.astype(np.float32)

        alpha_mask = self.get_alpha_mask(params, timestep)

        alpha_3d = alpha_mask[:, :, np.newaxis].astype(np.float32)
        color = self.smudge_color[np.newaxis, np.newaxis, :]
        blended = color * alpha_3d + img_float * (1.0 - alpha_3d)
        blended = np.clip(blended, 0.0, 1.0)

        if input_dtype == np.uint8:
            return (blended * 255).astype(np.uint8)
        return blended

    def get_cma_bounds(self) -> tuple:
        """Return (lower_bounds, upper_bounds) for CMA-ES."""
        lb, ub = [], []
        for _ in range(self.num_prints):
            lb.extend([0.0, 0.0])                    # cx, cy
            ub.extend([1.0, 1.0])
            lb.append(0.0)                            # angle
            ub.append(2 * np.pi)
            lb.append(self.config.scale_range[0])     # scale
            ub.append(self.config.scale_range[1])
            lb.append(self.config.freq_range[0])      # ridge_freq
            ub.append(self.config.freq_range[1])
            lb.append(0.0)                            # ridge_amp
            ub.append(1.0)
            lb.append(0.0)                            # opacity
            ub.append(self.config.max_opacity)
            lb.append(0.0)                            # smear_angle
            ub.append(np.pi)
        return (lb, ub)

    def get_cma_x0(self) -> np.ndarray:
        """Return a reasonable initial point for CMA-ES."""
        x0 = []
        for _ in range(self.num_prints):
            x0.extend([0.5, 0.5])           # cx, cy: center
            x0.append(0.0)                   # angle: no rotation
            mid_scale = sum(self.config.scale_range) / 2
            x0.append(mid_scale)             # scale: midpoint
            mid_freq = sum(self.config.freq_range) / 2
            x0.append(mid_freq)              # ridge_freq: midpoint
            x0.append(0.5)                   # ridge_amp: moderate
            x0.append(self.config.max_opacity / 2)  # opacity: half
            x0.append(0.0)                   # smear_angle: horizontal
        return np.array(x0, dtype=np.float64)

    def get_random_params(self, rng: np.random.Generator) -> np.ndarray:
        """Generate random fingerprint params projected to budget."""
        lb, ub = self.get_cma_bounds()
        raw = rng.uniform(lb, ub)
        prints = self.parse_params(raw)
        self.project_to_budget(prints, timestep=0)
        return raw
