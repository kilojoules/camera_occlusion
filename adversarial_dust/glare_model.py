"""Adversarial glare model: parametric lens flare/glare for CMA-ES optimization."""

import cv2
import numpy as np

from adversarial_dust.config import GlareConfig

# Parameters: source_x, source_y, intensity, streak_angle_offset, haze_spread, num_streaks_weight
NUM_GLARE_PARAMS = 6


class AdversarialGlareModel:
    """Parametric camera glare for adversarial optimization.

    Generates physically-inspired lens flare effects:
      - Starburst diffraction spikes
      - Soft haze/bloom around the light source
      - Lens ghosts (faint aperture reflections)

    Parameters (6 per model):
        source_x, source_y  - position of virtual light source (normalized 0-1)
        intensity           - overall glare strength (0 to max_intensity)
        streak_angle_offset - rotation of starburst pattern (degrees, 0-360)
        haze_spread         - how far the haze extends (0.5-3.0)
        streak_length       - length of diffraction spikes (0.5-3.0)
    """

    def __init__(
        self,
        config: GlareConfig,
        image_shape: tuple,
        budget_level: float,
    ):
        self.config = config
        self.image_h, self.image_w = image_shape[:2]
        self.budget_level = budget_level
        self.n_params = NUM_GLARE_PARAMS

    def parse_params(self, params: np.ndarray) -> dict:
        """Parse flat param vector into glare configuration dict."""
        params = np.asarray(params, dtype=np.float64)
        return {
            "source_x": np.clip(params[0], 0.0, 1.0),
            "source_y": np.clip(params[1], 0.0, 1.0),
            "intensity": np.clip(params[2], 0.0, self.config.max_intensity),
            "streak_angle": params[3] % 360.0,
            "haze_spread": np.clip(params[4], 0.5, 3.0),
            "streak_length": np.clip(params[5], 0.5, 3.0),
        }

    def _generate_haze(self, source: tuple, haze_spread: float) -> np.ndarray:
        """Generate soft radial haze/bloom centered on source."""
        h, w = self.image_h, self.image_w
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # Normalize to 0-1
        yy = yy / h
        xx = xx / w
        sx, sy = source

        dist = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
        max_dist = np.sqrt(max(sx, 1 - sx) ** 2 + max(sy, 1 - sy) ** 2) + 1e-6

        # Smooth falloff: higher haze_spread = wider bloom
        falloff = np.exp(-((dist / (max_dist * 0.3 * haze_spread)) ** 2))

        # Cool white haze color
        haze = np.zeros((h, w, 3), dtype=np.float32)
        haze[:, :, 0] = falloff * 0.9   # R
        haze[:, :, 1] = falloff * 0.95  # G
        haze[:, :, 2] = falloff * 1.0   # B
        return haze

    def _generate_streaks(
        self, source: tuple, streak_angle: float, streak_length: float
    ) -> np.ndarray:
        """Generate starburst diffraction spikes."""
        h, w = self.image_h, self.image_w
        streaks = np.zeros((h, w, 3), dtype=np.float32)

        sx_px = int(source[0] * w)
        sy_px = int(source[1] * h)
        diag = int(np.sqrt(h ** 2 + w ** 2))
        length = int(diag * streak_length * 0.3)

        num_streaks = self.config.num_streaks
        for i in range(num_streaks):
            angle_rad = np.deg2rad(streak_angle + (360.0 / num_streaks) * i)
            end_x = int(sx_px + length * np.cos(angle_rad))
            end_y = int(sy_px + length * np.sin(angle_rad))
            start_x = int(sx_px - length * np.cos(angle_rad))
            start_y = int(sy_px - length * np.sin(angle_rad))

            # Draw on each channel with slight chromatic offset
            for ch, shift in [(0, -1), (1, 0), (2, 1)]:
                offset = int(self.config.chromatic_aberration * shift)
                channel = np.ascontiguousarray(streaks[:, :, ch])
                cv2.line(
                    channel,
                    (start_x + offset, start_y),
                    (end_x + offset, end_y),
                    1.0, 1, lineType=cv2.LINE_AA,
                )
                streaks[:, :, ch] = channel

        # Directional blur to make streaks soft
        k_size = max(3, int(diag / 100)) | 1
        blur_kernel = np.zeros((k_size, k_size), dtype=np.float32)
        blur_kernel[k_size // 2, :] = 1.0
        k_sum = blur_kernel.sum()
        if k_sum > 0:
            blur_kernel /= k_sum
        streaks = cv2.filter2D(streaks, -1, blur_kernel)

        max_val = streaks.max()
        if max_val > 0:
            streaks /= max_val
        return streaks

    def _generate_ghosts(self, source: tuple) -> np.ndarray:
        """Generate lens ghost artifacts."""
        h, w = self.image_h, self.image_w
        ghosts = np.zeros((h, w, 3), dtype=np.float32)

        center = (0.5, 0.5)
        vec_x = center[0] - source[0]
        vec_y = center[1] - source[1]

        ghost_count = self.config.ghost_count
        base_size = int(min(w, h) * 0.03)

        for i in range(1, ghost_count + 1):
            pos_x = source[0] + vec_x * i * self.config.ghost_spacing
            pos_y = source[1] + vec_y * i * self.config.ghost_spacing
            px_x = int(pos_x * w)
            px_y = int(pos_y * h)
            size = max(2, int(base_size * (self.config.ghost_decay ** i)))
            opacity = 0.3 * (self.config.ghost_decay ** i)

            cv2.circle(ghosts, (px_x, px_y), size, (opacity, opacity * 0.8, opacity * 0.6), -1, lineType=cv2.LINE_AA)

        k_size = max(3, int(min(w, h) * 0.05)) | 1
        ghosts = cv2.GaussianBlur(ghosts, (k_size, k_size), 0)
        return ghosts

    def render_glare(self, glare_config: dict) -> np.ndarray:
        """Render full glare effect as a float32 additive layer (H, W, 3) in [0, 1]."""
        source = (glare_config["source_x"], glare_config["source_y"])
        intensity = glare_config["intensity"]

        haze = self._generate_haze(source, glare_config["haze_spread"])
        streaks = self._generate_streaks(
            source, glare_config["streak_angle"], glare_config["streak_length"]
        )
        ghosts = self._generate_ghosts(source)

        combined = (haze * 0.5 + streaks * 0.3 + ghosts * 0.2) * intensity
        return np.clip(combined, 0.0, 1.0)

    def get_alpha_mask(self, params: np.ndarray, timestep: int = 0) -> np.ndarray:
        """Return scalar alpha mask (max across channels) for visualization/budget."""
        glare_config = self.parse_params(params)
        glare_layer = self.render_glare(glare_config)
        # Use max across color channels as effective alpha
        alpha = np.max(glare_layer, axis=2)

        if self.budget_level < 1.0:
            alpha = self._project_alpha_to_budget(alpha)
        return alpha

    def _project_alpha_to_budget(self, alpha: np.ndarray) -> np.ndarray:
        """Scale alpha so coverage <= budget_level."""
        coverage = self.compute_coverage(alpha)
        if coverage <= self.budget_level:
            return alpha

        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2.0
            cov = self.compute_coverage(alpha * mid)
            if cov <= self.budget_level:
                lo = mid
            else:
                hi = mid
        return alpha * lo

    def compute_coverage(self, alpha_mask: np.ndarray) -> float:
        """Fraction of pixels with brightness above threshold."""
        return float(np.mean(alpha_mask > self.config.dirty_threshold))

    def apply(
        self,
        image: np.ndarray,
        params: np.ndarray,
        timestep: int = 0,
    ) -> np.ndarray:
        """Apply glare to image using screen blending.

        Screen blend: out = 1 - (1 - base) * (1 - layer)
        This brightens the image without clipping as harshly as additive.
        """
        input_dtype = image.dtype
        if input_dtype == np.uint8:
            img_float = image.astype(np.float32) / 255.0
        else:
            img_float = image.astype(np.float32)

        glare_config = self.parse_params(params)
        glare_layer = self.render_glare(glare_config)

        # Budget projection: scale intensity if needed
        if self.budget_level < 1.0:
            alpha = np.max(glare_layer, axis=2)
            coverage = self.compute_coverage(alpha)
            if coverage > self.budget_level:
                lo, hi = 0.0, 1.0
                for _ in range(30):
                    mid = (lo + hi) / 2.0
                    cov = self.compute_coverage(alpha * mid)
                    if cov <= self.budget_level:
                        lo = mid
                    else:
                        hi = mid
                glare_layer = glare_layer * lo

        # Screen blend
        blended = 1.0 - (1.0 - img_float) * (1.0 - glare_layer.astype(np.float32))
        blended = np.clip(blended, 0.0, 1.0).astype(np.float32)

        if input_dtype == np.uint8:
            return (blended * 255).astype(np.uint8)
        return blended

    def get_cma_bounds(self) -> tuple:
        """Return (lower_bounds, upper_bounds) for CMA-ES."""
        lb = [0.0, 0.0, 0.0, 0.0, 0.5, 0.5]
        ub = [1.0, 1.0, self.config.max_intensity, 360.0, 3.0, 3.0]
        return (lb, ub)

    def get_cma_x0(self) -> np.ndarray:
        """Return reasonable initial point for CMA-ES."""
        return np.array([
            0.5, 0.5,                      # source center
            self.config.max_intensity / 2,  # moderate intensity
            0.0,                            # no angle offset
            1.5,                            # moderate haze
            1.5,                            # moderate streaks
        ], dtype=np.float64)

    def get_random_params(self, rng: np.random.Generator) -> np.ndarray:
        """Generate random glare params."""
        lb, ub = self.get_cma_bounds()
        return rng.uniform(lb, ub)
