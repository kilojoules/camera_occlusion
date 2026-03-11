"""Physics-based fingerprint smudge model for camera lens occlusion.

Based on the image formation model from Gu et al. (SIGGRAPH Asia 2009):
  I = I_0 * (alpha * k) + I_alpha * k

where alpha is the attenuation pattern of the oil film, k is the defocus
blur kernel (dirt on the lens is heavily out of focus), and I_alpha is the
scattered/intensified light from the oil film itself.

A fingerprint on a camera lens produces two effects:
  1. Attenuation: the oil film reduces local contrast (scene light is
     partially absorbed/scattered away from the sensor).
  2. Scattering (veiling glare): the oil scatters ambient light toward the
     sensor, creating a bright hazy glow over the affected region.

Because the fingerprint sits on the front element (far from the focal
plane), its pattern is heavily defocused — the sharp ridges of the print
become soft, low-frequency blobs in the captured image.
"""

import cv2
import numpy as np

from adversarial_dust.config import FingerprintConfig

# Parameters per fingerprint: cx, cy, angle, scale, ridge_freq, ridge_amp, opacity, smear_angle
PARAMS_PER_PRINT = 8


class FingerprintSmudgeModel:
    """Physics-based fingerprint smudge on a camera lens.

    Each fingerprint has 8 optimizable parameters:
        cx, cy         - center position (normalized 0-1)
        angle          - rotation of whorl pattern (radians, 0 to 2*pi)
        scale          - overall size (config.scale_range)
        ridge_freq     - frequency of ridge lines (config.freq_range)
        ridge_amp      - amplitude/visibility of ridges (0 to 1)
        opacity        - peak oil thickness / attenuation strength (0 to max_opacity)
        smear_angle    - directional smear/swipe angle (radians, 0 to pi)

    Rendering pipeline (physics-based):
      1. Generate the oil-film thickness pattern (whorl ridges + envelope)
      2. Derive attenuation alpha from oil thickness
      3. Apply heavy defocus blur (the print is on the lens, far from focal plane)
      4. Compute scattered light (veiling glare) from the oil film
      5. Combine: I = I_0 * blurred_alpha + scatter_intensity
    """

    # Defocus kernel radius as fraction of image diagonal.
    # Small blur — softens ridge detail but preserves the thick,
    # greasy smear character of a heavy oil deposit.
    DEFOCUS_FRAC = 0.015

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

        # Pre-compute normalized coordinate grids
        ys = np.linspace(0, 1, self.image_h, dtype=np.float64)
        xs = np.linspace(0, 1, self.image_w, dtype=np.float64)
        self.grid_x, self.grid_y = np.meshgrid(xs, ys)

        # Defocus blur kernel size in pixels
        diag = np.sqrt(self.image_h ** 2 + self.image_w ** 2)
        self._defocus_ksize = int(diag * self.DEFOCUS_FRAC) | 1  # ensure odd

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

    def _render_oil_thickness(self, fp: dict) -> np.ndarray:
        """Render the oil-film thickness pattern for one fingerprint.

        Returns a 2-D array in [0, opacity] representing how much oil
        is deposited at each pixel (before defocus blur).
        """
        cx, cy = fp["cx"], fp["cy"]
        angle = fp["angle"]
        scale = fp["scale"]
        ridge_freq = fp["ridge_freq"]
        ridge_amp = fp["ridge_amp"]
        opacity = fp["opacity"]
        smear_angle = fp["smear_angle"]

        dx = self.grid_x - cx
        dy = self.grid_y - cy

        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rx = cos_a * dx + sin_a * dy
        ry = -sin_a * dx + cos_a * dy

        dist = np.sqrt(rx ** 2 + ry ** 2) / max(scale, 1e-6)

        # Whorl/spiral ridge pattern
        theta = np.arctan2(ry, rx)
        spiral = np.sin(2 * np.pi * ridge_freq * (dist + theta / (2 * np.pi)))
        ridges = 0.5 + 0.5 * ridge_amp * spiral

        # Flat-topped envelope: a greasy finger leaves a thick, uniform
        # deposit across the whole contact patch — not a gentle falloff.
        # Clipped cosine gives a hard-edged pad of oil.
        envelope = np.clip(1.0 - dist ** 2, 0.0, 1.0)

        # Oil thickness = opacity * envelope * ridge modulation
        thickness = opacity * envelope * ridges

        # Optional directional smear (finger swipe direction)
        if scale > 0.02:
            smear_px = max(3, int(scale * min(self.image_h, self.image_w) * 0.3))
            if smear_px % 2 == 0:
                smear_px += 1
            kernel = np.zeros((smear_px, smear_px), dtype=np.float32)
            center_k = smear_px // 2
            cos_s, sin_s = np.cos(smear_angle), np.sin(smear_angle)
            for k in range(smear_px):
                off = k - center_k
                kx = int(center_k + off * cos_s)
                ky = int(center_k + off * sin_s)
                if 0 <= kx < smear_px and 0 <= ky < smear_px:
                    kernel[ky, kx] = 1.0
            ks = kernel.sum()
            if ks > 0:
                kernel /= ks
            thickness = cv2.filter2D(
                thickness.astype(np.float32), -1, kernel
            ).astype(np.float64)

        return np.clip(thickness, 0.0, self.config.max_opacity)

    def _defocus_blur(self, img: np.ndarray) -> np.ndarray:
        """Apply heavy Gaussian defocus blur (simulates out-of-focus dirt)."""
        ks = self._defocus_ksize
        sigma = ks / 4.0
        if img.ndim == 2:
            return cv2.GaussianBlur(
                img.astype(np.float32), (ks, ks), sigma
            ).astype(img.dtype)
        return cv2.GaussianBlur(
            img.astype(np.float32), (ks, ks), sigma
        ).astype(img.dtype)

    def render_alpha_mask(self, prints: list, timestep: int = 0) -> np.ndarray:
        """Render all fingerprints into a combined attenuation mask.

        Returns the defocus-blurred attenuation pattern (0 = no effect,
        higher = more attenuation).  This is (1 - blurred_alpha) in
        the Gu et al. model, i.e. the fraction of light *lost*.
        """
        # Combine oil thickness maps (additive — overlapping smears reinforce)
        thickness = np.zeros((self.image_h, self.image_w), dtype=np.float64)
        for fp in prints:
            fp_thick = self._render_oil_thickness(fp)
            thickness += fp_thick
        thickness = np.clip(thickness, 0.0, self.config.max_opacity)

        # Defocus blur softens the ridge detail into blobs
        blurred = self._defocus_blur(thickness)

        # The blur dilutes peak values — compensate so that thick grease
        # deposits remain nearly opaque.  A real grease smear on glass is
        # opaque in the center; you can't see straight through it.
        blurred = np.clip(blurred * 3.0, 0.0, self.config.max_opacity)
        return blurred

    def compute_coverage(self, alpha_mask: np.ndarray) -> float:
        """Fraction of pixels with attenuation above the dirty threshold."""
        return float(np.mean(alpha_mask > self.config.dirty_threshold))

    def project_to_budget(self, prints: list, timestep: int = 0) -> np.ndarray:
        """Scale attenuation so coverage <= budget_level."""
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
        """Parse params and return budget-projected attenuation mask."""
        prints = self.parse_params(params)
        return self.project_to_budget(prints, timestep)

    def apply(
        self,
        image: np.ndarray,
        params: np.ndarray,
        timestep: int = 0,
    ) -> np.ndarray:
        """Apply physics-based fingerprint degradation to an image.

        Model: I = I_0 * (1 - attenuation) + c * scatter
        where:
          - attenuation is the defocus-blurred oil thickness (contrast loss)
          - scatter is the veiling glare from the oil film (bright haze)
          - c scales the scatter intensity relative to the scene
        """
        input_dtype = image.dtype
        if input_dtype == np.uint8:
            img_float = image.astype(np.float32) / 255.0
        else:
            img_float = image.astype(np.float32)

        alpha_mask = self.get_alpha_mask(params, timestep)
        attn = alpha_mask[:, :, np.newaxis].astype(np.float32)

        # (1) Attenuation: scene light blocked/scattered by grease
        attenuated = img_float * (1.0 - attn)

        # (2) Visible grease deposit: oil/grease on glass is a translucent
        # milky film — mostly white with a very slight warm cast, like
        # smearing Vaseline on a lens.  (BGR order for OpenCV)
        grease_color = np.array([0.38, 0.40, 0.42], dtype=np.float32)
        grease_layer = attn * grease_color * 2.0

        # (3) Veiling glare: grease scatters ambient light into a bright
        # milky haze — nearly white, slight warmth.
        scene_brightness = img_float.mean()
        scatter_tint = np.array([0.90, 0.95, 1.0], dtype=np.float32)  # BGR near-white
        scatter = (attn ** 2) * scene_brightness * scatter_tint * 1.5

        result = attenuated + grease_layer + scatter
        result = np.clip(result, 0.0, 1.0)

        if input_dtype == np.uint8:
            return (result * 255).astype(np.uint8)
        return result

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
