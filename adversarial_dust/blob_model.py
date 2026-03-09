"""Dynamic Gaussian blob dust model: moving/growing blobs → alpha mask → dirty image."""

import cv2
import numpy as np

from adversarial_dust.config import BlobConfig

# Number of CMA-ES parameters per blob
PARAMS_PER_BLOB = 8  # cx, cy, sigma_x, sigma_y, opacity, vx, vy, growth_rate


class DynamicBlobDustModel:
    """Parameterizes camera dust as a set of moving, growing Gaussian blobs.

    Each blob has 8 optimizable parameters:
        cx, cy       — initial center (normalized 0-1)
        sigma_x, sigma_y — initial spread (config.sigma_range)
        opacity      — peak opacity (0 to max_opacity)
        vx, vy       — velocity per step (config.velocity_range)
        growth_rate  — sigma growth multiplier per step (config.growth_rate_range)

    At timestep t:
        center(t) = (cx + vx*t, cy + vy*t)
        sigma(t)  = sigma * (1 + growth_rate * t)

    Overlapping blobs are combined with max (not additive).
    Budget is enforced per-frame via binary search.
    """

    def __init__(
        self,
        config: BlobConfig,
        image_shape: tuple,
        budget_level: float,
    ):
        self.config = config
        self.image_h, self.image_w = image_shape[:2]
        self.budget_level = budget_level
        self.num_blobs = config.num_blobs
        self.n_params = self.num_blobs * PARAMS_PER_BLOB

        self.dust_color = np.array(config.dust_color, dtype=np.float32) / 255.0

        # Pre-compute pixel coordinate grids (normalized 0-1)
        ys = np.linspace(0, 1, self.image_h, dtype=np.float64)
        xs = np.linspace(0, 1, self.image_w, dtype=np.float64)
        self.grid_x, self.grid_y = np.meshgrid(xs, ys)

    # ------------------------------------------------------------------
    # Parameter parsing
    # ------------------------------------------------------------------

    def parse_params(self, params: np.ndarray) -> list:
        """Parse flat param vector into a list of blob dicts."""
        params = np.asarray(params, dtype=np.float64)
        blobs = []
        for i in range(self.num_blobs):
            offset = i * PARAMS_PER_BLOB
            p = params[offset : offset + PARAMS_PER_BLOB]
            blobs.append(
                {
                    "cx": np.clip(p[0], 0.0, 1.0),
                    "cy": np.clip(p[1], 0.0, 1.0),
                    "sigma_x": np.clip(
                        p[2], self.config.sigma_range[0], self.config.sigma_range[1]
                    ),
                    "sigma_y": np.clip(
                        p[3], self.config.sigma_range[0], self.config.sigma_range[1]
                    ),
                    "opacity": np.clip(p[4], 0.0, self.config.max_opacity),
                    "vx": np.clip(
                        p[5],
                        self.config.velocity_range[0],
                        self.config.velocity_range[1],
                    ),
                    "vy": np.clip(
                        p[6],
                        self.config.velocity_range[0],
                        self.config.velocity_range[1],
                    ),
                    "growth_rate": np.clip(
                        p[7],
                        self.config.growth_rate_range[0],
                        self.config.growth_rate_range[1],
                    ),
                }
            )
        return blobs

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_alpha_mask(
        self, blobs: list, timestep: int = 0
    ) -> np.ndarray:
        """Render all blobs into a full-resolution alpha mask at the given timestep.

        Uses max-combination of overlapping blobs.
        """
        mask = np.zeros((self.image_h, self.image_w), dtype=np.float64)

        for b in blobs:
            # Evolve center and sigma with time
            cx_t = b["cx"] + b["vx"] * timestep
            cy_t = b["cy"] + b["vy"] * timestep
            sx_t = b["sigma_x"] * (1.0 + b["growth_rate"] * timestep)
            sy_t = b["sigma_y"] * (1.0 + b["growth_rate"] * timestep)

            # Clamp sigma to stay positive
            sx_t = max(sx_t, 1e-6)
            sy_t = max(sy_t, 1e-6)

            # Gaussian blob
            dx = self.grid_x - cx_t
            dy = self.grid_y - cy_t
            exponent = -0.5 * ((dx / sx_t) ** 2 + (dy / sy_t) ** 2)
            blob_mask = b["opacity"] * np.exp(exponent)

            # Max combination
            mask = np.maximum(mask, blob_mask)

        return np.clip(mask, 0.0, self.config.max_opacity)

    # ------------------------------------------------------------------
    # Budget enforcement
    # ------------------------------------------------------------------

    def compute_coverage(self, alpha_mask: np.ndarray) -> float:
        """Fraction of pixels with opacity above the dirty threshold."""
        return float(np.mean(alpha_mask > self.config.dirty_threshold))

    def project_to_budget(
        self, blobs: list, timestep: int = 0
    ) -> np.ndarray:
        """Scale blob opacities so coverage <= budget_level at given timestep.

        Returns the (possibly scaled) alpha mask.
        """
        alpha_mask = self.render_alpha_mask(blobs, timestep)

        if self.budget_level >= 1.0:
            return alpha_mask

        coverage = self.compute_coverage(alpha_mask)
        if coverage <= self.budget_level:
            return alpha_mask

        # Binary search for a scale factor on the mask
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

    # ------------------------------------------------------------------
    # Alpha mask accessor (for animation / visualization)
    # ------------------------------------------------------------------

    def get_alpha_mask(
        self, params: np.ndarray, timestep: int = 0
    ) -> np.ndarray:
        """Parse params and return the budget-projected alpha mask."""
        blobs = self.parse_params(params)
        return self.project_to_budget(blobs, timestep)

    # ------------------------------------------------------------------
    # Apply: full pipeline
    # ------------------------------------------------------------------

    def apply(
        self,
        image: np.ndarray,
        params: np.ndarray,
        timestep: int = 0,
    ) -> np.ndarray:
        """Full pipeline: params → blobs → render → budget project → blend.

        Args:
            image: Input image, uint8 HxWx3 or float32 HxWx3 in [0,1].
            params: Flat array of blob parameters (num_blobs * 8).
            timestep: Current episode timestep for dynamics.

        Returns:
            Dirty image in same dtype/range as input.
        """
        input_dtype = image.dtype
        if input_dtype == np.uint8:
            img_float = image.astype(np.float32) / 255.0
        else:
            img_float = image.astype(np.float32)

        alpha_mask = self.get_alpha_mask(params, timestep)

        # Alpha blend
        alpha_3d = alpha_mask[:, :, np.newaxis].astype(np.float32)
        dust = self.dust_color[np.newaxis, np.newaxis, :]
        blended = dust * alpha_3d + img_float * (1.0 - alpha_3d)
        blended = np.clip(blended, 0.0, 1.0)

        if input_dtype == np.uint8:
            return (blended * 255).astype(np.uint8)
        return blended

    # ------------------------------------------------------------------
    # CMA-ES interface
    # ------------------------------------------------------------------

    def get_cma_bounds(self) -> tuple:
        """Return (lower_bounds, upper_bounds) arrays for CMA-ES."""
        lb = []
        ub = []
        for _ in range(self.num_blobs):
            # cx, cy
            lb.extend([0.0, 0.0])
            ub.extend([1.0, 1.0])
            # sigma_x, sigma_y
            lb.extend([self.config.sigma_range[0], self.config.sigma_range[0]])
            ub.extend([self.config.sigma_range[1], self.config.sigma_range[1]])
            # opacity
            lb.append(0.0)
            ub.append(self.config.max_opacity)
            # vx, vy
            lb.extend([self.config.velocity_range[0], self.config.velocity_range[0]])
            ub.extend([self.config.velocity_range[1], self.config.velocity_range[1]])
            # growth_rate
            lb.append(self.config.growth_rate_range[0])
            ub.append(self.config.growth_rate_range[1])
        return (lb, ub)

    def get_cma_x0(self) -> np.ndarray:
        """Return a reasonable initial point for CMA-ES."""
        x0 = []
        for _ in range(self.num_blobs):
            # cx, cy — center of image
            x0.extend([0.5, 0.5])
            # sigma_x, sigma_y — midpoint of range
            mid_sigma = (
                self.config.sigma_range[0] + self.config.sigma_range[1]
            ) / 2.0
            x0.extend([mid_sigma, mid_sigma])
            # opacity — half of max
            x0.append(self.config.max_opacity / 2.0)
            # vx, vy — zero (static start)
            x0.extend([0.0, 0.0])
            # growth_rate — zero (no growth start)
            x0.append(0.0)
        return np.array(x0, dtype=np.float64)

    def get_random_params(self, rng: np.random.Generator) -> np.ndarray:
        """Generate random blob params projected to budget."""
        lb, ub = self.get_cma_bounds()
        raw = rng.uniform(lb, ub)
        # Verify budget compliance at t=0
        blobs = self.parse_params(raw)
        self.project_to_budget(blobs, timestep=0)
        return raw
