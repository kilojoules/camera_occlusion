"""Adversarial dust model: low-res grid → alpha mask → dirty image pipeline."""

import cv2
import numpy as np

from adversarial_dust.config import DustGridConfig


class AdversarialDustModel:
    """Parameterizes camera dust as a low-resolution opacity grid.

    The grid is upsampled via bicubic interpolation and Gaussian-blurred
    to produce a smooth alpha mask at image resolution. A budget constraint
    limits the fraction of "dirty" pixels (opacity > threshold).
    """

    def __init__(
        self,
        config: DustGridConfig,
        image_shape: tuple,
        budget_level: float,
    ):
        self.config = config
        self.image_h, self.image_w = image_shape[:2]
        self.budget_level = budget_level
        self.grid_h, self.grid_w = config.grid_resolution
        self.n_params = self.grid_h * self.grid_w

        self.dust_color = np.array(config.dust_color, dtype=np.float32) / 255.0

    def params_to_grid(self, params: np.ndarray) -> np.ndarray:
        """Reshape flat params to 2D grid, clipped to [0, max_opacity]."""
        grid = np.array(params, dtype=np.float64).reshape(self.grid_h, self.grid_w)
        return np.clip(grid, 0.0, self.config.max_cell_opacity)

    def compute_coverage(self, alpha_mask: np.ndarray) -> float:
        """Fraction of pixels with opacity above the dirty threshold."""
        return float(np.mean(alpha_mask > self.config.dirty_threshold))

    def _upsample_grid(self, grid: np.ndarray) -> np.ndarray:
        """Bicubic upsample grid to image resolution + Gaussian blur."""
        grid_f32 = grid.astype(np.float32)
        upsampled = cv2.resize(
            grid_f32,
            (self.image_w, self.image_h),
            interpolation=cv2.INTER_CUBIC,
        )
        sigma = self.config.gaussian_blur_sigma
        if sigma > 0:
            ksize = int(6 * sigma + 1)
            if ksize % 2 == 0:
                ksize += 1
            upsampled = cv2.GaussianBlur(upsampled, (ksize, ksize), sigma)
        return np.clip(upsampled, 0.0, self.config.max_cell_opacity)

    def project_to_budget(self, grid: np.ndarray) -> np.ndarray:
        """Binary-search scale factor so upsampled coverage ≤ budget_level."""
        if self.budget_level >= 1.0:
            return grid

        alpha_mask = self._upsample_grid(grid)
        coverage = self.compute_coverage(alpha_mask)

        if coverage <= self.budget_level:
            return grid

        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2.0
            scaled = grid * mid
            alpha_mask = self._upsample_grid(scaled)
            cov = self.compute_coverage(alpha_mask)
            if cov <= self.budget_level:
                lo = mid
            else:
                hi = mid

        return grid * lo

    def grid_to_alpha_mask(self, grid: np.ndarray) -> np.ndarray:
        """Convert grid to full-resolution alpha mask."""
        return self._upsample_grid(grid)

    def get_alpha_mask(
        self, params: np.ndarray, timestep: int = 0
    ) -> np.ndarray:
        """Parse params and return the budget-projected alpha mask.

        The grid model is static, so timestep is ignored (accepted for API compat).
        """
        grid = self.params_to_grid(params)
        grid = self.project_to_budget(grid)
        return self.grid_to_alpha_mask(grid)

    def get_cma_bounds(self) -> tuple:
        """Return (lower_bounds, upper_bounds) arrays for CMA-ES."""
        lb = [0.0] * self.n_params
        ub = [self.config.max_cell_opacity] * self.n_params
        return (lb, ub)

    def get_cma_x0(self) -> np.ndarray:
        """Return a reasonable initial point for CMA-ES."""
        return np.full(self.n_params, self.config.max_cell_opacity / 2)

    def apply(
        self,
        image: np.ndarray,
        params: np.ndarray,
        timestep: int = 0,
    ) -> np.ndarray:
        """Full pipeline: params → grid → project → alpha mask → blended image.

        Args:
            image: Input image, uint8 HxWx3 or float32 HxWx3 in [0,1].
            params: Flat array of grid parameters.
            timestep: Ignored for grid model (accepted for API compat).

        Returns:
            Dirty image in the same dtype/range as input.
        """
        input_dtype = image.dtype
        if input_dtype == np.uint8:
            img_float = image.astype(np.float32) / 255.0
        else:
            img_float = image.astype(np.float32)

        grid = self.params_to_grid(params)
        grid = self.project_to_budget(grid)
        alpha = self.grid_to_alpha_mask(grid)

        # Alpha blend: output = dust_color * alpha + image * (1 - alpha)
        alpha_3d = alpha[:, :, np.newaxis]
        dust = self.dust_color[np.newaxis, np.newaxis, :]
        blended = dust * alpha_3d + img_float * (1.0 - alpha_3d)
        blended = np.clip(blended, 0.0, 1.0)

        if input_dtype == np.uint8:
            return (blended * 255).astype(np.uint8)
        return blended

    def get_random_params(self, rng: np.random.Generator) -> np.ndarray:
        """Generate random grid params projected to budget."""
        raw = rng.uniform(0, self.config.max_cell_opacity, size=self.n_params)
        grid = self.params_to_grid(raw)
        grid = self.project_to_budget(grid)
        return grid.flatten()
