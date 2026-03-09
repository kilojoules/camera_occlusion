"""Tests for the adversarial dust model (no GPU/SimplerEnv needed)."""

import numpy as np
import pytest

from adversarial_dust.config import DustGridConfig
from adversarial_dust.dust_model import AdversarialDustModel


@pytest.fixture
def config():
    return DustGridConfig(
        grid_resolution=(8, 8),
        max_cell_opacity=0.6,
        dust_color=(180, 160, 140),
        gaussian_blur_sigma=3.0,
        dirty_threshold=0.05,
    )


@pytest.fixture
def image_shape():
    return (256, 256, 3)


@pytest.fixture
def sample_image(image_shape):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=image_shape, dtype=np.uint8)


class TestParamsToGrid:
    def test_reshape(self, config, image_shape):
        model = AdversarialDustModel(config, image_shape, budget_level=0.2)
        params = np.random.uniform(0, 0.6, size=64)
        grid = model.params_to_grid(params)
        assert grid.shape == (8, 8)

    def test_clipping(self, config, image_shape):
        model = AdversarialDustModel(config, image_shape, budget_level=0.2)
        params = np.full(64, 1.0)  # exceeds max_cell_opacity
        grid = model.params_to_grid(params)
        assert grid.max() <= config.max_cell_opacity

    def test_negative_clipping(self, config, image_shape):
        model = AdversarialDustModel(config, image_shape, budget_level=0.2)
        params = np.full(64, -0.5)
        grid = model.params_to_grid(params)
        assert grid.min() >= 0.0


class TestProjectToBudget:
    def test_zero_budget(self, config, image_shape):
        model = AdversarialDustModel(config, image_shape, budget_level=0.0)
        grid = np.full((8, 8), 0.3)
        projected = model.project_to_budget(grid)
        alpha_mask = model.grid_to_alpha_mask(projected)
        coverage = model.compute_coverage(alpha_mask)
        assert coverage == pytest.approx(0.0, abs=0.01)

    def test_small_budget_respected(self, config, image_shape):
        model = AdversarialDustModel(config, image_shape, budget_level=0.1)
        grid = np.full((8, 8), config.max_cell_opacity)
        projected = model.project_to_budget(grid)
        alpha_mask = model.grid_to_alpha_mask(projected)
        coverage = model.compute_coverage(alpha_mask)
        assert coverage <= 0.1 + 0.02  # small tolerance

    def test_under_budget_unchanged(self, config, image_shape):
        model = AdversarialDustModel(config, image_shape, budget_level=1.0)
        grid = np.full((8, 8), 0.01)  # very low opacity, well under any budget
        projected = model.project_to_budget(grid)
        np.testing.assert_array_almost_equal(grid, projected)

    def test_large_budget(self, config, image_shape):
        model = AdversarialDustModel(config, image_shape, budget_level=0.5)
        grid = np.full((8, 8), config.max_cell_opacity)
        projected = model.project_to_budget(grid)
        alpha_mask = model.grid_to_alpha_mask(projected)
        coverage = model.compute_coverage(alpha_mask)
        assert coverage <= 0.5 + 0.02


class TestApply:
    def test_output_shape_uint8(self, config, image_shape, sample_image):
        model = AdversarialDustModel(config, image_shape, budget_level=0.2)
        params = np.random.uniform(0, 0.3, size=64)
        result = model.apply(sample_image, params)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_output_shape_float(self, config, image_shape):
        model = AdversarialDustModel(config, image_shape, budget_level=0.2)
        img_float = np.random.uniform(0, 1, size=image_shape).astype(np.float32)
        params = np.random.uniform(0, 0.3, size=64)
        result = model.apply(img_float, params)
        assert result.shape == image_shape
        assert result.dtype == np.float32
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_zero_params_no_change(self, config, image_shape, sample_image):
        model = AdversarialDustModel(config, image_shape, budget_level=0.2)
        params = np.zeros(64)
        result = model.apply(sample_image, params)
        np.testing.assert_array_equal(result, sample_image)

    def test_blending_shifts_toward_dust_color(self, config, image_shape):
        model = AdversarialDustModel(config, image_shape, budget_level=1.0)
        # Pure black image
        img = np.zeros(image_shape, dtype=np.uint8)
        params = np.full(64, config.max_cell_opacity)
        result = model.apply(img, params)
        # Result should be shifted toward dust color
        assert result.mean() > 0


class TestGetRandomParams:
    def test_shape(self, config, image_shape):
        model = AdversarialDustModel(config, image_shape, budget_level=0.2)
        rng = np.random.default_rng(42)
        params = model.get_random_params(rng)
        assert params.shape == (64,)

    def test_budget_respected(self, config, image_shape):
        model = AdversarialDustModel(config, image_shape, budget_level=0.15)
        rng = np.random.default_rng(42)
        params = model.get_random_params(rng)
        grid = model.params_to_grid(params)
        alpha_mask = model.grid_to_alpha_mask(grid)
        coverage = model.compute_coverage(alpha_mask)
        assert coverage <= 0.15 + 0.02

    def test_different_seeds_different_results(self, config, image_shape):
        model = AdversarialDustModel(config, image_shape, budget_level=0.2)
        p1 = model.get_random_params(np.random.default_rng(1))
        p2 = model.get_random_params(np.random.default_rng(2))
        assert not np.allclose(p1, p2)


class TestComputeCoverage:
    def test_zero_coverage(self, config, image_shape):
        model = AdversarialDustModel(config, image_shape, budget_level=0.5)
        alpha_mask = np.zeros((256, 256))
        assert model.compute_coverage(alpha_mask) == 0.0

    def test_full_coverage(self, config, image_shape):
        model = AdversarialDustModel(config, image_shape, budget_level=0.5)
        alpha_mask = np.ones((256, 256))
        assert model.compute_coverage(alpha_mask) == 1.0

    def test_partial_coverage(self, config, image_shape):
        model = AdversarialDustModel(config, image_shape, budget_level=0.5)
        alpha_mask = np.zeros((256, 256))
        alpha_mask[:128, :] = 0.1  # above threshold
        coverage = model.compute_coverage(alpha_mask)
        assert coverage == pytest.approx(0.5, abs=0.01)
