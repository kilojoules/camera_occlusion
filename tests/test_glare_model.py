"""Tests for the adversarial glare model (no GPU/SimplerEnv needed)."""

import numpy as np
import pytest

from adversarial_dust.config import GlareConfig
from adversarial_dust.glare_model import AdversarialGlareModel, NUM_GLARE_PARAMS


@pytest.fixture
def config():
    return GlareConfig(
        max_intensity=0.8,
        num_streaks=6,
        chromatic_aberration=1.5,
        ghost_count=4,
        ghost_spacing=0.4,
        ghost_decay=0.85,
        dirty_threshold=0.05,
    )


@pytest.fixture
def image_shape():
    return (256, 256, 3)


@pytest.fixture
def sample_image(image_shape):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=image_shape, dtype=np.uint8)


class TestParseParams:
    def test_correct_count(self, config, image_shape):
        model = AdversarialGlareModel(config, image_shape, budget_level=0.2)
        assert model.n_params == NUM_GLARE_PARAMS

    def test_parse_keys(self, config, image_shape):
        model = AdversarialGlareModel(config, image_shape, budget_level=0.2)
        params = np.array([0.5, 0.5, 0.4, 30.0, 1.5, 1.5])
        gc = model.parse_params(params)
        assert "source_x" in gc
        assert "intensity" in gc
        assert "streak_angle" in gc

    def test_clipping(self, config, image_shape):
        model = AdversarialGlareModel(config, image_shape, budget_level=0.2)
        params = np.full(NUM_GLARE_PARAMS, 100.0)
        gc = model.parse_params(params)
        assert gc["source_x"] <= 1.0
        assert gc["intensity"] <= config.max_intensity


class TestRenderGlare:
    def test_output_shape(self, config, image_shape):
        model = AdversarialGlareModel(config, image_shape, budget_level=0.5)
        params = model.get_cma_x0()
        gc = model.parse_params(params)
        glare = model.render_glare(gc)
        assert glare.shape == (256, 256, 3)

    def test_values_in_range(self, config, image_shape):
        model = AdversarialGlareModel(config, image_shape, budget_level=0.5)
        params = model.get_cma_x0()
        gc = model.parse_params(params)
        glare = model.render_glare(gc)
        assert glare.min() >= 0.0
        assert glare.max() <= 1.0


class TestGetAlphaMask:
    def test_output_shape(self, config, image_shape):
        model = AdversarialGlareModel(config, image_shape, budget_level=0.5)
        params = model.get_cma_x0()
        alpha = model.get_alpha_mask(params)
        assert alpha.shape == (256, 256)

    def test_budget_respected(self, config, image_shape):
        model = AdversarialGlareModel(config, image_shape, budget_level=0.1)
        params = model.get_cma_x0()
        params[2] = config.max_intensity  # max intensity
        alpha = model.get_alpha_mask(params)
        coverage = model.compute_coverage(alpha)
        assert coverage <= 0.1 + 0.02


class TestApply:
    def test_output_shape_uint8(self, config, image_shape, sample_image):
        model = AdversarialGlareModel(config, image_shape, budget_level=0.2)
        params = model.get_cma_x0()
        result = model.apply(sample_image, params)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_output_shape_float(self, config, image_shape):
        model = AdversarialGlareModel(config, image_shape, budget_level=0.2)
        img_float = np.random.uniform(0, 1, size=image_shape).astype(np.float32)
        params = model.get_cma_x0()
        result = model.apply(img_float, params)
        assert result.shape == image_shape
        assert result.dtype == np.float32
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_screen_blend_brightens(self, config, image_shape):
        """Screen blend should only brighten, never darken."""
        model = AdversarialGlareModel(config, image_shape, budget_level=1.0)
        img = np.full(image_shape, 128, dtype=np.uint8)
        params = model.get_cma_x0()
        params[2] = 0.5  # moderate intensity
        result = model.apply(img, params)
        # On average, screen-blended image should be >= original
        assert result.mean() >= img.mean() - 1.0

    def test_zero_intensity_minimal_change(self, config, image_shape, sample_image):
        model = AdversarialGlareModel(config, image_shape, budget_level=0.2)
        params = model.get_cma_x0()
        params[2] = 0.0  # zero intensity
        result = model.apply(sample_image, params)
        # Should be nearly identical (screen blend of zero = identity)
        np.testing.assert_array_almost_equal(
            result.astype(float), sample_image.astype(float), decimal=0
        )


class TestCMAInterface:
    def test_bounds_shape(self, config, image_shape):
        model = AdversarialGlareModel(config, image_shape, budget_level=0.2)
        lb, ub = model.get_cma_bounds()
        assert len(lb) == model.n_params
        assert len(ub) == model.n_params

    def test_x0_shape(self, config, image_shape):
        model = AdversarialGlareModel(config, image_shape, budget_level=0.2)
        x0 = model.get_cma_x0()
        assert x0.shape == (model.n_params,)

    def test_x0_within_bounds(self, config, image_shape):
        model = AdversarialGlareModel(config, image_shape, budget_level=0.2)
        x0 = model.get_cma_x0()
        lb, ub = model.get_cma_bounds()
        for i in range(len(x0)):
            assert lb[i] <= x0[i] <= ub[i]

    def test_random_params(self, config, image_shape):
        model = AdversarialGlareModel(config, image_shape, budget_level=0.2)
        rng = np.random.default_rng(42)
        params = model.get_random_params(rng)
        assert params.shape == (model.n_params,)


class TestComputeCoverage:
    def test_zero(self, config, image_shape):
        model = AdversarialGlareModel(config, image_shape, budget_level=0.5)
        assert model.compute_coverage(np.zeros((256, 256))) == 0.0

    def test_full(self, config, image_shape):
        model = AdversarialGlareModel(config, image_shape, budget_level=0.5)
        assert model.compute_coverage(np.ones((256, 256))) == 1.0
