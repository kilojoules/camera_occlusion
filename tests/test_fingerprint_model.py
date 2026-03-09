"""Tests for the fingerprint smudge model (no GPU/SimplerEnv needed)."""

import numpy as np
import pytest

from adversarial_dust.config import FingerprintConfig
from adversarial_dust.fingerprint_model import FingerprintSmudgeModel, PARAMS_PER_PRINT


@pytest.fixture
def config():
    return FingerprintConfig(
        num_prints=2,
        max_opacity=0.5,
        smudge_color=(200, 190, 170),
        dirty_threshold=0.05,
        scale_range=(0.05, 0.25),
        freq_range=(3.0, 15.0),
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
        model = FingerprintSmudgeModel(config, image_shape, budget_level=0.2)
        assert model.n_params == config.num_prints * PARAMS_PER_PRINT

    def test_parse_returns_list(self, config, image_shape):
        model = FingerprintSmudgeModel(config, image_shape, budget_level=0.2)
        params = np.random.uniform(0, 1, size=model.n_params)
        prints = model.parse_params(params)
        assert len(prints) == config.num_prints
        assert all("cx" in p for p in prints)

    def test_clipping(self, config, image_shape):
        model = FingerprintSmudgeModel(config, image_shape, budget_level=0.2)
        params = np.full(model.n_params, 100.0)
        prints = model.parse_params(params)
        for p in prints:
            assert p["cx"] <= 1.0
            assert p["opacity"] <= config.max_opacity


class TestRenderAlphaMask:
    def test_output_shape(self, config, image_shape):
        model = FingerprintSmudgeModel(config, image_shape, budget_level=0.5)
        params = model.get_cma_x0()
        prints = model.parse_params(params)
        mask = model.render_alpha_mask(prints)
        assert mask.shape == (256, 256)

    def test_values_in_range(self, config, image_shape):
        model = FingerprintSmudgeModel(config, image_shape, budget_level=0.5)
        params = model.get_cma_x0()
        prints = model.parse_params(params)
        mask = model.render_alpha_mask(prints)
        assert mask.min() >= 0.0
        assert mask.max() <= config.max_opacity + 1e-6


class TestBudgetProjection:
    def test_budget_respected(self, config, image_shape):
        model = FingerprintSmudgeModel(config, image_shape, budget_level=0.1)
        params = np.random.uniform(0, 1, size=model.n_params)
        params[6] = config.max_opacity  # max opacity
        params[6 + PARAMS_PER_PRINT] = config.max_opacity
        alpha = model.get_alpha_mask(params)
        coverage = model.compute_coverage(alpha)
        assert coverage <= 0.1 + 0.02

    def test_full_budget_no_scaling(self, config, image_shape):
        model = FingerprintSmudgeModel(config, image_shape, budget_level=1.0)
        params = model.get_cma_x0()
        prints = model.parse_params(params)
        mask = model.render_alpha_mask(prints)
        projected = model.project_to_budget(prints)
        # With full budget, projected should match raw render
        np.testing.assert_array_almost_equal(mask, projected, decimal=5)


class TestApply:
    def test_output_shape_uint8(self, config, image_shape, sample_image):
        model = FingerprintSmudgeModel(config, image_shape, budget_level=0.2)
        params = model.get_cma_x0()
        result = model.apply(sample_image, params)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_output_shape_float(self, config, image_shape):
        model = FingerprintSmudgeModel(config, image_shape, budget_level=0.2)
        img_float = np.random.uniform(0, 1, size=image_shape).astype(np.float32)
        params = model.get_cma_x0()
        result = model.apply(img_float, params)
        assert result.shape == image_shape
        assert result.dtype == np.float32
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_zero_opacity_no_change(self, config, image_shape, sample_image):
        model = FingerprintSmudgeModel(config, image_shape, budget_level=0.2)
        params = np.zeros(model.n_params)  # zero opacity
        result = model.apply(sample_image, params)
        np.testing.assert_array_equal(result, sample_image)


class TestCMAInterface:
    def test_bounds_shape(self, config, image_shape):
        model = FingerprintSmudgeModel(config, image_shape, budget_level=0.2)
        lb, ub = model.get_cma_bounds()
        assert len(lb) == model.n_params
        assert len(ub) == model.n_params
        assert all(l <= u for l, u in zip(lb, ub))

    def test_x0_shape(self, config, image_shape):
        model = FingerprintSmudgeModel(config, image_shape, budget_level=0.2)
        x0 = model.get_cma_x0()
        assert x0.shape == (model.n_params,)

    def test_x0_within_bounds(self, config, image_shape):
        model = FingerprintSmudgeModel(config, image_shape, budget_level=0.2)
        x0 = model.get_cma_x0()
        lb, ub = model.get_cma_bounds()
        for i in range(len(x0)):
            assert lb[i] <= x0[i] <= ub[i]

    def test_random_params_shape(self, config, image_shape):
        model = FingerprintSmudgeModel(config, image_shape, budget_level=0.2)
        rng = np.random.default_rng(42)
        params = model.get_random_params(rng)
        assert params.shape == (model.n_params,)

    def test_different_seeds(self, config, image_shape):
        model = FingerprintSmudgeModel(config, image_shape, budget_level=0.2)
        p1 = model.get_random_params(np.random.default_rng(1))
        p2 = model.get_random_params(np.random.default_rng(2))
        assert not np.allclose(p1, p2)


class TestComputeCoverage:
    def test_zero(self, config, image_shape):
        model = FingerprintSmudgeModel(config, image_shape, budget_level=0.5)
        assert model.compute_coverage(np.zeros((256, 256))) == 0.0

    def test_full(self, config, image_shape):
        model = FingerprintSmudgeModel(config, image_shape, budget_level=0.5)
        assert model.compute_coverage(np.ones((256, 256))) == 1.0
