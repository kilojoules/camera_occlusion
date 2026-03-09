"""Tests for the dynamic Gaussian blob dust model (no GPU/SimplerEnv needed)."""

import numpy as np
import pytest

from adversarial_dust.config import BlobConfig
from adversarial_dust.blob_model import DynamicBlobDustModel, PARAMS_PER_BLOB


@pytest.fixture
def config():
    return BlobConfig(
        num_blobs=5,
        max_opacity=0.6,
        dust_color=(180, 160, 140),
        dirty_threshold=0.05,
        sigma_range=(0.02, 0.30),
        velocity_range=(-0.005, 0.005),
        growth_rate_range=(-0.002, 0.005),
    )


@pytest.fixture
def image_shape():
    return (256, 256, 3)


@pytest.fixture
def model(config, image_shape):
    return DynamicBlobDustModel(config, image_shape, budget_level=0.2)


@pytest.fixture
def sample_image(image_shape):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=image_shape, dtype=np.uint8)


class TestParseParams:
    def test_num_blobs(self, model):
        params = model.get_cma_x0()
        blobs = model.parse_params(params)
        assert len(blobs) == 5

    def test_blob_keys(self, model):
        blobs = model.parse_params(model.get_cma_x0())
        expected_keys = {"cx", "cy", "sigma_x", "sigma_y", "opacity", "vx", "vy", "growth_rate"}
        assert set(blobs[0].keys()) == expected_keys

    def test_clipping_opacity(self, config, image_shape):
        model = DynamicBlobDustModel(config, image_shape, budget_level=0.5)
        params = np.ones(model.n_params) * 10.0  # way over max
        blobs = model.parse_params(params)
        for b in blobs:
            assert b["opacity"] <= config.max_opacity

    def test_clipping_center(self, model):
        params = np.full(model.n_params, -5.0)
        blobs = model.parse_params(params)
        for b in blobs:
            assert b["cx"] >= 0.0
            assert b["cy"] >= 0.0

    def test_clipping_velocity(self, config, image_shape):
        model = DynamicBlobDustModel(config, image_shape, budget_level=0.5)
        params = np.ones(model.n_params) * 10.0
        blobs = model.parse_params(params)
        for b in blobs:
            assert b["vx"] <= config.velocity_range[1]
            assert b["vy"] <= config.velocity_range[1]


class TestRenderAlphaMask:
    def test_output_shape(self, model):
        blobs = model.parse_params(model.get_cma_x0())
        mask = model.render_alpha_mask(blobs, timestep=0)
        assert mask.shape == (256, 256)

    def test_output_range(self, model):
        blobs = model.parse_params(model.get_cma_x0())
        mask = model.render_alpha_mask(blobs, timestep=0)
        assert mask.min() >= 0.0
        assert mask.max() <= model.config.max_opacity

    def test_zero_opacity_blobs(self, config, image_shape):
        model = DynamicBlobDustModel(config, image_shape, budget_level=0.5)
        params = model.get_cma_x0()
        # Zero out all opacity values
        for i in range(config.num_blobs):
            params[i * PARAMS_PER_BLOB + 4] = 0.0
        blobs = model.parse_params(params)
        mask = model.render_alpha_mask(blobs)
        assert mask.max() == pytest.approx(0.0, abs=1e-10)

    def test_single_blob_centered(self, image_shape):
        cfg = BlobConfig(num_blobs=1, max_opacity=0.6, sigma_range=(0.05, 0.30))
        model = DynamicBlobDustModel(cfg, image_shape, budget_level=1.0)
        # Single blob at center, moderate sigma
        params = np.array([0.5, 0.5, 0.10, 0.10, 0.6, 0.0, 0.0, 0.0])
        blobs = model.parse_params(params)
        mask = model.render_alpha_mask(blobs)
        # Peak should be at center
        cy, cx = 128, 128  # center of 256x256
        assert mask[cy, cx] == pytest.approx(0.6, abs=0.01)


class TestDynamics:
    def test_blob_moves_with_velocity(self, image_shape):
        cfg = BlobConfig(num_blobs=1, max_opacity=0.6, sigma_range=(0.02, 0.30),
                         velocity_range=(-0.01, 0.01))
        model = DynamicBlobDustModel(cfg, image_shape, budget_level=1.0)
        # Blob moving right: vx > 0
        params = np.array([0.3, 0.5, 0.05, 0.05, 0.5, 0.005, 0.0, 0.0])
        blobs = model.parse_params(params)

        mask_t0 = model.render_alpha_mask(blobs, timestep=0)
        mask_t100 = model.render_alpha_mask(blobs, timestep=100)

        # Peak column should shift right
        peak_col_t0 = np.argmax(mask_t0.max(axis=0))
        peak_col_t100 = np.argmax(mask_t100.max(axis=0))
        assert peak_col_t100 > peak_col_t0

    def test_blob_grows_with_positive_growth(self, image_shape):
        cfg = BlobConfig(num_blobs=1, max_opacity=0.6, sigma_range=(0.02, 0.30),
                         growth_rate_range=(-0.002, 0.01))
        model = DynamicBlobDustModel(cfg, image_shape, budget_level=1.0)
        # Blob with positive growth rate
        params = np.array([0.5, 0.5, 0.05, 0.05, 0.5, 0.0, 0.0, 0.005])
        blobs = model.parse_params(params)

        mask_t0 = model.render_alpha_mask(blobs, timestep=0)
        mask_t50 = model.render_alpha_mask(blobs, timestep=50)

        # Wider blob should have more non-zero area
        area_t0 = (mask_t0 > 0.01).sum()
        area_t50 = (mask_t50 > 0.01).sum()
        assert area_t50 > area_t0

    def test_static_blob_unchanged(self, image_shape):
        cfg = BlobConfig(num_blobs=1, max_opacity=0.6, sigma_range=(0.02, 0.30))
        model = DynamicBlobDustModel(cfg, image_shape, budget_level=1.0)
        # Static blob: zero velocity and growth
        params = np.array([0.5, 0.5, 0.10, 0.10, 0.5, 0.0, 0.0, 0.0])
        blobs = model.parse_params(params)

        mask_t0 = model.render_alpha_mask(blobs, timestep=0)
        mask_t100 = model.render_alpha_mask(blobs, timestep=100)
        np.testing.assert_array_almost_equal(mask_t0, mask_t100)


class TestBudgetProjection:
    def test_zero_budget(self, config, image_shape):
        model = DynamicBlobDustModel(config, image_shape, budget_level=0.0)
        params = model.get_cma_x0()
        blobs = model.parse_params(params)
        mask = model.project_to_budget(blobs)
        coverage = model.compute_coverage(mask)
        assert coverage == pytest.approx(0.0, abs=0.01)

    def test_small_budget_respected(self, config, image_shape):
        model = DynamicBlobDustModel(config, image_shape, budget_level=0.1)
        # High-opacity blobs
        params = model.get_cma_x0()
        for i in range(config.num_blobs):
            params[i * PARAMS_PER_BLOB + 4] = config.max_opacity
        blobs = model.parse_params(params)
        mask = model.project_to_budget(blobs)
        coverage = model.compute_coverage(mask)
        assert coverage <= 0.1 + 0.02

    def test_under_budget_unchanged(self, config, image_shape):
        model = DynamicBlobDustModel(config, image_shape, budget_level=1.0)
        # Very low opacity blobs
        params = model.get_cma_x0()
        for i in range(config.num_blobs):
            params[i * PARAMS_PER_BLOB + 4] = 0.01
        blobs = model.parse_params(params)
        mask_direct = model.render_alpha_mask(blobs)
        mask_projected = model.project_to_budget(blobs)
        np.testing.assert_array_almost_equal(mask_direct, mask_projected)

    def test_budget_at_different_timesteps(self, config, image_shape):
        model = DynamicBlobDustModel(config, image_shape, budget_level=0.15)
        params = model.get_cma_x0()
        for i in range(config.num_blobs):
            params[i * PARAMS_PER_BLOB + 4] = config.max_opacity
            params[i * PARAMS_PER_BLOB + 7] = 0.003  # growth
        blobs = model.parse_params(params)

        for t in [0, 50, 100]:
            mask = model.project_to_budget(blobs, timestep=t)
            coverage = model.compute_coverage(mask)
            assert coverage <= 0.15 + 0.02, f"Budget exceeded at t={t}: {coverage}"


class TestApply:
    def test_output_shape_uint8(self, model, sample_image):
        params = model.get_cma_x0()
        result = model.apply(sample_image, params)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_output_shape_float(self, model, image_shape):
        img_float = np.random.uniform(0, 1, size=image_shape).astype(np.float32)
        params = model.get_cma_x0()
        result = model.apply(img_float, params)
        assert result.shape == image_shape
        assert result.dtype == np.float32
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_timestep_changes_output(self, image_shape):
        cfg = BlobConfig(num_blobs=1, max_opacity=0.6,
                         velocity_range=(-0.01, 0.01))
        model = DynamicBlobDustModel(cfg, image_shape, budget_level=1.0)
        img = np.full(image_shape, 128, dtype=np.uint8)
        params = np.array([0.3, 0.5, 0.10, 0.10, 0.5, 0.005, 0.0, 0.0])
        r0 = model.apply(img, params, timestep=0)
        r100 = model.apply(img, params, timestep=100)
        assert not np.array_equal(r0, r100)

    def test_zero_opacity_no_change(self, config, image_shape, sample_image):
        model = DynamicBlobDustModel(config, image_shape, budget_level=0.5)
        params = model.get_cma_x0()
        for i in range(config.num_blobs):
            params[i * PARAMS_PER_BLOB + 4] = 0.0
        result = model.apply(sample_image, params)
        np.testing.assert_array_equal(result, sample_image)


class TestCMABounds:
    def test_bounds_length(self, model):
        lb, ub = model.get_cma_bounds()
        assert len(lb) == model.n_params
        assert len(ub) == model.n_params

    def test_bounds_valid(self, model):
        lb, ub = model.get_cma_bounds()
        for l, u in zip(lb, ub):
            assert l <= u

    def test_x0_within_bounds(self, model):
        lb, ub = model.get_cma_bounds()
        x0 = model.get_cma_x0()
        assert len(x0) == model.n_params
        for val, l, u in zip(x0, lb, ub):
            assert l <= val <= u

    def test_n_params(self, config, image_shape):
        model = DynamicBlobDustModel(config, image_shape, budget_level=0.2)
        assert model.n_params == config.num_blobs * PARAMS_PER_BLOB
        assert model.n_params == 40


class TestComputeCoverage:
    def test_zero_coverage(self, model):
        alpha_mask = np.zeros((256, 256))
        assert model.compute_coverage(alpha_mask) == 0.0

    def test_full_coverage(self, model):
        alpha_mask = np.ones((256, 256))
        assert model.compute_coverage(alpha_mask) == 1.0

    def test_partial_coverage(self, model):
        alpha_mask = np.zeros((256, 256))
        alpha_mask[:128, :] = 0.1  # above threshold
        coverage = model.compute_coverage(alpha_mask)
        assert coverage == pytest.approx(0.5, abs=0.01)


class TestGetRandomParams:
    def test_shape(self, model):
        rng = np.random.default_rng(42)
        params = model.get_random_params(rng)
        assert params.shape == (model.n_params,)

    def test_within_bounds(self, model):
        rng = np.random.default_rng(42)
        params = model.get_random_params(rng)
        lb, ub = model.get_cma_bounds()
        for val, l, u in zip(params, lb, ub):
            assert l <= val <= u

    def test_different_seeds_different_results(self, model):
        p1 = model.get_random_params(np.random.default_rng(1))
        p2 = model.get_random_params(np.random.default_rng(2))
        assert not np.allclose(p1, p2)
