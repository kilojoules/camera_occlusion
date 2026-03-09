"""Unit tests for BlobGenerator (no GPU needed)."""

import numpy as np
import pytest
import torch

from adversarial_dust.config import BlobGeneratorConfig
from adversarial_dust.blob_generator import BlobGenerator

IMAGE_SHAPE = (512, 640, 3)
SMALL_IMAGE_SHAPE = (64, 80, 3)


@pytest.fixture
def config():
    return BlobGeneratorConfig(
        k_max=4,
        encoder_channels=[4, 8, 16],
        mlp_hidden=32,
        sigma_min=0.02,
        sigma_max=0.30,
        max_opacity=0.6,
    )


@pytest.fixture
def small_config():
    return BlobGeneratorConfig(
        k_max=4,
        encoder_channels=[4, 8],
        mlp_hidden=16,
        sigma_min=0.02,
        sigma_max=0.30,
        max_opacity=0.6,
    )


@pytest.fixture
def generator(small_config):
    return BlobGenerator(small_config, SMALL_IMAGE_SHAPE)


@pytest.fixture
def dummy_image():
    return torch.rand(1, 3, SMALL_IMAGE_SHAPE[0], SMALL_IMAGE_SHAPE[1])


class TestBlobGeneratorOutput:
    def test_output_shape(self, generator, dummy_image):
        mask = generator(dummy_image)
        assert mask.shape == (1, 1, SMALL_IMAGE_SHAPE[0], SMALL_IMAGE_SHAPE[1])

    def test_output_range(self, generator, dummy_image):
        mask = generator(dummy_image)
        assert mask.min() >= 0.0
        assert mask.max() <= generator.max_opacity + 1e-6

    def test_batch_processing(self, generator):
        B = 3
        images = torch.rand(B, 3, SMALL_IMAGE_SHAPE[0], SMALL_IMAGE_SHAPE[1])
        mask = generator(images)
        assert mask.shape == (B, 1, SMALL_IMAGE_SHAPE[0], SMALL_IMAGE_SHAPE[1])

    def test_generate_mask_alias(self, generator, dummy_image):
        mask = generator.generate_mask(dummy_image)
        assert mask.shape == (1, 1, SMALL_IMAGE_SHAPE[0], SMALL_IMAGE_SHAPE[1])

    def test_mask_to_numpy(self, generator, dummy_image):
        mask = generator.generate_mask(dummy_image)
        arr = generator.mask_to_numpy(mask)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (1, SMALL_IMAGE_SHAPE[0], SMALL_IMAGE_SHAPE[1])
        assert arr.dtype == np.float32


class TestBlobTruncation:
    def test_truncated_k_reduces_blobs(self, generator, dummy_image):
        mask_full = generator(dummy_image, k=4)
        mask_half = generator(dummy_image, k=2)
        mask_one = generator(dummy_image, k=1)
        # All should have correct shape
        assert mask_full.shape == mask_half.shape == mask_one.shape

    def test_fewer_blobs_less_or_equal_coverage(self, generator, dummy_image):
        # With fewer blobs, the mask should generally have less or equal coverage
        # (max combination means adding blobs can only increase the mask)
        mask_one = generator(dummy_image, k=1)
        mask_all = generator(dummy_image, k=generator.k_max)
        # Every pixel in 1-blob mask should be <= every pixel in all-blob mask
        # (since max over more blobs >= max over fewer blobs)
        assert (mask_one <= mask_all + 1e-6).all()

    def test_k_none_uses_all_blobs(self, generator, dummy_image):
        mask_none = generator(dummy_image, k=None)
        mask_kmax = generator(dummy_image, k=generator.k_max)
        assert torch.allclose(mask_none, mask_kmax, atol=1e-6)


class TestBlobRendering:
    def test_render_mask_shapes(self, generator):
        B, K = 2, 3
        params = torch.rand(B, K, 5)
        # Constrain manually for test
        params[:, :, 0:2] = torch.sigmoid(params[:, :, 0:2])
        params[:, :, 2:4] = 0.02 + 0.28 * torch.sigmoid(params[:, :, 2:4])
        params[:, :, 4] = 0.6 * torch.sigmoid(params[:, :, 4])
        mask = generator.render_mask(params, 64, 80, k=2)
        assert mask.shape == (B, 1, 64, 80)

    def test_render_mask_zero_blobs(self, generator):
        params = torch.rand(2, 4, 5)
        mask = generator.render_mask(params, 64, 80, k=0)
        assert mask.shape == (2, 1, 64, 80)
        assert mask.sum() == 0.0

    def test_blob_center_has_peak(self, generator):
        """A blob centered at (0.5, 0.5) should have its peak near the image center."""
        B = 1
        params = torch.zeros(B, 1, 5)
        params[0, 0, 0] = 0.5  # cx
        params[0, 0, 1] = 0.5  # cy
        params[0, 0, 2] = 0.05  # sigma_x
        params[0, 0, 3] = 0.05  # sigma_y
        params[0, 0, 4] = 0.6  # opacity
        mask = generator.render_mask(params, 64, 80, k=1)
        # Peak should be near center
        peak_y, peak_x = torch.where(mask[0, 0] == mask[0, 0].max())
        assert abs(peak_y[0].item() - 32) < 5
        assert abs(peak_x[0].item() - 40) < 5


class TestExploration:
    def test_sample_with_exploration_shapes(self, generator, dummy_image):
        mean_masks, sampled_masks, log_probs, noisy_params = generator.sample_with_exploration(
            dummy_image, sigma=0.02, n_samples=4
        )
        H, W = SMALL_IMAGE_SHAPE[:2]
        assert mean_masks.shape == (4, 1, H, W)
        assert sampled_masks.shape == (4, 1, H, W)
        assert log_probs.shape == (4,)
        assert noisy_params.shape == (4, generator.k_max, 5)

    def test_sampled_differ_from_mean(self, generator, dummy_image):
        mean_masks, sampled_masks, _, _ = generator.sample_with_exploration(
            dummy_image, sigma=0.02, n_samples=4
        )
        # With noise, sampled masks should differ from mean
        assert not torch.allclose(mean_masks, sampled_masks, atol=1e-6)

    def test_sampled_masks_in_range(self, generator, dummy_image):
        _, sampled, _, _ = generator.sample_with_exploration(
            dummy_image, sigma=0.02, n_samples=4
        )
        assert sampled.min() >= 0.0
        assert sampled.max() <= generator.max_opacity + 1e-6

    def test_exploration_with_truncated_k(self, generator, dummy_image):
        mean_masks, sampled_masks, log_probs, noisy_params = generator.sample_with_exploration(
            dummy_image, sigma=0.02, n_samples=4, k=2
        )
        H, W = SMALL_IMAGE_SHAPE[:2]
        assert mean_masks.shape == (4, 1, H, W)
        assert sampled_masks.shape == (4, 1, H, W)
        assert log_probs.shape == (4,)
        assert noisy_params.shape == (4, generator.k_max, 5)

    def test_log_probs_are_finite(self, generator, dummy_image):
        _, _, log_probs, _ = generator.sample_with_exploration(
            dummy_image, sigma=0.02, n_samples=4
        )
        assert torch.isfinite(log_probs).all()


class TestParamCount:
    def test_param_count_under_40k(self):
        """Plan specifies ~29K params with [16,32,64] encoder and mlp_hidden=64."""
        config = BlobGeneratorConfig(
            k_max=4,
            encoder_channels=[16, 32, 64],
            mlp_hidden=64,
        )
        gen = BlobGenerator(config, IMAGE_SHAPE)
        n_params = sum(p.numel() for p in gen.parameters())
        assert n_params < 40_000, f"Too many params: {n_params}"

    def test_param_count_reasonable_small(self):
        config = BlobGeneratorConfig(
            k_max=4,
            encoder_channels=[4, 8],
            mlp_hidden=16,
        )
        gen = BlobGenerator(config, SMALL_IMAGE_SHAPE)
        n_params = sum(p.numel() for p in gen.parameters())
        assert n_params < 10_000, f"Too many params: {n_params}"


# ---------------------------------------------------------------------------
# Temporal blob tests
# ---------------------------------------------------------------------------


@pytest.fixture
def temporal_config():
    return BlobGeneratorConfig(
        k_max=4,
        encoder_channels=[4, 8],
        mlp_hidden=16,
        sigma_min=0.02,
        sigma_max=0.30,
        max_opacity=0.6,
        temporal=True,
        max_t_width=1.0,
        temporal_sharpness=20.0,
    )


@pytest.fixture
def temporal_generator(temporal_config):
    return BlobGenerator(temporal_config, SMALL_IMAGE_SHAPE)


class TestTemporalBlobs:
    def test_output_7_params(self, temporal_generator, dummy_image):
        """Temporal generator should predict 7 params per blob."""
        params = temporal_generator._predict_blob_params(dummy_image)
        assert params.shape == (1, temporal_generator.k_max, 7)

    def test_nontemporal_still_5_params(self, generator, dummy_image):
        """Non-temporal generator should still produce 5 params (backward compat)."""
        params = generator._predict_blob_params(dummy_image)
        assert params.shape == (1, generator.k_max, 5)

    def test_temporal_param_ranges(self, temporal_generator, dummy_image):
        """t_center in [0,1], t_width in [0, max_t_width]."""
        params = temporal_generator._predict_blob_params(dummy_image)
        t_center = params[:, :, 5]
        t_width = params[:, :, 6]
        assert t_center.min() >= 0.0
        assert t_center.max() <= 1.0
        assert t_width.min() >= 0.0
        assert t_width.max() <= temporal_generator.max_t_width + 1e-6

    def test_temporal_gate_inside_window(self, temporal_generator):
        """Gate should be ~1.0 when t is inside the blob's active window."""
        B, K = 1, 1
        params = torch.zeros(B, K, 7)
        params[0, 0, 5] = 0.5   # t_center
        params[0, 0, 6] = 0.4   # t_width -> active from 0.3 to 0.7
        gate = temporal_generator.temporal_gate(params, 0.5)
        assert gate.shape == (B, K)
        assert gate[0, 0].item() > 0.95  # should be ~1.0 at center

    def test_temporal_gate_outside_window(self, temporal_generator):
        """Gate should be ~0.0 when t is outside the blob's active window."""
        B, K = 1, 1
        params = torch.zeros(B, K, 7)
        params[0, 0, 5] = 0.5   # t_center
        params[0, 0, 6] = 0.2   # t_width -> active from 0.4 to 0.6
        gate_early = temporal_generator.temporal_gate(params, 0.0)
        gate_late = temporal_generator.temporal_gate(params, 1.0)
        assert gate_early[0, 0].item() < 0.05
        assert gate_late[0, 0].item() < 0.05

    def test_render_mask_at_time_shape(self, temporal_generator):
        """render_mask_at_time should return correct shape."""
        B, K = 2, 3
        params = torch.rand(B, K, 7)
        params[:, :, 0:2] = torch.sigmoid(params[:, :, 0:2])
        params[:, :, 2:4] = 0.02 + 0.28 * torch.sigmoid(params[:, :, 2:4])
        params[:, :, 4] = 0.6 * torch.sigmoid(params[:, :, 4])
        params[:, :, 5] = torch.sigmoid(params[:, :, 5])
        params[:, :, 6] = torch.sigmoid(params[:, :, 6])
        mask = temporal_generator.render_mask_at_time(params, 64, 80, 0.5)
        assert mask.shape == (B, 1, 64, 80)

    def test_render_mask_at_time_inactive_is_zero(self, temporal_generator):
        """When t is far outside window, mask should be near zero."""
        B, K = 1, 1
        params = torch.zeros(B, K, 7)
        params[0, 0, 0] = 0.5   # cx
        params[0, 0, 1] = 0.5   # cy
        params[0, 0, 2] = 0.1   # sigma_x
        params[0, 0, 3] = 0.1   # sigma_y
        params[0, 0, 4] = 0.6   # opacity
        params[0, 0, 5] = 0.5   # t_center
        params[0, 0, 6] = 0.1   # t_width -> active from 0.45 to 0.55
        mask = temporal_generator.render_mask_at_time(params, 32, 32, 0.0)
        assert mask.max().item() < 0.01

    def test_compute_area_time_positive(self, temporal_generator):
        """compute_area_time should be > 0 for active blobs."""
        B, K = 1, 2
        params = torch.zeros(B, K, 7)
        params[0, 0] = torch.tensor([0.5, 0.5, 0.1, 0.1, 0.6, 0.5, 0.5])
        params[0, 1] = torch.tensor([0.3, 0.3, 0.1, 0.1, 0.4, 0.2, 0.3])
        at = temporal_generator.compute_area_time(params, 32, 32, T=80)
        assert at.shape == (B,)
        assert at[0].item() > 0.0

    def test_sample_with_exploration_temporal_4tuple(self, temporal_generator, dummy_image):
        """Temporal sample_with_exploration should return 4-tuple with (n, K, 7) params."""
        mean_masks, sampled_masks, log_probs, noisy_params = (
            temporal_generator.sample_with_exploration(
                dummy_image, sigma=0.02, n_samples=4
            )
        )
        H, W = SMALL_IMAGE_SHAPE[:2]
        assert mean_masks.shape == (4, 1, H, W)
        assert sampled_masks.shape == (4, 1, H, W)
        assert log_probs.shape == (4,)
        assert noisy_params.shape == (4, temporal_generator.k_max, 7)

    def test_sample_temporal_params_in_range(self, temporal_generator, dummy_image):
        """Noisy temporal params should be clamped to valid ranges."""
        _, _, _, noisy_params = temporal_generator.sample_with_exploration(
            dummy_image, sigma=0.02, n_samples=4
        )
        assert noisy_params[:, :, 5].min() >= 0.0
        assert noisy_params[:, :, 5].max() <= 1.0
        assert noisy_params[:, :, 6].min() >= 0.0
        assert noisy_params[:, :, 6].max() <= temporal_generator.max_t_width + 1e-6
