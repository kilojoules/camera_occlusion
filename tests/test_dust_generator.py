"""Unit tests for the DustGenerator (no GPU needed)."""

import numpy as np
import pytest
import torch

from adversarial_dust.config import GeneratorConfig
from adversarial_dust.dust_generator import (
    BlobPrior,
    DustGenerator,
    ImageEncoder,
    MaskDecoder,
)

IMAGE_SHAPE = (512, 640, 3)


@pytest.fixture
def config():
    return GeneratorConfig(
        latent_dim=8,
        encoder_channels=[4, 8, 16],
        decoder_channels=[16, 8, 4],
        intermediate_resolution=(8, 8),
        max_opacity=0.6,
        num_blobs=3,
    )


@pytest.fixture
def generator(config):
    return DustGenerator(config, IMAGE_SHAPE)


@pytest.fixture
def dummy_image():
    return torch.rand(1, 3, IMAGE_SHAPE[0], IMAGE_SHAPE[1])


class TestBlobPrior:
    def test_output_shape(self):
        prior = BlobPrior(num_blobs=3, resolution=(8, 8))
        out = prior(batch_size=2)
        assert out.shape == (2, 1, 8, 8)

    def test_output_range(self):
        prior = BlobPrior(num_blobs=5, resolution=(16, 16))
        out = prior(batch_size=1)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_different_batch_sizes(self):
        prior = BlobPrior(num_blobs=3, resolution=(8, 8))
        for bs in [1, 4, 8]:
            out = prior(batch_size=bs)
            assert out.shape[0] == bs


class TestImageEncoder:
    def test_output_shape(self):
        enc = ImageEncoder(channels=[4, 8, 16], intermediate_resolution=(8, 8))
        img = torch.rand(2, 3, 512, 640)
        out = enc(img)
        assert out.shape == (2, 16, 8, 8)

    def test_output_channels(self):
        enc = ImageEncoder(channels=[4, 8], intermediate_resolution=(16, 16))
        assert enc.out_channels == 8


class TestMaskDecoder:
    def test_output_shape(self):
        dec = MaskDecoder(in_channels=25, channels=[16, 8, 4])
        x = torch.rand(2, 25, 8, 8)
        out = dec(x)
        assert out.shape == (2, 1, 8, 8)

    def test_output_range(self):
        dec = MaskDecoder(in_channels=10, channels=[8, 4])
        x = torch.rand(3, 10, 8, 8)
        out = dec(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestDustGenerator:
    def test_output_shape(self, generator, dummy_image):
        z = generator.sample_z(1)
        mask = generator(dummy_image, z)
        assert mask.shape == (1, 1, IMAGE_SHAPE[0], IMAGE_SHAPE[1])

    def test_output_range(self, generator, dummy_image):
        z = generator.sample_z(1)
        mask = generator(dummy_image, z)
        assert mask.min() >= 0.0
        assert mask.max() <= generator.max_opacity + 1e-6

    def test_batch_processing(self, generator):
        B = 3
        images = torch.rand(B, 3, IMAGE_SHAPE[0], IMAGE_SHAPE[1])
        z = generator.sample_z(B)
        mask = generator(images, z)
        assert mask.shape == (B, 1, IMAGE_SHAPE[0], IMAGE_SHAPE[1])

    def test_different_z_different_output(self, generator, dummy_image):
        z1 = torch.randn(1, generator.latent_dim)
        z2 = torch.randn(1, generator.latent_dim)
        mask1 = generator(dummy_image, z1)
        mask2 = generator(dummy_image, z2)
        # Different z should produce different masks (with overwhelming probability)
        assert not torch.allclose(mask1, mask2, atol=1e-4)

    def test_generate_mask_auto_z(self, generator, dummy_image):
        mask = generator.generate_mask(dummy_image)
        assert mask.shape == (1, 1, IMAGE_SHAPE[0], IMAGE_SHAPE[1])

    def test_sample_z(self, generator):
        z = generator.sample_z(5)
        assert z.shape == (5, generator.latent_dim)

    def test_mask_to_numpy(self, generator, dummy_image):
        mask = generator.generate_mask(dummy_image)
        arr = generator.mask_to_numpy(mask)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (1, IMAGE_SHAPE[0], IMAGE_SHAPE[1])

    def test_sample_with_exploration(self, generator, dummy_image):
        mean_masks, sampled_masks, log_probs, noisy_params = generator.sample_with_exploration(
            dummy_image, sigma=0.05, n_samples=4
        )
        assert mean_masks.shape == (4, 1, IMAGE_SHAPE[0], IMAGE_SHAPE[1])
        assert sampled_masks.shape == (4, 1, IMAGE_SHAPE[0], IMAGE_SHAPE[1])
        assert log_probs.shape == (4,)
        assert noisy_params is None  # DustGenerator has no blob params
        # Sampled masks should differ from mean (with exploration noise)
        assert not torch.allclose(mean_masks, sampled_masks, atol=1e-6)

    def test_sampled_masks_in_range(self, generator, dummy_image):
        _, sampled, _, _ = generator.sample_with_exploration(
            dummy_image, sigma=0.05, n_samples=4
        )
        assert sampled.min() >= 0.0
        assert sampled.max() <= generator.max_opacity + 1e-6

    def test_param_count_reasonable(self, generator):
        n_params = sum(p.numel() for p in generator.parameters())
        # Should be small (plan says ~40K)
        assert n_params < 200_000, f"Too many params: {n_params}"
