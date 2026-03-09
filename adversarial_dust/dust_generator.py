"""PyTorch neural dust generator: image-conditioned mask producer.

The generator produces realistic camera-grime alpha masks conditioned on the
current camera image.  A BlobPrior biases toward circular blotch patterns.
The architecture is kept small (~40K params) so REINFORCE converges reliably.
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adversarial_dust.config import GeneratorConfig


class BlobPrior(nn.Module):
    """Learned Gaussian blob prior that biases masks toward circular grime.

    Maintains ``num_blobs`` learnable (cx, cy, sx, sy, amplitude) tuples and
    renders them as a soft prior map at the intermediate resolution.
    """

    def __init__(self, num_blobs: int, resolution: Tuple[int, int]):
        super().__init__()
        self.num_blobs = num_blobs
        self.h, self.w = resolution

        # Learnable blob parameters: center (cx, cy), spread (sx, sy), amp
        self.centers = nn.Parameter(torch.rand(num_blobs, 2))  # in [0,1]
        self.log_spreads = nn.Parameter(torch.zeros(num_blobs, 2))  # log(sigma)
        self.amplitudes = nn.Parameter(torch.ones(num_blobs) * 0.5)

    def forward(self, batch_size: int) -> torch.Tensor:
        """Render blob prior map.

        Returns:
            (B, 1, H, W) soft blob map in [0, ~1].
        """
        device = self.centers.device
        gy = torch.linspace(0, 1, self.h, device=device)
        gx = torch.linspace(0, 1, self.w, device=device)
        yy, xx = torch.meshgrid(gy, gx, indexing="ij")  # (H, W)

        prior = torch.zeros(self.h, self.w, device=device)
        for i in range(self.num_blobs):
            cx, cy = torch.sigmoid(self.centers[i])  # keep in [0,1]
            sx, sy = torch.exp(self.log_spreads[i]).clamp(min=0.01, max=1.0)
            amp = torch.sigmoid(self.amplitudes[i])
            gauss = amp * torch.exp(
                -0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)
            )
            prior = prior + gauss

        prior = prior.clamp(0.0, 1.0)
        return prior.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)


class ImageEncoder(nn.Module):
    """Lightweight CNN encoder that extracts a spatial feature from the image.

    Three stride-2 conv layers downsample the image and produce a feature map
    that is interpolated to the intermediate resolution.
    """

    def __init__(
        self,
        channels: List[int],
        intermediate_resolution: Tuple[int, int],
    ):
        super().__init__()
        self.intermediate_resolution = intermediate_resolution
        layers = []
        in_c = 3
        for out_c in channels:
            layers.append(nn.Conv2d(in_c, out_c, 3, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_c = out_c
        self.net = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to feature map at intermediate resolution.

        Args:
            image: (B, 3, H, W) float in [0, 1].

        Returns:
            (B, C, ih, iw) feature map.
        """
        feat = self.net(image)
        return F.interpolate(
            feat,
            size=self.intermediate_resolution,
            mode="bilinear",
            align_corners=False,
        )


class MaskDecoder(nn.Module):
    """Decodes concatenated features into a single-channel alpha mask.

    Input: encoder features + noise channel + blob prior channel.
    Output: (B, 1, ih, iw) raw sigmoid mask at intermediate resolution.
    """

    def __init__(
        self,
        in_channels: int,
        channels: List[int],
    ):
        super().__init__()
        layers = []
        in_c = in_channels
        for out_c in channels:
            layers.append(nn.Conv2d(in_c, out_c, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_c = out_c
        layers.append(nn.Conv2d(in_c, 1, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DustGenerator(nn.Module):
    """Image-conditioned dust mask generator.

    Pipeline:
        image → ImageEncoder → features (ih, iw)
        noise z → tile to (ih, iw)
        BlobPrior → prior map (ih, iw)
        concat [features, tiled_z, prior] → MaskDecoder → sigmoid mask (ih, iw)
        scale by max_opacity → bilinear upsample to (H, W) → budget projection

    The generator is meant to be trained with REINFORCE since we cannot
    backprop through the simulator.
    """

    def __init__(
        self,
        config: GeneratorConfig,
        image_shape: Tuple[int, int, int],
    ):
        super().__init__()
        self.config = config
        self.image_h, self.image_w = image_shape[:2]
        ih, iw = config.intermediate_resolution

        self.encoder = ImageEncoder(
            config.encoder_channels, config.intermediate_resolution
        )
        self.blob_prior = BlobPrior(config.num_blobs, config.intermediate_resolution)

        # Decoder input: encoder features + latent channels + blob prior
        decoder_in = config.encoder_channels[-1] + config.latent_dim + 1
        self.decoder = MaskDecoder(decoder_in, config.decoder_channels)

        self.max_opacity = config.max_opacity
        self.latent_dim = config.latent_dim
        self.intermediate_resolution = config.intermediate_resolution

    def forward(
        self,
        image: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Produce a full-resolution alpha mask.

        Args:
            image: (B, 3, H, W) float in [0, 1].
            z: (B, latent_dim) noise vector.

        Returns:
            (B, 1, image_H, image_W) alpha mask in [0, max_opacity].
        """
        B = image.shape[0]
        ih, iw = self.intermediate_resolution

        # Encode image
        feat = self.encoder(image)  # (B, C, ih, iw)

        # Tile latent z spatially
        z_tiled = z.unsqueeze(-1).unsqueeze(-1).expand(B, self.latent_dim, ih, iw)

        # Blob prior
        prior = self.blob_prior(B)  # (B, 1, ih, iw)

        # Concat and decode
        x = torch.cat([feat, z_tiled, prior], dim=1)
        mask_intermediate = self.decoder(x)  # (B, 1, ih, iw) in [0, 1]

        # Scale by max opacity
        mask_intermediate = mask_intermediate * self.max_opacity

        # Bilinear upsample to full resolution
        mask_full = F.interpolate(
            mask_intermediate,
            size=(self.image_h, self.image_w),
            mode="bilinear",
            align_corners=False,
        )

        return mask_full

    def sample_z(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Sample latent noise vectors."""
        if device is None:
            device = next(self.parameters()).device
        return torch.randn(batch_size, self.latent_dim, device=device)

    def generate_mask(
        self,
        image: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience: generate a mask, sampling z if not provided.

        Args:
            image: (B, 3, H, W).
            z: Optional (B, latent_dim). Sampled if None.

        Returns:
            (B, 1, H, W) alpha mask.
        """
        B = image.shape[0]
        if z is None:
            z = self.sample_z(B, device=image.device)
        return self.forward(image, z)

    def sample_with_exploration(
        self,
        image: torch.Tensor,
        sigma: float,
        n_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Sample masks with exploration noise for REINFORCE.

        For each of ``n_samples`` samples, generate the mean mask μ and add
        Gaussian exploration noise ε to get the sampled mask.

        Args:
            image: (1, 3, H, W) single image (will be repeated).
            sigma: Exploration noise std.
            n_samples: Number of mask samples.

        Returns:
            mean_masks: (n_samples, 1, H, W) deterministic generator output.
            sampled_masks: (n_samples, 1, H, W) masks with noise.
            log_probs: (n_samples,) log probability of each sample.
            noisy_params: Always None for DustGenerator (per-pixel, no blob params).
        """
        device = image.device
        image_batch = image.expand(n_samples, -1, -1, -1)

        z_batch = self.sample_z(n_samples, device=device)
        mean_masks = self.forward(image_batch, z_batch)  # (n, 1, H, W)

        # Add exploration noise
        noise = torch.randn_like(mean_masks) * sigma
        sampled_masks = (mean_masks + noise).clamp(0.0, self.max_opacity)

        # Log prob under isotropic Gaussian (score function for REINFORCE).
        # Detach sampled_masks so gradient flows only through mean_masks,
        # giving the correct score ∇_θ log π = (a - μ(θ)) / σ² · ∇_θ μ(θ).
        diff = sampled_masks.detach() - mean_masks
        log_probs = -0.5 * (diff / sigma).pow(2).sum(dim=(1, 2, 3))
        log_probs = log_probs - 0.5 * mean_masks[0].numel() * math.log(
            2 * math.pi * sigma ** 2
        )

        return mean_masks, sampled_masks, log_probs, None

    def mask_to_numpy(self, mask: torch.Tensor) -> np.ndarray:
        """Convert (B, 1, H, W) torch mask to (B, H, W) numpy array."""
        return np.array(mask.squeeze(1).detach().cpu().tolist(), dtype=np.float32)
