"""Neural Gaussian blob generator: image-conditioned blob parameter predictor.

The generator predicts parameters for K_max Gaussian blobs; a differentiable
renderer produces the smooth alpha mask.  Budget is expressed as number of
blobs (1, 2, 4) rather than pixel coverage fraction.

When ``temporal=True``, each blob has 7 parameters including temporal gating:
    (cx, cy, sigma_x, sigma_y, opacity, t_center, t_width)
The adversary has an **area-time budget** — total area × time must stay within
a budget constraint. Big blobs for a short time, or small blobs for a long
time — any combination allowed.

Architecture:
    image (B,3,H,W)
      -> ImageEncoder (stride-2 convs, reused from dust_generator)
      -> Global Average Pool -> (B, C)
      -> MLP head -> (B, K_max, 5 or 7) blob params
      -> constrain: sigmoid for cx,cy,opacity; scaled sigmoid for sigma_x,sigma_y
      -> render_mask: Gaussian formula per blob, combine with max -> (B,1,H,W)

Each blob (spatial): (cx, cy, sigma_x, sigma_y, opacity) -- 5 params.
Each blob (temporal): (cx, cy, sigma_x, sigma_y, opacity, t_center, t_width) -- 7 params.
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adversarial_dust.config import BlobGeneratorConfig
from adversarial_dust.dust_generator import ImageEncoder


class BlobGenerator(nn.Module):
    """Image-conditioned Gaussian blob dust generator.

    Predicts K_max blob parameters from the image and renders them as a smooth
    alpha mask via differentiable Gaussian rendering.  Budget is controlled by
    truncating to the first K blobs (the network learns to place the most
    impactful blob first).
    """

    def __init__(
        self,
        config: BlobGeneratorConfig,
        image_shape: Tuple[int, int, int],
    ):
        super().__init__()
        self.config = config
        self.image_h, self.image_w = image_shape[:2]
        self.k_max = config.k_max
        self.max_opacity = config.max_opacity
        self.sigma_min = config.sigma_min
        self.sigma_max = config.sigma_max
        self.temporal = config.temporal
        self.max_t_width = config.max_t_width
        self.temporal_sharpness = config.temporal_sharpness

        # 7 params per blob in temporal mode, 5 otherwise
        self.params_per_blob = 7 if self.temporal else 5

        # Reuse ImageEncoder from dust_generator -- but we only need GAP output,
        # so use a small intermediate resolution (doesn't matter, we pool it).
        self.encoder = ImageEncoder(
            channels=config.encoder_channels,
            intermediate_resolution=(8, 8),  # arbitrary, gets pooled away
        )
        enc_out = config.encoder_channels[-1]

        # MLP: GAP features -> K_max * params_per_blob blob parameters
        self.mlp = nn.Sequential(
            nn.Linear(enc_out, config.mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(config.mlp_hidden, config.k_max * self.params_per_blob),
        )

        # Pre-register coordinate grids (created on first forward, cached)
        self._grid_h = 0
        self._grid_w = 0

    def _ensure_grid(self, H: int, W: int, device: torch.device):
        """Lazily create coordinate grids matching the target resolution."""
        if self._grid_h == H and self._grid_w == W:
            gx = self._gx.to(device)
            gy = self._gy.to(device)
            return gx, gy

        gy = torch.linspace(0, 1, H, device=device)
        gx = torch.linspace(0, 1, W, device=device)
        gy, gx = torch.meshgrid(gy, gx, indexing="ij")  # (H, W)
        # Register as buffers so they move with .to(device)
        self.register_buffer("_gx", gx, persistent=False)
        self.register_buffer("_gy", gy, persistent=False)
        self._grid_h = H
        self._grid_w = W
        return gx, gy

    def _predict_blob_params(self, image: torch.Tensor) -> torch.Tensor:
        """Predict raw blob parameters from image.

        Args:
            image: (B, 3, H, W) float in [0, 1].

        Returns:
            (B, K_max, P) constrained blob params where P=5 or P=7:
                [:, :, 0:2] = cx, cy in [0, 1]
                [:, :, 2:4] = sigma_x, sigma_y in [sigma_min, sigma_max]
                [:, :, 4]   = opacity in [0, max_opacity]
            If temporal:
                [:, :, 5]   = t_center in [0, 1]
                [:, :, 6]   = t_width in [0, max_t_width]
        """
        feat = self.encoder(image)  # (B, C, ih, iw)
        pooled = feat.mean(dim=(2, 3))  # (B, C) -- global average pool
        raw = self.mlp(pooled)  # (B, K_max * params_per_blob)
        raw = raw.view(-1, self.k_max, self.params_per_blob)

        # Constrain spatial parameters
        cx = torch.sigmoid(raw[:, :, 0])  # [0, 1]
        cy = torch.sigmoid(raw[:, :, 1])  # [0, 1]
        sx = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(raw[:, :, 2])
        sy = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(raw[:, :, 3])
        opacity = self.max_opacity * torch.sigmoid(raw[:, :, 4])

        if self.temporal:
            t_center = torch.sigmoid(raw[:, :, 5])  # [0, 1]
            t_width = self.max_t_width * torch.sigmoid(raw[:, :, 6])  # [0, max_t_width]
            return torch.stack([cx, cy, sx, sy, opacity, t_center, t_width], dim=-1)

        return torch.stack([cx, cy, sx, sy, opacity], dim=-1)

    def render_mask(
        self,
        blob_params: torch.Tensor,
        H: int,
        W: int,
        k: Optional[int] = None,
    ) -> torch.Tensor:
        """Render blob parameters into a smooth alpha mask.

        Args:
            blob_params: (B, K_max, 5) constrained blob params.
            H: Output height.
            W: Output width.
            k: Number of blobs to use (truncate to first k). None = all.

        Returns:
            (B, 1, H, W) alpha mask in [0, max_opacity].
        """
        B = blob_params.shape[0]
        device = blob_params.device

        if k is not None:
            blob_params = blob_params[:, :k, :]

        K = blob_params.shape[1]
        gx, gy = self._ensure_grid(H, W, device)

        # Expand grids: (1, 1, H, W)
        gx = gx.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        gy = gy.unsqueeze(0).unsqueeze(0)

        # Extract and expand params: (B, K, 1, 1)
        cx = blob_params[:, :, 0].unsqueeze(-1).unsqueeze(-1)
        cy = blob_params[:, :, 1].unsqueeze(-1).unsqueeze(-1)
        sx = blob_params[:, :, 2].unsqueeze(-1).unsqueeze(-1)
        sy = blob_params[:, :, 3].unsqueeze(-1).unsqueeze(-1)
        opacity = blob_params[:, :, 4].unsqueeze(-1).unsqueeze(-1)

        # Gaussian: (B, K, H, W)
        # Note: gx corresponds to x (width), gy to y (height)
        # cx, cy are in [0,1] normalized coords; gx is width-axis, gy is height-axis
        gauss = opacity * torch.exp(
            -0.5 * (((gx - cx) / sx) ** 2 + ((gy - cy) / sy) ** 2)
        )

        # Combine blobs via element-wise max
        if K > 0:
            mask = gauss.max(dim=1, keepdim=True).values  # (B, 1, H, W)
        else:
            mask = torch.zeros(B, 1, H, W, device=device)

        return mask

    def temporal_gate(
        self,
        blob_params: torch.Tensor,
        t_normalized: float,
    ) -> torch.Tensor:
        """Compute temporal gate values for each blob at a given timestep.

        Uses a smooth rectangular window via product of two sigmoids:
            gate(t) = sigmoid(s*(t - (tc - tw/2))) * sigmoid(s*((tc + tw/2) - t))

        Args:
            blob_params: (B, K, 7) blob params with temporal columns.
            t_normalized: Normalized timestep in [0, 1].

        Returns:
            (B, K) gate values in [0, 1].
        """
        t_center = blob_params[:, :, 5]  # (B, K)
        t_width = blob_params[:, :, 6]   # (B, K)
        s = self.temporal_sharpness

        t = torch.tensor(t_normalized, dtype=blob_params.dtype, device=blob_params.device)
        left = torch.sigmoid(s * (t - (t_center - t_width / 2)))
        right = torch.sigmoid(s * ((t_center + t_width / 2) - t))
        return left * right  # (B, K)

    def render_mask_at_time(
        self,
        blob_params: torch.Tensor,
        H: int,
        W: int,
        t_normalized: float,
        k: Optional[int] = None,
    ) -> torch.Tensor:
        """Render blob mask at a specific timestep, modulating opacity by temporal gate.

        Args:
            blob_params: (B, K_max, 7) temporal blob params.
            H: Output height.
            W: Output width.
            t_normalized: Normalized timestep in [0, 1].
            k: Number of blobs to use. None = all.

        Returns:
            (B, 1, H, W) alpha mask.
        """
        if k is not None:
            blob_params = blob_params[:, :k, :]

        gate = self.temporal_gate(blob_params, t_normalized)  # (B, K)

        # Create a copy with opacity modulated by gate
        gated_params = blob_params.clone()
        gated_params[:, :, 4] = blob_params[:, :, 4] * gate

        # Delegate to spatial render_mask (which reads only columns 0:5)
        return self.render_mask(gated_params[:, :, :5], H, W)

    def compute_area_time(
        self,
        blob_params: torch.Tensor,
        H: int,
        W: int,
        T: int,
        k: Optional[int] = None,
        n_time_samples: int = 16,
    ) -> torch.Tensor:
        """Approximate area-time budget usage via subsampled timesteps.

        area_time = (1/T) * sum_t mean(mask_t) over all pixels

        Args:
            blob_params: (B, K_max, 7) temporal blob params.
            H: Render height.
            W: Render width.
            T: Total number of episode timesteps.
            k: Number of blobs to use. None = all.
            n_time_samples: Number of timesteps to subsample.

        Returns:
            (B,) area-time values.
        """
        B = blob_params.shape[0]
        device = blob_params.device

        # Subsample uniformly in [0, 1]
        time_points = torch.linspace(0, 1, n_time_samples, device=device)

        area_sum = torch.zeros(B, device=device)
        for t_val in time_points:
            mask_t = self.render_mask_at_time(
                blob_params, H, W, t_val.item(), k=k
            )  # (B, 1, H, W)
            area_sum += mask_t.mean(dim=(1, 2, 3))  # mean over spatial dims

        return area_sum / n_time_samples

    def forward(
        self,
        image: torch.Tensor,
        k: Optional[int] = None,
    ) -> torch.Tensor:
        """Produce a full-resolution alpha mask.

        Args:
            image: (B, 3, H, W) float in [0, 1].
            k: Number of blobs to render. None = k_max.

        Returns:
            (B, 1, H, W) alpha mask in [0, max_opacity].
        """
        blob_params = self._predict_blob_params(image)
        return self.render_mask(blob_params, self.image_h, self.image_w, k=k)

    def generate_mask(
        self,
        image: torch.Tensor,
        k: Optional[int] = None,
    ) -> torch.Tensor:
        """Convenience alias for forward."""
        return self.forward(image, k=k)

    def sample_with_exploration(
        self,
        image: torch.Tensor,
        sigma: float,
        n_samples: int,
        k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample masks with exploration noise for REINFORCE.

        Noise is added in the **blob parameter space** (PK values) instead of
        per-pixel, giving a 16,000x smaller action space and much lower variance.

        Args:
            image: (1, 3, H, W) single image (will be repeated).
            sigma: Exploration noise std in constrained param space.
            n_samples: Number of mask samples per step.
            k: Number of blobs to render. None = k_max.

        Returns:
            mean_masks: (n_samples, 1, H, W) deterministic generator output.
            sampled_masks: (n_samples, 1, H, W) masks from noisy params.
            log_probs: (n_samples,) log probability of each sample.
            noisy_params: (n_samples, K_max, P) noisy blob params (P=5 or 7).
        """
        device = image.device
        image_batch = image.expand(n_samples, -1, -1, -1)

        # Predict blob params (deterministic)
        mean_params = self._predict_blob_params(image_batch)  # (n, K_max, P)

        # Add exploration noise in constrained space
        noise = torch.randn_like(mean_params) * sigma
        noisy_params = mean_params + noise

        # Clamp to valid ranges
        noisy_params = noisy_params.clone()
        noisy_params[:, :, 0:2] = noisy_params[:, :, 0:2].clamp(0.0, 1.0)
        noisy_params[:, :, 2:4] = noisy_params[:, :, 2:4].clamp(self.sigma_min, self.sigma_max)
        noisy_params[:, :, 4] = noisy_params[:, :, 4].clamp(0.0, self.max_opacity)
        if self.temporal:
            noisy_params[:, :, 5] = noisy_params[:, :, 5].clamp(0.0, 1.0)
            noisy_params[:, :, 6] = noisy_params[:, :, 6].clamp(0.0, self.max_t_width)

        # Render masks (static snapshot at t=0.5 for mean/sampled masks)
        if self.temporal:
            mean_masks = self.render_mask_at_time(
                mean_params, self.image_h, self.image_w, 0.5, k=k
            )
            sampled_masks = self.render_mask_at_time(
                noisy_params, self.image_h, self.image_w, 0.5, k=k
            )
        else:
            mean_masks = self.render_mask(mean_params, self.image_h, self.image_w, k=k)
            sampled_masks = self.render_mask(noisy_params, self.image_h, self.image_w, k=k)

        # Log prob under isotropic Gaussian over the blob params (score function).
        # Use only the active k blobs for the log prob computation.
        active_k = k if k is not None else self.k_max
        active_mean = mean_params[:, :active_k, :]
        active_noisy = noisy_params[:, :active_k, :].detach()
        diff = active_noisy - active_mean
        n_params = active_mean[:, :, :].numel() // n_samples
        log_probs = -0.5 * (diff / sigma).pow(2).sum(dim=(1, 2))
        log_probs = log_probs - 0.5 * n_params * math.log(2 * math.pi * sigma ** 2)

        return mean_masks, sampled_masks, log_probs, noisy_params

    def mask_to_numpy(self, mask: torch.Tensor) -> np.ndarray:
        """Convert (B, 1, H, W) torch mask to (B, H, W) numpy array."""
        return np.array(mask.squeeze(1).detach().cpu().tolist(), dtype=np.float32)
