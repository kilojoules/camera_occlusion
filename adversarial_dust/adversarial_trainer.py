"""Zero-sum adversarial training: generator (attacker) vs Octo (defender).

MaskApplicator — alpha blending + budget projection for generator masks.
ReinforceTrainer — REINFORCE update for the generator (reward = 1 - success_rate).
AdversarialTrainer — orchestrates the alternating attacker/defender training loop.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.optim as optim

from adversarial_dust.config import (
    AdversarialExperimentConfig,
    BlobGeneratorConfig,
    GeneratorConfig,
)
from adversarial_dust.blob_generator import BlobGenerator
from adversarial_dust.dust_generator import DustGenerator

logger = logging.getLogger(__name__)


class MaskApplicator:
    """Applies a generator-produced alpha mask to images with budget projection.

    Handles the numpy-level alpha blending and budget constraint that mirrors
    ``AdversarialDustModel`` but works with masks produced by the generator.
    """

    def __init__(self, config: GeneratorConfig, image_shape: Tuple[int, int, int]):
        self.max_opacity = config.max_opacity
        self.dust_color = np.array(config.dust_color, dtype=np.float32) / 255.0
        self.dirty_threshold = config.dirty_threshold
        self.image_h, self.image_w = image_shape[:2]

    def compute_coverage(self, alpha_mask: np.ndarray) -> float:
        """Fraction of pixels with opacity above dirty threshold."""
        return float(np.mean(alpha_mask > self.dirty_threshold))

    def project_to_budget(
        self, alpha_mask: np.ndarray, budget_level: float
    ) -> np.ndarray:
        """Scale mask down so coverage <= budget_level via binary search."""
        if budget_level >= 1.0:
            return alpha_mask

        coverage = self.compute_coverage(alpha_mask)
        if coverage <= budget_level:
            return alpha_mask

        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2.0
            scaled = alpha_mask * mid
            if self.compute_coverage(scaled) <= budget_level:
                lo = mid
            else:
                hi = mid

        return alpha_mask * lo

    def project_area_time_budget(
        self,
        blob_params: torch.Tensor,
        budget: float,
        generator: "BlobGenerator",
        T: int = 80,
    ) -> torch.Tensor:
        """Scale t_width down so area-time <= budget via binary search.

        Args:
            blob_params: (B, K, 7) temporal blob params.
            budget: Maximum area-time fraction.
            generator: BlobGenerator instance (for compute_area_time).
            T: Episode length for area-time computation.

        Returns:
            (B, K, 7) projected blob params with scaled t_width.
        """
        H, W = self.image_h, self.image_w
        with torch.no_grad():
            area_time = generator.compute_area_time(blob_params, H, W, T)
        # Check if all samples are within budget
        if (area_time <= budget).all():
            return blob_params

        # Binary search on a uniform t_width scale factor
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2.0
            scaled = blob_params.clone()
            scaled[:, :, 6] = blob_params[:, :, 6] * mid
            with torch.no_grad():
                at = generator.compute_area_time(scaled, H, W, T)
            if (at <= budget).all():
                lo = mid
            else:
                hi = mid

        result = blob_params.clone()
        result[:, :, 6] = blob_params[:, :, 6] * lo
        return result

    def apply(
        self,
        image: np.ndarray,
        alpha_mask: np.ndarray,
        budget_level: float,
    ) -> np.ndarray:
        """Apply alpha mask to image with budget projection.

        Args:
            image: (H, W, 3) uint8 or float32 in [0, 1].
            alpha_mask: (H, W) float in [0, max_opacity].
            budget_level: Maximum coverage fraction.

        Returns:
            Dirty image in same dtype as input.
        """
        input_dtype = image.dtype
        if input_dtype == np.uint8:
            img_float = image.astype(np.float32) / 255.0
        else:
            img_float = image.astype(np.float32)

        alpha_proj = self.project_to_budget(alpha_mask, budget_level)
        alpha_3d = alpha_proj[:, :, np.newaxis]
        dust = self.dust_color[np.newaxis, np.newaxis, :]
        blended = dust * alpha_3d + img_float * (1.0 - alpha_3d)
        blended = np.clip(blended, 0.0, 1.0)

        if input_dtype == np.uint8:
            return (blended * 255).astype(np.uint8)
        return blended


class ReinforceTrainer:
    """REINFORCE trainer for the dust generator.

    Since we cannot backprop through the simulator, we use the REINFORCE
    estimator with an EMA baseline for variance reduction.

    For each step:
        1. Sample n masks from the generator (with exploration noise)
        2. Evaluate each mask (run episodes, get success rate)
        3. Compute REINFORCE loss: -(reward - baseline) * log_prob
        4. Update generator parameters

    Works with both ``DustGenerator`` and ``BlobGenerator``.
    """

    def __init__(
        self,
        generator,
        config,
        baseline_ema: float = 0.9,
    ):
        self.generator = generator
        self.config = config
        self.optimizer = optim.Adam(
            generator.parameters(), lr=config.learning_rate
        )
        self.baseline = 0.5  # initial baseline
        self.baseline_ema = baseline_ema

    def step(
        self,
        image: torch.Tensor,
        evaluate_fn: Callable,
        n_samples: int,
        sigma: float,
        k: Optional[int] = None,
        temporal: bool = False,
    ) -> Dict[str, float]:
        """One REINFORCE update step.

        Args:
            image: (1, 3, H, W) reference image for conditioning.
            evaluate_fn: fn(alpha_mask_HW) -> success_rate (static mode), or
                fn(blob_params_K7) -> success_rate (temporal mode).
            n_samples: Number of mask samples per step.
            sigma: Exploration noise std.
            k: Number of blobs (for BlobGenerator). Ignored by DustGenerator.
            temporal: If True, pass blob_params (K,7) to evaluate_fn
                instead of a static mask.

        Returns:
            Dict with 'loss', 'mean_reward', 'mean_success_rate'.
        """
        self.generator.train()

        # BlobGenerator.sample_with_exploration accepts k;
        # DustGenerator.sample_with_exploration does not.
        if k is not None:
            mean_masks, sampled_masks, log_probs, noisy_params = (
                self.generator.sample_with_exploration(
                    image, sigma=sigma, n_samples=n_samples, k=k
                )
            )
        else:
            mean_masks, sampled_masks, log_probs, noisy_params = (
                self.generator.sample_with_exploration(
                    image, sigma=sigma, n_samples=n_samples
                )
            )

        # Evaluate each sampled mask
        rewards = []
        success_rates = []
        for i in range(n_samples):
            if temporal and noisy_params is not None:
                # Pass blob_params (K, 7) numpy array to evaluate_fn
                bp_np = noisy_params[i].detach().cpu().numpy()  # (K, 7)
                if k is not None:
                    bp_np = bp_np[:k]
                sr = evaluate_fn(bp_np)
            else:
                mask_np = self.generator.mask_to_numpy(sampled_masks[i:i+1])[0]  # (H, W)
                sr = evaluate_fn(mask_np)
            reward = 1.0 - sr  # attacker wants to minimize success
            rewards.append(reward)
            success_rates.append(sr)

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=image.device)
        advantages = rewards_t - self.baseline

        # REINFORCE loss: negative because optimizer minimizes
        loss = -(advantages * log_probs).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update EMA baseline
        mean_reward = float(rewards_t.mean())
        self.baseline = self.baseline_ema * self.baseline + (
            1 - self.baseline_ema
        ) * mean_reward

        return {
            "loss": float(loss),
            "mean_reward": mean_reward,
            "mean_success_rate": float(np.mean(success_rates)),
        }


@dataclass
class RoundMetrics:
    """Metrics collected during one round of adversarial training."""

    round_idx: int
    budget_level: float
    attacker_losses: List[float]
    attacker_mean_rewards: List[float]
    attacker_mean_srs: List[float]
    adversarial_eval_sr: float
    clean_eval_sr: float
    defender_updated: bool


class AdversarialTrainer:
    """Orchestrates the alternating attacker/defender zero-sum training loop.

    For each round and each budget level:
        1. ATTACKER PHASE: Train generator with REINFORCE for K steps
        2. DEFENDER PHASE: Fine-tune Octo with GRPO using collected trajectories
        3. EVAL PHASE: Evaluate current equilibrium

    The defender is optional — if ``defender`` is None, only the attacker trains
    (useful for testing the generator in isolation).

    Supports both ``DustGenerator`` (coverage-fraction budgets) and
    ``BlobGenerator`` (blob-count budgets).
    """

    def __init__(
        self,
        config: AdversarialExperimentConfig,
        generator,
        mask_applicator: MaskApplicator,
        evaluate_fn: Callable[[Optional[np.ndarray], int], float],
        defender=None,
        reference_image: Optional[np.ndarray] = None,
        record_fn: Optional[Callable[[Optional[np.ndarray], int, float], None]] = None,
    ):
        """
        Args:
            config: Full experiment config.
            generator: DustGenerator or BlobGenerator network.
            mask_applicator: Applies masks to images.
            evaluate_fn: fn(alpha_mask_or_None, n_episodes) -> success_rate.
                Pass None for clean eval.
            defender: OctoGRPO instance or None.
            reference_image: (H, W, 3) uint8 image for generator conditioning.
            record_fn: Optional fn(alpha_mask_or_None, round_idx, budget_level)
                to record sample episode animations after eval.
        """
        self.config = config
        self.generator = generator
        self.mask_applicator = mask_applicator
        self.evaluate_fn = evaluate_fn
        self.defender = defender
        self.record_fn = record_fn
        self.use_blob_budget = config.blob_generator is not None
        self.temporal = (
            config.blob_generator is not None
            and config.blob_generator.temporal
        )

        # Convert reference image to torch for generator input.
        # Place on same device as generator.
        if reference_image is not None:
            img_float = reference_image.astype(np.float32) / 255.0
            device = next(generator.parameters()).device
            self.ref_image_torch = (
                torch.tensor(img_float.tolist(), dtype=torch.float32)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )
        else:
            self.ref_image_torch = None

        self.history: List[RoundMetrics] = []

    def _make_evaluate_with_budget(
        self, budget_level: float, n_episodes: int
    ) -> Callable:
        """Create evaluation function that applies budget projection.

        For blob budgets (integer blob count), no projection is needed — the
        mask is already rendered with the correct number of blobs.
        For temporal mode, the evaluate_fn receives blob_params (K, 7) directly.
        """
        if self.temporal:
            # Temporal mode: evaluate_fn receives blob_params (K, 7)
            # Area-time budget projection is applied inside
            def _eval(blob_params_or_mask) -> float:
                return self.evaluate_fn(blob_params_or_mask, n_episodes)
            return _eval

        if self.use_blob_budget:
            # No coverage projection for blob budgets
            def _eval(alpha_mask: np.ndarray) -> float:
                return self.evaluate_fn(alpha_mask, n_episodes)
            return _eval

        def _eval(alpha_mask: np.ndarray) -> float:
            projected = self.mask_applicator.project_to_budget(
                alpha_mask, budget_level
            )
            return self.evaluate_fn(projected, n_episodes)

        return _eval

    def train_round(
        self, round_idx: int, budget_level: float, clean_sr: Optional[float] = None
    ) -> RoundMetrics:
        """Run one round of adversarial training at a given budget level.

        Args:
            round_idx: Current round index.
            budget_level: Dust budget fraction (DustGenerator) or blob count (BlobGenerator).
            clean_sr: Pre-computed clean success rate for this round.
                If None, evaluates clean SR here.
        """
        tc = self.config.training
        gc = self.config.blob_generator if self.use_blob_budget else self.config.generator

        # Format budget label
        if self.temporal:
            budget_label = f"{budget_level:.0%} area-time"
        elif self.use_blob_budget:
            budget_label = f"{int(budget_level)} blobs"
        else:
            budget_label = f"{budget_level:.0%}"

        logger.info(
            f"Round {round_idx + 1}/{tc.n_rounds}, budget={budget_label}"
        )

        # --- ATTACKER PHASE ---
        logger.info("  Attacker phase: training generator")
        reinforce = ReinforceTrainer(self.generator, gc)

        eval_fn = self._make_evaluate_with_budget(
            budget_level, tc.episodes_per_reinforce_sample
        )

        # For blob budgets, pass k to REINFORCE step
        blob_k = int(budget_level) if (self.use_blob_budget and not self.temporal) else None

        attacker_losses = []
        attacker_rewards = []
        attacker_srs = []

        for step in range(tc.attacker_steps_per_round):
            metrics = reinforce.step(
                self.ref_image_torch,
                evaluate_fn=eval_fn,
                n_samples=tc.n_samples_per_step,
                sigma=gc.explore_sigma,
                k=blob_k,
                temporal=self.temporal,
            )
            attacker_losses.append(metrics["loss"])
            attacker_rewards.append(metrics["mean_reward"])
            attacker_srs.append(metrics["mean_success_rate"])

            if (step + 1) % 10 == 0:
                logger.info(
                    f"    Step {step + 1}/{tc.attacker_steps_per_round}: "
                    f"loss={metrics['loss']:.4f}, "
                    f"sr={metrics['mean_success_rate']:.3f}"
                )

        # --- DEFENDER PHASE ---
        defender_updated = False
        if self.defender is not None:
            logger.info("  Defender phase: fine-tuning Octo with GRPO")
            self.defender.train_round(self.generator, budget_level)
            defender_updated = True

        # --- EVAL PHASE ---
        logger.info("  Eval phase")

        self.generator.eval()
        with torch.no_grad():
            if self.temporal:
                # In temporal mode, get blob_params (K, 7) for evaluation
                blob_params = self.generator._predict_blob_params(
                    self.ref_image_torch
                )  # (1, K_max, 7)
                # Project area-time budget
                blob_params = self.mask_applicator.project_area_time_budget(
                    blob_params, budget_level, self.generator,
                    T=self.config.env.max_episode_steps,
                )
                best_eval_input = blob_params[0].cpu().numpy()  # (K, 7)
                adversarial_sr = self.evaluate_fn(best_eval_input, tc.eval_episodes)
            else:
                # Static mask mode
                if self.use_blob_budget:
                    best_mask = self.generator.generate_mask(
                        self.ref_image_torch, k=blob_k
                    )
                else:
                    best_mask = self.generator.generate_mask(self.ref_image_torch)
                best_mask_np = self.generator.mask_to_numpy(best_mask)[0]  # (H, W)

                if self.use_blob_budget:
                    best_mask_proj = best_mask_np
                else:
                    best_mask_proj = self.mask_applicator.project_to_budget(
                        best_mask_np, budget_level
                    )
                best_eval_input = best_mask_proj
                adversarial_sr = self.evaluate_fn(best_eval_input, tc.eval_episodes)

        if clean_sr is None:
            clean_sr = self.evaluate_fn(None, tc.eval_episodes)

        logger.info(
            f"  Round {round_idx + 1} results: "
            f"adv_sr={adversarial_sr:.3f}, clean_sr={clean_sr:.3f}"
        )

        # Record sample episode animation
        if self.record_fn is not None:
            try:
                self.record_fn(best_eval_input, round_idx, budget_level)
            except Exception as e:
                logger.warning(f"  Failed to record animation: {e}")

        metrics = RoundMetrics(
            round_idx=round_idx,
            budget_level=budget_level,
            attacker_losses=attacker_losses,
            attacker_mean_rewards=attacker_rewards,
            attacker_mean_srs=attacker_srs,
            adversarial_eval_sr=adversarial_sr,
            clean_eval_sr=clean_sr,
            defender_updated=defender_updated,
        )
        self.history.append(metrics)
        return metrics

    def train(self) -> List[RoundMetrics]:
        """Run the full adversarial training loop across all rounds and budgets."""
        tc = self.config.training
        all_metrics = []

        for round_idx in range(tc.n_rounds):
            # Evaluate clean SR once per round (budget-independent)
            clean_sr = self.evaluate_fn(None, tc.eval_episodes)
            logger.info(f"Round {round_idx + 1} clean baseline: sr={clean_sr:.3f}")
            for budget in tc.budget_levels:
                metrics = self.train_round(round_idx, budget, clean_sr=clean_sr)
                all_metrics.append(metrics)

        return all_metrics

    def get_history_dict(self) -> Dict[str, List]:
        """Convert history to a serializable dict for logging/plotting."""
        return {
            "rounds": [m.round_idx for m in self.history],
            "budgets": [m.budget_level for m in self.history],
            "adversarial_srs": [m.adversarial_eval_sr for m in self.history],
            "clean_srs": [m.clean_eval_sr for m in self.history],
            "attacker_final_srs": [
                m.attacker_mean_srs[-1] if m.attacker_mean_srs else 0.0
                for m in self.history
            ],
        }
