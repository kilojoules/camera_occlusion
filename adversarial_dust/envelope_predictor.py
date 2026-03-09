"""Safe operating envelope predictor for vision-based robotic manipulation.

Runs adversarial auditing across multiple occlusion types and budget levels
to determine the certified safe operating region for a given policy+task.
"""

import json
import logging
from pathlib import Path

import numpy as np

from adversarial_dust.config import (
    EnvelopeExperimentConfig,
    FingerprintConfig,
    GlareConfig,
    DustGridConfig,
    BlobConfig,
    OptimizationConfig,
)
from adversarial_dust.dust_model import AdversarialDustModel
from adversarial_dust.blob_model import DynamicBlobDustModel
from adversarial_dust.fingerprint_model import FingerprintSmudgeModel
from adversarial_dust.glare_model import AdversarialGlareModel
from adversarial_dust.evaluator import PolicyEvaluator
from adversarial_dust.optimizer import AdversarialDustOptimizer

logger = logging.getLogger(__name__)


def make_occlusion_model(
    occlusion_type: str,
    config: EnvelopeExperimentConfig,
    image_shape: tuple,
    budget_level: float,
):
    """Factory: create an occlusion model by type name."""
    if occlusion_type == "fingerprint":
        return FingerprintSmudgeModel(config.fingerprint, image_shape, budget_level)
    elif occlusion_type == "glare":
        return AdversarialGlareModel(config.glare, image_shape, budget_level)
    elif occlusion_type == "blob":
        return DynamicBlobDustModel(config.blob, image_shape, budget_level)
    elif occlusion_type == "grid":
        return AdversarialDustModel(config.dust, image_shape, budget_level)
    else:
        raise ValueError(f"Unknown occlusion type: {occlusion_type}")


class SafeOperatingEnvelopePredictor:
    """Predicts the safe operating envelope for a robotic manipulation policy.

    For each occlusion type (fingerprint, glare, dust, etc.), runs adversarial
    CMA-ES optimization at each budget level to find the worst-case success rate.
    The operating envelope is the region where worst-case SR >= safe_threshold.

    Results structure:
        {
            "task": str,
            "policy": str,
            "clean_sr": float,
            "safe_threshold": float,
            "marginal_threshold": float,
            "occlusion_results": {
                "fingerprint": {
                    "0.01": {
                        "adversarial_sr": float,
                        "random_mean_sr": float,
                        "random_std_sr": float,
                        "adversarial_params": list,
                        "zone": "safe" | "marginal" | "unsafe"
                    },
                    ...
                },
                ...
            },
            "envelope_summary": {
                "fingerprint": {
                    "max_safe_budget": float,  # largest budget still certified safe
                    "breakdown_budget": float,  # smallest budget causing failure
                },
                ...
            }
        }
    """

    def __init__(
        self,
        config: EnvelopeExperimentConfig,
        policy,
        image_shape: tuple,
    ):
        self.config = config
        self.policy = policy
        self.image_shape = image_shape
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _make_opt_config(self) -> OptimizationConfig:
        """Build OptimizationConfig from envelope settings."""
        ec = self.config.envelope
        return OptimizationConfig(
            population_size=ec.population_size,
            max_generations=ec.max_generations,
            sigma0=ec.sigma0,
            seed=ec.seed,
            episodes_per_eval=ec.episodes_per_eval,
            episodes_final_eval=ec.episodes_final_eval,
            n_random_baselines=ec.n_random_baselines,
        )

    def _classify_zone(self, sr: float) -> str:
        """Classify a success rate into safe/marginal/unsafe."""
        if sr >= self.config.envelope.safe_threshold:
            return "safe"
        elif sr >= self.config.envelope.marginal_threshold:
            return "marginal"
        return "unsafe"

    def _evaluate_clean_baseline(self) -> float:
        """Evaluate policy with no occlusion."""
        logger.info("Evaluating clean baseline (no occlusion)")
        # Use grid model with zero budget as a no-op
        model = AdversarialDustModel(
            self.config.dust, self.image_shape, budget_level=0.0
        )
        evaluator = PolicyEvaluator(self.config.env, self.policy, model)
        return evaluator.evaluate(
            dust_params=None,
            n_episodes=self.config.envelope.episodes_final_eval,
        )

    def _audit_single(
        self, occlusion_type: str, budget_level: float
    ) -> dict:
        """Run adversarial audit for one occlusion type at one budget level."""
        logger.info(f"Auditing {occlusion_type} at budget={budget_level:.2%}")

        model = make_occlusion_model(
            occlusion_type, self.config, self.image_shape, budget_level
        )
        evaluator = PolicyEvaluator(self.config.env, self.policy, model)
        opt_config = self._make_opt_config()

        budget_dir = self.output_dir / occlusion_type / f"budget_{budget_level:.2f}"
        budget_dir.mkdir(parents=True, exist_ok=True)

        # Adversarial optimization (CMA-ES)
        optimizer = AdversarialDustOptimizer(
            dust_model=model,
            evaluator=evaluator,
            opt_config=opt_config,
            output_dir=str(budget_dir),
        )
        adv_result = optimizer.optimize()

        # Random baselines
        rng = np.random.default_rng(opt_config.seed)
        random_srs = []
        for i in range(opt_config.n_random_baselines):
            params = model.get_random_params(rng)
            sr = evaluator.evaluate(params, opt_config.episodes_final_eval)
            random_srs.append(sr)

        adv_sr = adv_result["best_success_rate"]
        result = {
            "adversarial_sr": adv_sr,
            "adversarial_params": adv_result["best_params"],
            "random_mean_sr": float(np.mean(random_srs)),
            "random_std_sr": float(np.std(random_srs)),
            "random_srs": random_srs,
            "zone": self._classify_zone(adv_sr),
            "optimization_history": adv_result["history"],
        }

        logger.info(
            f"  {occlusion_type} budget={budget_level:.2%}: "
            f"adv_sr={adv_sr:.3f} ({result['zone']}), "
            f"random={np.mean(random_srs):.3f}+/-{np.std(random_srs):.3f}"
        )
        return result

    def _compute_envelope_summary(self, type_results: dict) -> dict:
        """Compute max safe budget and breakdown budget for one occlusion type."""
        budgets = sorted(float(b) for b in type_results.keys())

        max_safe_budget = 0.0
        breakdown_budget = None

        for budget in budgets:
            zone = type_results[str(budget)]["zone"]
            if zone == "safe":
                max_safe_budget = budget
            if zone == "unsafe" and breakdown_budget is None:
                breakdown_budget = budget

        return {
            "max_safe_budget": max_safe_budget,
            "breakdown_budget": breakdown_budget if breakdown_budget is not None else float("inf"),
        }

    def predict(self) -> dict:
        """Run the full envelope prediction pipeline.

        Returns the complete results dict.
        """
        results = {
            "task": self.config.env.task_name,
            "policy": self.config.env.policy_model,
            "safe_threshold": self.config.envelope.safe_threshold,
            "marginal_threshold": self.config.envelope.marginal_threshold,
        }

        # Clean baseline
        clean_sr = self._evaluate_clean_baseline()
        results["clean_sr"] = clean_sr
        logger.info(f"Clean baseline: {clean_sr:.3f}")

        # Audit each occlusion type
        occlusion_results = {}
        for occ_type in self.config.envelope.occlusion_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"Occlusion type: {occ_type}")
            logger.info(f"{'='*60}")

            type_results = {}
            for budget in self.config.envelope.budget_levels:
                result = self._audit_single(occ_type, budget)
                type_results[str(budget)] = result

            occlusion_results[occ_type] = type_results

        results["occlusion_results"] = occlusion_results

        # Compute envelope summary
        envelope_summary = {}
        for occ_type, type_results in occlusion_results.items():
            envelope_summary[occ_type] = self._compute_envelope_summary(type_results)
        results["envelope_summary"] = envelope_summary

        # Save results
        results_path = self.output_dir / "envelope_results.json"
        # Convert params to lists for JSON serialization
        serializable = _make_serializable(results)
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Results saved to {results_path}")

        # Log summary
        logger.info("\n" + "=" * 60)
        logger.info("SAFE OPERATING ENVELOPE SUMMARY")
        logger.info("=" * 60)
        for occ_type, summary in envelope_summary.items():
            logger.info(
                f"  {occ_type}: safe up to {summary['max_safe_budget']:.1%} coverage, "
                f"breaks at {summary['breakdown_budget']:.1%}"
            )

        return results


def _make_serializable(obj):
    """Recursively convert numpy types to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if obj == float("inf"):
        return "inf"
    return obj
