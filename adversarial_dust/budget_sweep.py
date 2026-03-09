"""Orchestrate adversarial optimization across budget levels with random baselines."""

import json
import logging
from pathlib import Path

import numpy as np

from adversarial_dust.config import ExperimentConfig
from adversarial_dust.dust_model import AdversarialDustModel
from adversarial_dust.blob_model import DynamicBlobDustModel
from adversarial_dust.fingerprint_model import FingerprintSmudgeModel
from adversarial_dust.glare_model import AdversarialGlareModel
from adversarial_dust.evaluator import PolicyEvaluator
from adversarial_dust.optimizer import AdversarialDustOptimizer

logger = logging.getLogger(__name__)


def make_dust_model(config: ExperimentConfig, image_shape: tuple, budget_level: float):
    """Factory: create dust model based on config.dust_model_type."""
    if config.dust_model_type == "blob":
        return DynamicBlobDustModel(config.blob, image_shape, budget_level)
    if config.dust_model_type == "fingerprint":
        return FingerprintSmudgeModel(config.fingerprint, image_shape, budget_level)
    if config.dust_model_type == "glare":
        return AdversarialGlareModel(config.glare, image_shape, budget_level)
    return AdversarialDustModel(config.dust, image_shape, budget_level)


class BudgetSweep:
    """Run adversarial optimization + random baselines across budget levels."""

    def __init__(self, config: ExperimentConfig, policy, image_shape: tuple):
        self.config = config
        self.policy = policy
        self.image_shape = image_shape
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _make_dust_model(self, budget_level: float):
        return make_dust_model(self.config, self.image_shape, budget_level)

    def _make_evaluator(self, dust_model: AdversarialDustModel) -> PolicyEvaluator:
        return PolicyEvaluator(self.config.env, self.policy, dust_model)

    def _evaluate_clean_baseline(self) -> float:
        """Evaluate policy with no dust."""
        logger.info("Evaluating clean baseline (no dust)")
        dust_model = self._make_dust_model(budget_level=0.0)
        evaluator = self._make_evaluator(dust_model)
        return evaluator.evaluate(
            dust_params=None,
            n_episodes=self.config.optimization.episodes_final_eval,
        )

    def _evaluate_random_baselines(
        self, budget_level: float, n_baselines: int
    ) -> list:
        """Evaluate random dust patterns at a given budget level."""
        dust_model = self._make_dust_model(budget_level)
        evaluator = self._make_evaluator(dust_model)
        rng = np.random.default_rng(self.config.optimization.seed)

        random_srs = []
        for i in range(n_baselines):
            params = dust_model.get_random_params(rng)
            sr = evaluator.evaluate(
                params, self.config.optimization.episodes_final_eval
            )
            random_srs.append(sr)
            logger.info(
                f"Random baseline {i + 1}/{n_baselines} at budget={budget_level}: sr={sr:.3f}"
            )
        return random_srs

    def run(self) -> dict:
        """Execute full budget sweep. Returns results dict."""
        results = {}

        # Clean baseline
        clean_sr = self._evaluate_clean_baseline()
        results["clean_sr"] = clean_sr
        logger.info(f"Clean baseline success rate: {clean_sr:.3f}")

        # Sweep over budget levels
        budget_results = {}
        for budget_level in self.config.sweep.budget_levels:
            logger.info(f"\n{'='*60}")
            logger.info(f"Budget level: {budget_level}")
            logger.info(f"{'='*60}")

            budget_dir = self.output_dir / f"budget_{budget_level:.2f}"
            budget_dir.mkdir(parents=True, exist_ok=True)

            # Adversarial optimization
            dust_model = self._make_dust_model(budget_level)
            evaluator = self._make_evaluator(dust_model)
            optimizer = AdversarialDustOptimizer(
                dust_model=dust_model,
                evaluator=evaluator,
                opt_config=self.config.optimization,
                output_dir=str(budget_dir),
            )
            adv_result = optimizer.optimize()

            # Random baselines
            random_srs = self._evaluate_random_baselines(
                budget_level, self.config.optimization.n_random_baselines
            )

            budget_results[str(budget_level)] = {
                "adversarial_sr": adv_result["best_success_rate"],
                "adversarial_params": adv_result["best_params"],
                "random_srs": random_srs,
                "random_mean_sr": float(np.mean(random_srs)),
                "random_std_sr": float(np.std(random_srs)),
                "clean_sr": clean_sr,
                "optimization_history": adv_result["history"],
            }

            logger.info(
                f"Budget {budget_level}: "
                f"adv_sr={adv_result['best_success_rate']:.3f}, "
                f"random_mean={np.mean(random_srs):.3f}±{np.std(random_srs):.3f}, "
                f"clean={clean_sr:.3f}"
            )

        results["budget_results"] = budget_results

        # Save results
        results_path = self.output_dir / "sweep_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_path}")

        return results
