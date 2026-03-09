"""CMA-ES optimizer for finding adversarial dust patterns."""

import json
import logging
from pathlib import Path

import cma
import numpy as np

from adversarial_dust.config import OptimizationConfig
from adversarial_dust.evaluator import PolicyEvaluator

logger = logging.getLogger(__name__)


class AdversarialDustOptimizer:
    """Uses CMA-ES to find dust patterns that minimize policy success rate.

    Works with any dust model that exposes:
        n_params, get_cma_bounds(), get_cma_x0(), apply(image, params, timestep)
    """

    def __init__(
        self,
        dust_model,
        evaluator: PolicyEvaluator,
        opt_config: OptimizationConfig,
        output_dir: str,
    ):
        self.dust_model = dust_model
        self.evaluator = evaluator
        self.opt_config = opt_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _evaluate_candidate(self, params: np.ndarray) -> float:
        """Evaluate a single candidate: run episodes, return success rate.

        Budget projection happens inside dust_model.apply() at each timestep.
        """
        return self.evaluator.evaluate(params, self.opt_config.episodes_per_eval)

    def optimize(self) -> dict:
        """Run CMA-ES optimization loop.

        Returns:
            dict with keys: best_params, best_success_rate, history
        """
        x0 = self.dust_model.get_cma_x0()
        lb, ub = self.dust_model.get_cma_bounds()

        opts = {
            "popsize": self.opt_config.population_size,
            "maxiter": self.opt_config.max_generations,
            "seed": self.opt_config.seed,
            "bounds": [lb, ub],
            "verbose": -1,
        }
        es = cma.CMAEvolutionStrategy(x0, self.opt_config.sigma0, opts)

        history = []
        best_params = None
        best_fitness = float("inf")

        generation = 0
        while not es.stop():
            candidates = es.ask()

            # Evaluate each candidate (minimizing success rate)
            fitness_values = []
            for candidate in candidates:
                sr = self._evaluate_candidate(np.array(candidate))
                fitness_values.append(sr)

            es.tell(candidates, fitness_values)

            gen_best_idx = int(np.argmin(fitness_values))
            gen_best_sr = fitness_values[gen_best_idx]
            gen_mean_sr = float(np.mean(fitness_values))

            if gen_best_sr < best_fitness:
                best_fitness = gen_best_sr
                best_params = np.array(candidates[gen_best_idx])

            gen_stats = {
                "generation": generation,
                "best_sr": gen_best_sr,
                "mean_sr": gen_mean_sr,
                "overall_best_sr": best_fitness,
            }
            history.append(gen_stats)
            logger.info(
                f"Gen {generation}: best_sr={gen_best_sr:.3f}, "
                f"mean_sr={gen_mean_sr:.3f}, overall_best={best_fitness:.3f}"
            )
            generation += 1

        # Final evaluation with more episodes
        logger.info(
            f"Final evaluation of best pattern with {self.opt_config.episodes_final_eval} episodes"
        )
        final_sr = self.evaluator.evaluate(
            best_params, self.opt_config.episodes_final_eval
        )
        logger.info(f"Final adversarial success rate: {final_sr:.3f}")

        result = {
            "best_params": best_params.tolist(),
            "best_success_rate": final_sr,
            "optimization_best_sr": best_fitness,
            "history": history,
        }

        # Save optimization log
        log_path = self.output_dir / "optimization_log.json"
        with open(log_path, "w") as f:
            json.dump(result, f, indent=2)

        return result
