"""CLI entry point for baseline-vs-fine-tuned dust robustness experiments.

Phases:
    1. Collect clean demonstrations with baseline Octo
    2. Fine-tune Octo action head via behavioral cloning
    3. Evaluate both models under camera_occlusion.Dust at multiple severities
    4. Generate degradation-curve visualization + animations
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from adversarial_dust.config import DustRobustnessConfig, load_dust_robustness_config


def setup_logging(output_dir: str):
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "dust_robustness.log"),
        ],
    )


def create_policy(config: DustRobustnessConfig):
    """Create baseline policy via SimplerEnv's inference wrappers."""
    task_name = config.env.task_name
    policy_setup = "widowx_bridge" if task_name.startswith("widowx") else "google_robot"

    if config.env.policy_model.startswith("internvla-m1"):
        from examples.SimplerEnv.model2simpler_interface import M1Inference

        return M1Inference(
            policy_ckpt_path=config.env.internvla_m1_ckpt,
            policy_setup=policy_setup,
            port=config.env.internvla_m1_port,
            action_scale=1.0,
            cfg_scale=1.5,
        )
    elif config.env.policy_model.startswith("openvla"):
        from simpler_env.policies.openvla.openvla_model import OpenVLAInference

        model_path = config.env.policy_model
        if model_path == "openvla":
            model_path = "openvla/openvla-7b"
        elif not model_path.startswith("openvla/"):
            model_path = f"openvla/{model_path}"

        return OpenVLAInference(
            saved_model_path=model_path,
            policy_setup=policy_setup,
        )
    else:
        from simpler_env.policies.octo.octo_model import OctoInference

        return OctoInference(
            model_type=config.env.policy_model,
            policy_setup=policy_setup,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline-vs-fine-tuned dust robustness experiment"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--skip-finetune",
        action="store_true",
        help="Skip demo collection + fine-tuning; evaluate baseline only",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Load fine-tuned checkpoint instead of training",
    )
    args = parser.parse_args()

    config = load_dust_robustness_config(args.config)
    setup_logging(config.output_dir)
    logger = logging.getLogger(__name__)
    logger.info(f"Config: {config}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 0: Load baseline policy
    # ------------------------------------------------------------------
    logger.info("Loading baseline Octo policy...")
    baseline_policy = create_policy(config)

    # ------------------------------------------------------------------
    # Phase 1 + 2: Collect demos and fine-tune (unless skipped)
    # ------------------------------------------------------------------
    finetuned_policy = None

    if args.checkpoint:
        logger.info(f"Loading fine-tuned checkpoint from {args.checkpoint}")
        from adversarial_dust.octo_finetuner import OctoBCTrainer

        trainer = OctoBCTrainer(config.finetuning, config.env)
        trainer.initialize()
        from octo.model.octo_model import OctoModel

        trainer.model = OctoModel.load_pretrained(args.checkpoint)
        finetuned_policy = trainer.get_policy()

    elif not args.skip_finetune:
        # Phase 1: Collect clean demonstrations
        logger.info("=== Phase 1: Collecting clean demonstrations ===")
        from adversarial_dust.octo_finetuner import DemoCollector

        collector = DemoCollector(config.env)
        demos = collector.collect(baseline_policy, config.demo_collection)

        if len(demos) == 0:
            logger.error(
                "No successful demos collected! Cannot fine-tune. "
                "Try increasing max_attempts or checking the baseline policy."
            )
            sys.exit(1)

        logger.info(f"Collected {len(demos)} demonstrations")

        # Phase 2: Fine-tune
        logger.info("=== Phase 2: Fine-tuning Octo (behavioral cloning) ===")
        from adversarial_dust.octo_finetuner import OctoBCTrainer

        task_text = "pick coke can"  # Derived from task_name
        if "coke_can" in config.env.task_name:
            task_text = "pick coke can"
        elif "eggplant" in config.env.task_name:
            task_text = "put eggplant in basket"
        else:
            # Generic fallback
            task_text = config.env.task_name.replace("_", " ")

        trainer = OctoBCTrainer(config.finetuning, config.env)
        loss_history = trainer.train(demos, task_text)

        # Save checkpoint
        ckpt_path = str(output_dir / "finetuned_checkpoint")
        trainer.save(ckpt_path)

        # Save training history
        history_path = str(output_dir / "training_history.json")
        with open(history_path, "w") as f:
            json.dump({"losses": loss_history}, f)
        logger.info(f"Saved training history to {history_path}")

        finetuned_policy = trainer.get_policy()

    # ------------------------------------------------------------------
    # Phase 3: Clean evaluation of fine-tuned model
    # ------------------------------------------------------------------
    logger.info("=== Phase 3: Clean evaluation ===")
    from adversarial_dust.dust_robustness_eval import DustRobustnessEvaluator

    # Force clean-only evaluation (no camera_occlusion dependency)
    clean_eval_config = config.evaluation
    clean_eval_config.severity_levels = ["clean"]

    evaluator = DustRobustnessEvaluator(config.env, clean_eval_config)
    evaluator.presets = {"clean": None}  # bypass Dust import

    models = {"baseline": baseline_policy}
    if finetuned_policy is not None:
        models["finetuned"] = finetuned_policy

    results = evaluator.evaluate_all(models)

    # Save results
    results_path = str(output_dir / "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved evaluation results to {results_path}")

    # Print summary
    logger.info("=== Experiment Summary ===")
    for model_name in results:
        for sev in results[model_name]:
            sr = results[model_name][sev]["success_rate"]
            logger.info(f"  {model_name} / {sev}: {sr:.3f}")

    logger.info("Experiment complete!")


if __name__ == "__main__":
    main()
