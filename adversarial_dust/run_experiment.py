"""CLI entry point for adversarial dust experiments."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from adversarial_dust.budget_sweep import BudgetSweep, make_dust_model
from adversarial_dust.config import ExperimentConfig, load_config
from adversarial_dust.visualization import (
    plot_degradation_curves,
    visualize_budget_comparison,
    visualize_dust_pattern,
)


def setup_logging(output_dir: str):
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "experiment.log"),
        ],
    )


def create_policy(config: ExperimentConfig):
    """Create policy via SimplerEnv's inference wrappers."""
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


def get_image_shape(config: ExperimentConfig) -> tuple:
    """Get image shape from environment by doing a quick reset."""
    import simpler_env
    from simpler_env.utils.env.observation_utils import (
        get_image_from_maniskill2_obs_dict,
    )

    env = simpler_env.make(
        config.env.task_name,
        **config.env.make_kwargs(),
    )
    obs, _ = env.reset()
    image = get_image_from_maniskill2_obs_dict(env, obs)
    shape = image.shape
    env.close()
    return shape


def main():
    parser = argparse.ArgumentParser(
        description="Run adversarial camera dust experiments"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Override: run a single budget level instead of full sweep",
    )
    parser.add_argument(
        "--video-fmt",
        type=str,
        default="mp4",
        choices=["mp4", "gif"],
        help="Format for episode animation recordings",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.budget is not None:
        config.sweep.budget_levels = [args.budget]

    setup_logging(config.output_dir)
    logger = logging.getLogger(__name__)
    logger.info(f"Config: {config}")
    logger.info(f"Output dir: {config.output_dir}")

    # Get image shape from env
    logger.info("Getting image shape from environment...")
    image_shape = get_image_shape(config)
    logger.info(f"Image shape: {image_shape}")

    # Create policy
    logger.info("Loading Octo policy...")
    policy = create_policy(config)

    # Run budget sweep
    sweep = BudgetSweep(config, policy, image_shape)
    results = sweep.run()

    # Generate visualizations
    output_dir = Path(config.output_dir)

    logger.info("Generating degradation curves plot...")
    plot_degradation_curves(results, str(output_dir / "degradation_curves.png"))

    # Generate dust pattern visualizations using a sample image from env
    logger.info("Generating dust pattern visualizations...")
    import simpler_env
    from simpler_env.utils.env.observation_utils import (
        get_image_from_maniskill2_obs_dict,
    )

    env = simpler_env.make(
        config.env.task_name,
        **config.env.make_kwargs(),
    )
    obs, _ = env.reset()
    sample_image = get_image_from_maniskill2_obs_dict(env, obs)
    env.close()

    # Visualize adversarial pattern for each budget level
    for budget_str, br in results["budget_results"].items():
        budget = float(budget_str)
        dust_model = make_dust_model(config, image_shape, budget)
        params = np.array(br["adversarial_params"])
        visualize_dust_pattern(
            dust_model,
            params,
            sample_image,
            str(output_dir / f"dust_pattern_budget_{budget:.2f}.png"),
        )

    # Budget comparison grid
    dust_cfg = config.blob if config.dust_model_type == "blob" else config.dust
    model_factory = lambda shape, budget: make_dust_model(config, shape, budget)
    visualize_budget_comparison(
        results,
        image_shape,
        dust_cfg,
        sample_image,
        str(output_dir / "budget_comparison.png"),
        model_factory=model_factory,
    )

    # Record episode animations
    logger.info("Recording episode animations...")
    from adversarial_dust.animation import record_all_budget_animations

    record_all_budget_animations(
        config, policy, image_shape, results, config.output_dir, fmt=args.video_fmt
    )

    logger.info("Experiment complete!")


if __name__ == "__main__":
    main()
