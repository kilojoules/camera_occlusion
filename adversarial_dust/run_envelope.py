"""CLI entry point for safe operating envelope prediction experiments."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from adversarial_dust.config import EnvelopeExperimentConfig, load_envelope_config
from adversarial_dust.envelope_predictor import SafeOperatingEnvelopePredictor
from adversarial_dust.visualization import plot_operating_envelope, plot_occlusion_gallery


def setup_logging(output_dir: str):
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "envelope.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    # Flush on every log line so we can monitor progress via tail
    class FlushHandler(logging.StreamHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    stream_handler = FlushHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    logging.basicConfig(
        level=logging.INFO,
        handlers=[stream_handler, file_handler],
        force=True,
    )

    # Also explicitly configure the adversarial_dust loggers
    # (some library imports may interfere with propagation)
    for name in ["adversarial_dust", "adversarial_dust.envelope_predictor",
                 "adversarial_dust.evaluator", "adversarial_dust.optimizer"]:
        lgr = logging.getLogger(name)
        lgr.handlers = []
        lgr.addHandler(file_handler)
        lgr.addHandler(stream_handler)
        lgr.setLevel(logging.INFO)
        lgr.propagate = False


def create_policy(config: EnvelopeExperimentConfig):
    """Create policy via SimplerEnv's inference wrappers."""
    task_name = config.env.task_name
    policy_setup = "widowx_bridge" if task_name.startswith("widowx") else "google_robot"

    if config.env.policy_model.startswith("internvla-m1"):
        import sys
        internvla_root = "/root/InternVLA-M1"
        if internvla_root not in sys.path:
            sys.path.insert(0, internvla_root)
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


def get_image_shape(config: EnvelopeExperimentConfig) -> tuple:
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
        description="Run safe operating envelope prediction for robotic manipulation"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--video-fmt",
        type=str,
        default="mp4",
        choices=["mp4", "gif"],
        help="Format for episode animation recordings",
    )
    args = parser.parse_args()

    config = load_envelope_config(args.config)
    setup_logging(config.output_dir)

    logger = logging.getLogger(__name__)
    logger.info(f"Config: {config}")
    logger.info(f"Output dir: {config.output_dir}")
    logger.info(f"Task: {config.env.task_name}")
    logger.info(f"Policy: {config.env.policy_model}")
    logger.info(f"Occlusion types: {config.envelope.occlusion_types}")

    # Get image shape from env
    logger.info("Getting image shape from environment...")
    image_shape = get_image_shape(config)
    logger.info(f"Image shape: {image_shape}")

    # Create policy
    logger.info(f"Loading {config.env.policy_model} policy...")
    policy = create_policy(config)

    # Run envelope prediction
    predictor = SafeOperatingEnvelopePredictor(config, policy, image_shape)
    results = predictor.predict()

    # Generate visualizations
    output_dir = Path(config.output_dir)

    logger.info("Generating operating envelope plot...")
    plot_operating_envelope(results, str(output_dir / "operating_envelope.png"))

    # Generate occlusion gallery with sample image
    logger.info("Generating occlusion gallery...")
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

    plot_occlusion_gallery(
        results, image_shape, config, sample_image,
        str(output_dir / "occlusion_gallery.png"),
    )

    # Record episode animations for worst-case patterns
    logger.info("Recording worst-case episode animations...")
    from adversarial_dust.animation import record_adversarial_episode
    from adversarial_dust.envelope_predictor import make_occlusion_model

    anim_dir = output_dir / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)

    # Clean episode
    record_adversarial_episode(
        config.env, policy, None,
        str(anim_dir / f"clean.{args.video_fmt}"),
        budget_label="none",
    )

    # Worst-case per type at max tested budget
    for occ_type, type_results in results["occlusion_results"].items():
        budgets = sorted(float(b) for b in type_results.keys())
        if not budgets:
            continue
        max_budget = budgets[-1]
        br = type_results[str(max_budget)]
        params = np.array(br["adversarial_params"])
        model = make_occlusion_model(occ_type, config, image_shape, max_budget)
        alpha_mask = model.get_alpha_mask(params)

        record_adversarial_episode(
            config.env, policy, alpha_mask,
            str(anim_dir / f"worst_{occ_type}_budget_{max_budget:.2f}.{args.video_fmt}"),
            budget_label=f"{occ_type} {max_budget:.0%}",
        )

    logger.info("Envelope prediction complete!")
    logger.info(f"Results: {output_dir / 'envelope_results.json'}")
    logger.info(f"Envelope plot: {output_dir / 'operating_envelope.png'}")


if __name__ == "__main__":
    main()
