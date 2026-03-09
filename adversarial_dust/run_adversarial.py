"""CLI entry point for adversarial zero-sum dust training.

Usage:
    python -m adversarial_dust.run_adversarial --config configs/adversarial_cube.yaml
    python -m adversarial_dust.run_adversarial --config configs/adversarial_quick_test.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from adversarial_dust.config import load_adversarial_config
from adversarial_dust.dust_generator import DustGenerator
from adversarial_dust.adversarial_trainer import AdversarialTrainer, MaskApplicator
from adversarial_dust.blob_generator import BlobGenerator

logger = logging.getLogger(__name__)


def render_temporal_mask_np(
    blob_params: np.ndarray,
    H: int,
    W: int,
    t_normalized: float,
    sharpness: float = 20.0,
) -> np.ndarray:
    """Render a temporal blob mask at a specific timestep (numpy).

    Args:
        blob_params: (K, 7) blob parameters [cx, cy, sx, sy, opacity, t_center, t_width].
        H: Image height.
        W: Image width.
        t_normalized: Normalized timestep in [0, 1].
        sharpness: Temporal gate sharpness.

    Returns:
        (H, W) alpha mask as float32.
    """
    K = blob_params.shape[0]
    if K == 0:
        return np.zeros((H, W), dtype=np.float32)

    # Build coordinate grids
    gy = np.linspace(0, 1, H, dtype=np.float32)
    gx = np.linspace(0, 1, W, dtype=np.float32)
    gx, gy = np.meshgrid(gx, gy)  # (H, W) each

    mask = np.zeros((H, W), dtype=np.float32)

    for i in range(K):
        cx, cy, sx, sy, opacity, t_center, t_width = blob_params[i]

        # Temporal gate: product of two sigmoids
        def _sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

        left = _sigmoid(sharpness * (t_normalized - (t_center - t_width / 2)))
        right = _sigmoid(sharpness * ((t_center + t_width / 2) - t_normalized))
        gate = left * right

        # Gaussian blob
        gauss = opacity * gate * np.exp(
            -0.5 * (((gx - cx) / max(sx, 1e-6)) ** 2
                     + ((gy - cy) / max(sy, 1e-6)) ** 2)
        )
        mask = np.maximum(mask, gauss)

    return mask


def setup_logging(output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(output_dir) / "training.log"),
        ],
    )


def create_evaluate_fn(
    env_config,
    mask_applicator,
    policy,
    dust_color=(180, 160, 140),
    temporal_config=None,
):
    """Create the black-box evaluation function for the trainer.

    Returns a function: evaluate(alpha_mask_or_blob_params_or_None, n_episodes) -> success_rate

    When ``temporal_config`` is provided and temporal is enabled, the evaluate
    function accepts (K, 7) blob_params and renders per-step temporal masks.
    """
    dust_f = np.array(dust_color, dtype=np.float32) / 255.0
    is_temporal = temporal_config is not None and temporal_config.temporal
    sharpness = temporal_config.temporal_sharpness if is_temporal else 20.0

    def evaluate(mask_or_params, n_episodes):
        """Evaluate policy under a given mask or temporal blob params.

        Args:
            mask_or_params: (H, W) numpy array for static mask,
                (K, 7) numpy array for temporal blob_params,
                or None for clean eval.
            n_episodes: Number of episodes to run.

        Returns:
            Mean success rate.
        """
        import simpler_env
        from simpler_env.utils.env.observation_utils import (
            get_image_from_maniskill2_obs_dict,
        )

        env = simpler_env.make(
            env_config.task_name,
            **env_config.make_kwargs(),
        )
        successes = []

        # Detect temporal blob_params: shape (K, 7) where K is small
        use_temporal = (
            is_temporal
            and mask_or_params is not None
            and mask_or_params.ndim == 2
            and mask_or_params.shape[1] == 7
        )

        for ep in range(n_episodes):
            obs, _ = env.reset()
            instruction = env.get_language_instruction()
            policy.reset(instruction)

            truncated = False
            step_count = 0
            success = False
            T = env_config.max_episode_steps

            while (
                not truncated
                and step_count < T
            ):
                image = get_image_from_maniskill2_obs_dict(env, obs)

                if mask_or_params is not None:
                    if use_temporal:
                        # Render temporal mask at this timestep
                        t_norm = step_count / max(T - 1, 1)
                        H, W = image.shape[:2]
                        alpha_mask = render_temporal_mask_np(
                            mask_or_params, H, W, t_norm, sharpness
                        )
                    else:
                        alpha_mask = mask_or_params

                    # Apply mask via alpha blending
                    img_f = image.astype(np.float32) / 255.0
                    a3d = alpha_mask[:, :, np.newaxis]
                    dc = dust_f[np.newaxis, np.newaxis, :]
                    blended = dc * a3d + img_f * (1.0 - a3d)
                    policy_input = np.clip(blended * 255, 0, 255).astype(np.uint8)
                else:
                    policy_input = image

                raw_action, action = policy.step(policy_input, instruction)
                action_array = np.concatenate([
                    action["world_vector"],
                    action["rot_axangle"],
                    action["gripper"].flatten(),
                ])

                obs, reward, done, truncated, info = env.step(action_array)
                step_count += 1

                new_instruction = env.get_language_instruction()
                if new_instruction != instruction:
                    instruction = new_instruction
                    policy.reset(instruction)

                if "success" in info and info["success"]:
                    success = True

            successes.append(float(success))

        env.close()
        sr = float(np.mean(successes))
        mode = 'temporal' if use_temporal else ('dust' if mask_or_params is not None else 'clean')
        logger.info(f"  Eval: {n_episodes} eps, mask={mode}, sr={sr:.3f}")
        return sr

    return evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Adversarial zero-sum dust training"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--no-defender",
        action="store_true",
        help="Skip defender (Octo fine-tuning), train attacker only",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to fine-tuned Octo checkpoint (replaces baseline model)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device for generator",
    )
    args = parser.parse_args()

    config = load_adversarial_config(args.config)
    setup_logging(config.output_dir)

    logger.info(f"Config loaded from {args.config}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Task: {config.env.task_name}")
    logger.info(f"Rounds: {config.training.n_rounds}")
    logger.info(f"Budgets: {config.training.budget_levels}")
    logger.info(f"Device: {args.device}")

    # Load policy (Octo, OpenVLA, or InternVLA-M1)
    task_name = config.env.task_name
    policy_setup = "widowx_bridge" if task_name.startswith("widowx") else "google_robot"
    logger.info(f"Detected policy_setup: {policy_setup}")

    if config.env.policy_model.startswith("internvla-m1"):
        logger.info(f"Loading InternVLA-M1 policy (server port {config.env.internvla_m1_port})...")
        from examples.SimplerEnv.model2simpler_interface import M1Inference

        policy = M1Inference(
            policy_ckpt_path=config.env.internvla_m1_ckpt,
            policy_setup=policy_setup,
            port=config.env.internvla_m1_port,
            action_scale=1.0,
            cfg_scale=1.5,
        )
    elif config.env.policy_model.startswith("openvla"):
        logger.info(f"Loading OpenVLA policy ({config.env.policy_model})...")
        from simpler_env.policies.openvla.openvla_model import OpenVLAInference

        # Map shorthand to HuggingFace path
        model_path = config.env.policy_model
        if model_path == "openvla":
            model_path = "openvla/openvla-7b"
        elif not model_path.startswith("openvla/"):
            model_path = f"openvla/{model_path}"

        policy = OpenVLAInference(
            saved_model_path=model_path,
            policy_setup=policy_setup,
        )
    else:
        logger.info(f"Loading Octo policy ({config.env.policy_model})...")
        from simpler_env.policies.octo.octo_model import OctoInference

        policy = OctoInference(
            model_type=config.env.policy_model,
            policy_setup=policy_setup,
        )

        if args.checkpoint:
            logger.info(f"Loading fine-tuned checkpoint from {args.checkpoint}")
            from octo.model.octo_model import OctoModel
            new_model = OctoModel.load_pretrained(args.checkpoint)
            policy.model = new_model
            dataset_id = policy.dataset_id if hasattr(policy, "dataset_id") else list(new_model.dataset_statistics.keys())[0]
            if dataset_id in new_model.dataset_statistics:
                stats = new_model.dataset_statistics[dataset_id]["action"]
                policy.action_mean = stats["mean"]
                policy.action_std = stats["std"]
                logger.info(f"Updated action stats from checkpoint (dataset: {dataset_id})")
            logger.info("Fine-tuned model loaded successfully")

    # Get a reference image for generator conditioning (determines image shape)
    import simpler_env
    from simpler_env.utils.env.observation_utils import (
        get_image_from_maniskill2_obs_dict,
    )

    env = simpler_env.make(
        config.env.task_name,
        **config.env.make_kwargs(),
    )
    obs, _ = env.reset()
    reference_image = get_image_from_maniskill2_obs_dict(env, obs)
    env.close()
    image_shape = reference_image.shape
    logger.info(f"Reference image shape: {image_shape}")

    # Create generator (use actual image shape from env)
    use_blob = config.blob_generator is not None
    if use_blob:
        generator = BlobGenerator(config.blob_generator, image_shape)
        gen_config = config.blob_generator
        logger.info("Using BlobGenerator (Gaussian blob mode)")
    else:
        generator = DustGenerator(config.generator, image_shape)
        gen_config = config.generator
        logger.info("Using DustGenerator (per-pixel mode)")

    generator = generator.to(args.device)
    n_params = sum(p.numel() for p in generator.parameters())
    logger.info(f"Generator created: {n_params} parameters")

    # Create mask applicator (uses generator config for dust_color, etc.)
    mask_applicator = MaskApplicator(config.generator, image_shape)

    # Create evaluation function
    temporal_config = config.blob_generator if (use_blob and config.blob_generator.temporal) else None
    evaluate_fn = create_evaluate_fn(
        config.env, mask_applicator, policy,
        dust_color=gen_config.dust_color,
        temporal_config=temporal_config,
    )

    # Create defender (optional)
    defender = None
    if not args.no_defender:
        from adversarial_dust.octo_finetuner import OctoGRPO

        defender = OctoGRPO(
            defender_config=config.defender,
            env_config=config.env,
            mask_applicator=mask_applicator,
        )

    # Create animation recording callback
    from adversarial_dust.animation import record_adversarial_episode

    anim_dir = Path(config.output_dir) / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)

    is_temporal = temporal_config is not None

    def record_fn(mask_or_params, round_idx, budget_level):
        if is_temporal:
            budget_str = f"{budget_level:.0%}".replace("%", "pct") + "_areatime"
            budget_label = f"{budget_level:.0%} area-time"
        elif use_blob:
            budget_str = f"{int(budget_level)}blobs"
            budget_label = f"{int(budget_level)} blobs"
        else:
            budget_str = f"{budget_level:.0%}".replace("%", "pct")
            budget_label = f"{budget_level:.0%}"
        path = str(anim_dir / f"round{round_idx + 1}_budget{budget_str}.mp4")
        record_adversarial_episode(
            config.env, policy, mask_or_params, path,
            budget_label=budget_label,
            dust_color=gen_config.dust_color,
            two_panel=True,
            temporal_sharpness=gen_config.temporal_sharpness if is_temporal else None,
        )

    # Create trainer and run
    trainer = AdversarialTrainer(
        config=config,
        generator=generator,
        mask_applicator=mask_applicator,
        evaluate_fn=evaluate_fn,
        defender=defender,
        reference_image=reference_image,
        record_fn=record_fn,
    )

    # Record clean baseline animation before training starts
    logger.info("Recording clean baseline animation...")
    try:
        record_adversarial_episode(
            config.env, policy, None,
            str(anim_dir / "clean_baseline.mp4"),
            budget_label="clean",
            dust_color=gen_config.dust_color,
            two_panel=True,
        )
    except Exception as e:
        logger.warning(f"Failed to record clean baseline animation: {e}")

    logger.info("Starting adversarial training...")
    all_metrics = trainer.train()

    # Save results
    output_dir = Path(config.output_dir)
    history = trainer.get_history_dict()

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save generator checkpoint
    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "config": config,
        },
        output_dir / "generator_checkpoint.pt",
    )

    # Plot training curves
    from adversarial_dust.visualization import plot_adversarial_training_curves

    plot_adversarial_training_curves(
        history, str(output_dir / "training_curves.png")
    )

    logger.info(f"Training complete. Results saved to {output_dir}")

    # Print summary
    logger.info("\n=== Training Summary ===")
    for m in all_metrics:
        if is_temporal:
            budget_str = f"{m.budget_level:.0%} area-time"
        elif use_blob:
            budget_str = f"{int(m.budget_level)} blobs"
        else:
            budget_str = f"{m.budget_level:.0%}"
        logger.info(
            f"  Round {m.round_idx + 1}, budget={budget_str}: "
            f"adv_sr={m.adversarial_eval_sr:.3f}, "
            f"clean_sr={m.clean_eval_sr:.3f}"
        )


if __name__ == "__main__":
    main()
