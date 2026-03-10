"""Record and render episode animations with dust overlays."""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from adversarial_dust.config import ExperimentConfig
from adversarial_dust.evaluator import PolicyEvaluator

logger = logging.getLogger(__name__)


def _make_dust_heatmap_overlay(
    image: np.ndarray, alpha_mask: np.ndarray, amplify: float = 5.0
) -> np.ndarray:
    """Overlay an amplified dust heatmap on the image to make dust visible.

    Args:
        image: Clean or dirty image (uint8 HxWx3).
        alpha_mask: Float alpha mask (H, W) in [0, max_opacity].
        amplify: Multiplier to amplify the alpha values for visibility.

    Returns:
        Image with colored heatmap overlay (uint8 HxWx3).
    """
    # Amplify and normalize the alpha mask for visibility
    amplified = np.clip(alpha_mask * amplify, 0.0, 1.0)

    # Apply "hot" colormap: low=transparent, high=bright red/yellow
    heatmap_uint8 = (amplified * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend: overlay heatmap where dust exists, keep original where it doesn't
    overlay_alpha = amplified[:, :, np.newaxis]
    base = image.astype(np.float32)
    heat = heatmap_colored.astype(np.float32)
    blended = base * (1.0 - overlay_alpha * 0.6) + heat * (overlay_alpha * 0.6)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _stitch_panels(
    clean: np.ndarray,
    dirty: np.ndarray,
    highlighted: Optional[np.ndarray] = None,
    label_clean: str = "Clean",
    label_dirty: str = "Policy View",
    label_highlight: str = "Dust Highlighted",
) -> np.ndarray:
    """Stitch two or three frames side-by-side with labels."""
    h, w = clean.shape[:2]
    gap = 4
    n_panels = 3 if highlighted is not None else 2
    canvas_w = w * n_panels + gap * (n_panels - 1)
    canvas = np.full((h + 30, canvas_w, 3), 255, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Panel 1: Clean
    canvas[30 : 30 + h, :w] = clean
    cv2.putText(canvas, label_clean, (10, 22), font, 0.7, (0, 0, 0), 2)

    # Panel 2: Dirty (what policy sees)
    x2 = w + gap
    canvas[30 : 30 + h, x2 : x2 + w] = dirty
    cv2.putText(canvas, label_dirty, (x2 + 10, 22), font, 0.7, (0, 0, 200), 2)

    # Panel 3: Highlighted (optional)
    if highlighted is not None:
        x3 = 2 * (w + gap)
        canvas[30 : 30 + h, x3 : x3 + w] = highlighted
        cv2.putText(canvas, label_highlight, (x3 + 10, 22), font, 0.7, (200, 100, 0), 2)

    return canvas


def record_episode_video(
    evaluator: PolicyEvaluator,
    dust_params: Optional[np.ndarray],
    output_path: str,
    budget_label: str = "",
    fps: int = 10,
):
    """Record a single episode as MP4 with clean, policy view, and dust highlighted panels.

    Args:
        evaluator: PolicyEvaluator with env and policy loaded.
        dust_params: Dust parameters, or None for clean-only.
        output_path: Path for output .mp4 file.
        budget_label: Label string for the dirty panel header.
        fps: Frames per second for the video.
    """
    logger.info(f"Recording episode to {output_path}")
    success, clean_frames, dirty_frames = evaluator.run_episode(dust_params, record=True)

    if not clean_frames:
        logger.warning("No frames recorded, skipping video.")
        return success

    dirty_label = f"Policy View (budget={budget_label})" if budget_label else "Policy View"
    highlight_label = f"Dust Highlighted ({budget_label})" if budget_label else "Dust Highlighted"

    stitched = []
    dust_model = evaluator.dust_model
    for t, (c, d) in enumerate(zip(clean_frames, dirty_frames)):
        if dust_params is not None:
            # Compute alpha mask per-frame (supports dynamic blob models)
            alpha_mask = dust_model.get_alpha_mask(dust_params, timestep=t)
            highlighted = _make_dust_heatmap_overlay(c, alpha_mask)
            frame = _stitch_panels(c, d, highlighted, "Clean", dirty_label, highlight_label)
        else:
            frame = _stitch_panels(c, d, label_clean="Clean", label_dirty="Clean")
        stitched.append(frame)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    h, w = stitched[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in stitched:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    logger.info(f"Saved video: {output_path} ({len(stitched)} frames, success={success})")
    return success


def record_episode_gif(
    evaluator: PolicyEvaluator,
    dust_params: Optional[np.ndarray],
    output_path: str,
    budget_label: str = "",
    fps: int = 10,
    max_frames: int = 200,
):
    """Record a single episode as GIF with clean, policy view, and dust highlighted panels.

    Uses imageio for GIF writing. Falls back to MP4 if imageio unavailable.
    """
    logger.info(f"Recording episode GIF to {output_path}")
    success, clean_frames, dirty_frames = evaluator.run_episode(dust_params, record=True)

    if not clean_frames:
        logger.warning("No frames recorded, skipping GIF.")
        return success

    dirty_label = f"Policy View (budget={budget_label})" if budget_label else "Policy View"
    highlight_label = f"Dust Highlighted ({budget_label})" if budget_label else "Dust Highlighted"

    stitched = []
    dust_model = evaluator.dust_model
    for t, (c, d) in enumerate(zip(clean_frames, dirty_frames)):
        if dust_params is not None:
            # Compute alpha mask per-frame (supports dynamic blob models)
            alpha_mask = dust_model.get_alpha_mask(dust_params, timestep=t)
            highlighted = _make_dust_heatmap_overlay(c, alpha_mask)
            frame = _stitch_panels(c, d, highlighted, "Clean", dirty_label, highlight_label)
        else:
            frame = _stitch_panels(c, d, label_clean="Clean", label_dirty="Clean")
        stitched.append(frame)

    # Subsample if too many frames
    if len(stitched) > max_frames:
        step = len(stitched) // max_frames
        stitched = stitched[::step][:max_frames]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio.v3 as iio
        duration_ms = int(1000 / fps)
        iio.imwrite(output_path, stitched, duration=duration_ms, loop=0)
    except ImportError:
        # Fallback: save as mp4
        mp4_path = str(output_path).replace(".gif", ".mp4")
        logger.warning(f"imageio not available, saving as {mp4_path} instead")
        h, w = stitched[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))
        for frame in stitched:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

    logger.info(f"Saved: {output_path} ({len(stitched)} frames, success={success})")
    return success


def record_all_budget_animations(
    config: ExperimentConfig,
    policy,
    image_shape: tuple,
    results: dict,
    output_dir: str,
    fmt: str = "mp4",
):
    """Record one episode per budget level using the adversarial dust pattern found.

    Args:
        config: Full experiment config.
        policy: Loaded Octo policy.
        image_shape: (H, W, C).
        results: Output from BudgetSweep.run().
        output_dir: Directory for output videos.
        fmt: "mp4" or "gif".
    """
    out = Path(output_dir) / "animations"
    out.mkdir(parents=True, exist_ok=True)

    from adversarial_dust.budget_sweep import make_dust_model

    budget_results = results["budget_results"]

    # Record clean episode first
    dust_model_clean = make_dust_model(config, image_shape, budget_level=0.0)
    evaluator_clean = PolicyEvaluator(config.env, policy, dust_model_clean)
    record_fn = record_episode_gif if fmt == "gif" else record_episode_video
    record_fn(evaluator_clean, None, str(out / f"clean.{fmt}"), budget_label="none")

    # Record one episode per budget level with adversarial dust
    for budget_str, br in sorted(budget_results.items(), key=lambda x: float(x[0])):
        budget = float(budget_str)
        params = np.array(br["adversarial_params"])
        dust_model = make_dust_model(config, image_shape, budget)
        evaluator = PolicyEvaluator(config.env, policy, dust_model)
        record_fn(
            evaluator,
            params,
            str(out / f"adversarial_budget_{budget:.2f}.{fmt}"),
            budget_label=f"{budget:.0%}",
        )

    logger.info(f"All animations saved to {out}")


def _render_temporal_mask_np(
    blob_params: np.ndarray,
    H: int,
    W: int,
    t_normalized: float,
    sharpness: float = 20.0,
) -> np.ndarray:
    """Render a temporal blob mask at a specific timestep (numpy).

    Args:
        blob_params: (K, 7) blob parameters.
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

    gy = np.linspace(0, 1, H, dtype=np.float32)
    gx = np.linspace(0, 1, W, dtype=np.float32)
    gx, gy = np.meshgrid(gx, gy)

    mask = np.zeros((H, W), dtype=np.float32)

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    for i in range(K):
        cx, cy, sx, sy, opacity, t_center, t_width = blob_params[i]
        left = _sigmoid(sharpness * (t_normalized - (t_center - t_width / 2)))
        right = _sigmoid(sharpness * ((t_center + t_width / 2) - t_normalized))
        gate = left * right
        gauss = opacity * gate * np.exp(
            -0.5 * (((gx - cx) / max(sx, 1e-6)) ** 2
                     + ((gy - cy) / max(sy, 1e-6)) ** 2)
        )
        mask = np.maximum(mask, gauss)

    return mask


def record_adversarial_episode(
    env_config,
    policy,
    mask_or_params: Optional[np.ndarray],
    output_path: str,
    budget_label: str = "",
    dust_color: tuple = (180, 160, 140),
    fps: int = 10,
    two_panel: bool = False,
    single_panel: bool = True,
    temporal_sharpness: Optional[float] = None,
) -> bool:
    """Record a single episode showing what the policy sees under occlusion.

    By default (``single_panel=True``), renders a single full-size frame
    showing the occluded image with the occlusion pattern highlighted so
    the viewer can clearly see the noise. A small success/fail badge is
    overlaid at the end.

    Set ``single_panel=False`` for the legacy multi-panel layout.

    Args:
        env_config: EnvConfig with task_name, max_episode_steps, etc.
        policy: Loaded Octo policy (with .reset() and .step()).
        mask_or_params: (H, W) float array, (K, 7) temporal blob params, or None.
        output_path: Path for output .mp4 file.
        budget_label: Label string for the dirty panel header.
        dust_color: RGB tuple for dust blending.
        fps: Frames per second for the video.
        two_panel: If True, render only clean | dirty (no heatmap). Ignored when single_panel=True.
        single_panel: If True (default), render one frame with highlighted occlusion.
        temporal_sharpness: If provided with (K, 7) params, enables temporal mode.

    Returns:
        Whether the episode was successful.
    """
    import simpler_env
    from simpler_env.utils.env.observation_utils import (
        get_image_from_maniskill2_obs_dict,
    )

    logger.info(f"Recording adversarial episode to {output_path}")

    # Detect temporal mode
    is_temporal = (
        mask_or_params is not None
        and mask_or_params.ndim == 2
        and mask_or_params.shape[1] == 7
        and temporal_sharpness is not None
    )

    env = simpler_env.make(
        env_config.task_name,
        **env_config.make_kwargs(),
    )
    obs, _ = env.reset()
    instruction = env.get_language_instruction()
    policy.reset(instruction)

    dust_f = np.array(dust_color, dtype=np.float32) / 255.0
    T = env_config.max_episode_steps

    clean_frames = []
    dirty_frames = []
    alpha_masks = []  # per-step masks for temporal mode
    truncated = False
    step_count = 0
    success = False

    while not truncated and step_count < T:
        image = get_image_from_maniskill2_obs_dict(env, obs)
        clean_frames.append(image.copy())

        if mask_or_params is not None:
            if is_temporal:
                t_norm = step_count / max(T - 1, 1)
                H, W = image.shape[:2]
                alpha_mask = _render_temporal_mask_np(
                    mask_or_params, H, W, t_norm, temporal_sharpness
                )
            else:
                alpha_mask = mask_or_params

            alpha_masks.append(alpha_mask)

            img_f = image.astype(np.float32) / 255.0
            a3d = alpha_mask[:, :, np.newaxis]
            dc = dust_f[np.newaxis, np.newaxis, :]
            blended = dc * a3d + img_f * (1.0 - a3d)
            policy_input = np.clip(blended * 255, 0, 255).astype(np.uint8)
        else:
            alpha_masks.append(None)
            policy_input = image

        dirty_frames.append(policy_input.copy())

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

    env.close()

    if not clean_frames:
        logger.warning("No frames recorded, skipping video.")
        return success

    dirty_label = f"Policy View (budget={budget_label})" if budget_label else "Policy View"

    stitched = []
    for t, (c, d) in enumerate(zip(clean_frames, dirty_frames)):
        am = alpha_masks[t] if t < len(alpha_masks) else None
        has_mask = am is not None

        if single_panel:
            # Single frame: show exactly what the model sees (raw occluded image)
            frame = d.copy()
            # Add label bar at top
            h, w = frame.shape[:2]
            canvas = np.full((h + 30, w, 3), 255, dtype=np.uint8)
            canvas[30:, :] = frame
            label = budget_label if budget_label else "Clean"
            cv2.putText(canvas, label, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            frame = canvas
        elif has_mask and not two_panel:
            highlight_label = f"Dust Highlighted ({budget_label})" if budget_label else "Dust Highlighted"
            highlighted = _make_dust_heatmap_overlay(c, am)
            frame = _stitch_panels(c, d, highlighted, "Clean", dirty_label, highlight_label)
        else:
            frame = _stitch_panels(
                c, d,
                label_clean="Clean",
                label_dirty=dirty_label if has_mask else "Clean",
            )

        # Add timestep indicator for temporal mode
        if is_temporal:
            t_norm = t / max(len(clean_frames) - 1, 1)
            cv2.putText(
                frame, f"t={t_norm:.2f}",
                (frame.shape[1] - 120, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
            )

        stitched.append(frame)

    # Stamp success/fail on last 10 frames
    badge_text = "SUCCESS" if success else "FAIL"
    badge_color = (0, 180, 0) if success else (220, 0, 0)
    n_badge = min(10, len(stitched))
    for i in range(len(stitched) - n_badge, len(stitched)):
        fh, fw = stitched[i].shape[:2]
        cv2.putText(stitched[i], badge_text, (fw - 160, fh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, badge_color, 3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    h, w = stitched[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in stitched:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    mode = 'temporal' if is_temporal else ('dust' if mask_or_params is not None else 'clean')
    logger.info(
        f"Saved video: {output_path} ({len(stitched)} frames, "
        f"success={success}, mask={mode})"
    )
    return success
