"""Evaluate policies under camera_occlusion.Dust at multiple severity levels.

Sweeps (model, severity) pairs and collects success rates + animations
for comparing baseline vs fine-tuned Octo robustness.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from adversarial_dust.config import DustEvalConfig, EnvConfig

logger = logging.getLogger(__name__)


def _import_dust_class():
    """Import the Dust class from camera_occlusion, handling path quirks."""
    try:
        from camera_occlusion import Dust
        return Dust
    except ImportError:
        pass

    # Fallback: try importing from the nested package directly
    import importlib
    import sys
    import os

    pkg_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "camera_occlusion",
    )
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

    from camera_occlusion.camera_noise import Dust
    return Dust


def make_dust_presets() -> dict:
    """Build severity-level presets using camera_occlusion.Dust.

    Returns:
        Dict mapping severity name to a Dust instance (or None for clean).
    """
    Dust = _import_dust_class()

    return {
        "clean": None,
        "light": Dust(
            num_specks=20,
            speck_opacity=0.4,
            num_scratches=1,
            splotch_opacity=0.1,
        ),
        "moderate": Dust(
            num_specks=40,
            speck_opacity=0.4,
            num_scratches=3,
            splotch_opacity=0.05,
        ),
        "heavy": Dust(
            num_specks=50,
            speck_opacity=0.8,
            num_scratches=5,
            splotch_opacity=0.02,
        ),
    }


class DustRobustnessEvaluator:
    """Evaluate policies under camera_occlusion.Dust at multiple severities.

    For each (model, severity) pair the evaluator runs *n_episodes* in
    SimplerEnv, applies the corresponding Dust effect to every camera
    frame before passing it to the policy, and records success/failure.
    """

    def __init__(self, env_config: EnvConfig, eval_config: DustEvalConfig):
        self.env_config = env_config
        self.eval_config = eval_config
        self._presets = None
        self.env = None

    @property
    def presets(self) -> dict:
        if self._presets is None:
            self._presets = make_dust_presets()
        return self._presets

    @presets.setter
    def presets(self, value: dict):
        self._presets = value

    def _ensure_env(self):
        if self.env is None:
            import simpler_env

            self.env = simpler_env.make(
                self.env_config.task_name,
                **self.env_config.make_kwargs(),
            )

    # ---- core evaluation ---------------------------------------------------

    def _run_episode(
        self,
        policy: Any,
        dust_effect: Any,
        record: bool = False,
    ) -> Tuple[bool, List[np.ndarray], List[np.ndarray]]:
        """Run one episode, optionally recording clean/dirty frames.

        Args:
            policy: Octo policy with ``.reset()`` / ``.step()``.
            dust_effect: A camera_occlusion.Dust instance, or None for clean.
            record: Whether to collect frame lists.

        Returns:
            (success, clean_frames, dirty_frames).
        """
        self._ensure_env()
        from simpler_env.utils.env.observation_utils import (
            get_image_from_maniskill2_obs_dict,
        )

        obs, _ = self.env.reset()
        instruction = self.env.get_language_instruction()
        policy.reset(instruction)

        clean_frames: List[np.ndarray] = []
        dirty_frames: List[np.ndarray] = []
        truncated = False
        step_count = 0
        success = False

        while not truncated and step_count < self.env_config.max_episode_steps:
            image = get_image_from_maniskill2_obs_dict(self.env, obs)

            if record:
                clean_frames.append(image.copy())

            if dust_effect is not None:
                dirty_image = dust_effect(image)
            else:
                dirty_image = image

            if record:
                dirty_frames.append(dirty_image.copy())

            raw_action, action = policy.step(dirty_image, instruction)
            action_array = np.concatenate([
                action["world_vector"],
                action["rot_axangle"],
                action["gripper"].flatten(),
            ])

            obs, reward, done, truncated, info = self.env.step(action_array)
            step_count += 1

            new_instruction = self.env.get_language_instruction()
            if new_instruction != instruction:
                instruction = new_instruction
                policy.reset(instruction)

            if "success" in info and info["success"]:
                success = True

        return success, clean_frames, dirty_frames

    def evaluate_model(
        self,
        policy: Any,
        severity_name: str,
        n_episodes: int,
    ) -> Tuple[float, List[bool]]:
        """Evaluate a single model at a single severity level.

        Args:
            policy: Octo policy wrapper.
            severity_name: Key into ``DUST_PRESETS``.
            n_episodes: Number of evaluation episodes.

        Returns:
            (mean_success_rate, per_episode_successes).
        """
        dust_effect = self.presets[severity_name]
        successes: List[bool] = []

        for ep in range(n_episodes):
            success, _, _ = self._run_episode(policy, dust_effect, record=False)
            successes.append(success)
            logger.debug(
                f"  [{severity_name}] Episode {ep + 1}/{n_episodes}: "
                f"success={success}"
            )

        sr = float(np.mean(successes))
        logger.info(
            f"  {severity_name}: {n_episodes} episodes, "
            f"success_rate={sr:.3f}"
        )
        return sr, successes

    def evaluate_all(
        self,
        models: Dict[str, Any],
        n_episodes: Optional[int] = None,
    ) -> dict:
        """Sweep all (model, severity) pairs.

        Args:
            models: Dict mapping model name → policy object.
            n_episodes: Override for episodes per evaluation cell.

        Returns:
            Nested dict: results[model_name][severity_name] = {
                "success_rate": float,
                "successes": list[bool],
            }
        """
        n_ep = n_episodes or self.eval_config.episodes_per_eval
        severities = self.eval_config.severity_levels

        results: Dict[str, Dict[str, dict]] = {}
        for model_name, policy in models.items():
            logger.info(f"Evaluating model: {model_name}")
            results[model_name] = {}
            for sev in severities:
                sr, successes = self.evaluate_model(policy, sev, n_ep)
                results[model_name][sev] = {
                    "success_rate": sr,
                    "successes": [bool(s) for s in successes],
                }

        return results

    # ---- animation recording -----------------------------------------------

    def record_animations(
        self,
        models: Dict[str, Any],
        output_dir: str,
        fps: int = 10,
    ):
        """Record one episode per (model, severity) pair as MP4.

        Args:
            models: Dict mapping model name → policy object.
            output_dir: Root directory for output videos.
            fps: Frames per second for the video.
        """
        out = Path(output_dir) / "animations"
        out.mkdir(parents=True, exist_ok=True)

        for model_name, policy in models.items():
            for sev in self.eval_config.severity_levels:
                dust_effect = self.presets[sev]
                success, clean_frames, dirty_frames = self._run_episode(
                    policy, dust_effect, record=True
                )

                if not clean_frames:
                    logger.warning(
                        f"No frames for {model_name}/{sev}, skipping."
                    )
                    continue

                # Stitch side-by-side: clean | policy view (with dust)
                stitched = []
                for c, d in zip(clean_frames, dirty_frames):
                    h, w = c.shape[:2]
                    gap = 4
                    canvas_w = w * 2 + gap
                    canvas = np.full((h + 30, canvas_w, 3), 255, dtype=np.uint8)
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    canvas[30 : 30 + h, :w] = c
                    cv2.putText(
                        canvas, "Clean", (10, 22), font, 0.7, (0, 0, 0), 2
                    )

                    x2 = w + gap
                    canvas[30 : 30 + h, x2 : x2 + w] = d
                    label = f"{model_name} / {sev}"
                    cv2.putText(
                        canvas, label, (x2 + 10, 22), font, 0.6, (0, 0, 200), 2
                    )
                    stitched.append(canvas)

                video_path = str(out / f"{model_name}_{sev}.mp4")
                vh, vw = stitched[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(video_path, fourcc, fps, (vw, vh))
                for frame in stitched:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                writer.release()

                result_str = "SUCCESS" if success else "FAIL"
                logger.info(
                    f"Saved {video_path} "
                    f"({len(stitched)} frames, {result_str})"
                )
