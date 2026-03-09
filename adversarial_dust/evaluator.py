"""Evaluate Octo policy under adversarial dust in SimplerEnv."""

import logging
from typing import List, Optional, Tuple

import numpy as np

from adversarial_dust.config import EnvConfig
from adversarial_dust.dust_model import AdversarialDustModel

logger = logging.getLogger(__name__)


class PolicyEvaluator:
    """Runs Octo policy episodes in SimplerEnv with dust-augmented observations."""

    def __init__(self, env_config: EnvConfig, policy, dust_model: AdversarialDustModel):
        self.env_config = env_config
        self.policy = policy
        self.dust_model = dust_model
        self.env = None

    def _ensure_env(self):
        if self.env is None:
            import simpler_env
            self.env = simpler_env.make(
                self.env_config.task_name,
                **self.env_config.make_kwargs(),
            )

    def run_episode(
        self, dust_params: Optional[np.ndarray], record: bool = False
    ) -> Tuple[bool, List[np.ndarray], List[np.ndarray]]:
        """Run a single episode, optionally recording frames.

        Args:
            dust_params: Flat dust grid params, or None for clean.
            record: If True, collect clean and dirty frames each step.

        Returns:
            (success, clean_frames, dirty_frames) — frames lists are empty if not recording.
        """
        self._ensure_env()
        from simpler_env.utils.env.observation_utils import (
            get_image_from_maniskill2_obs_dict,
        )

        obs, _ = self.env.reset()
        instruction = self.env.get_language_instruction()
        self.policy.reset(instruction)

        truncated = False
        step_count = 0
        success = False
        clean_frames: List[np.ndarray] = []
        dirty_frames: List[np.ndarray] = []

        while not truncated and step_count < self.env_config.max_episode_steps:
            image = get_image_from_maniskill2_obs_dict(self.env, obs)

            if record:
                clean_frames.append(image.copy())

            if dust_params is not None:
                dirty_image = self.dust_model.apply(image, dust_params, timestep=step_count)
                if record:
                    dirty_frames.append(dirty_image.copy())
                policy_input = dirty_image
            else:
                if record:
                    dirty_frames.append(image.copy())
                policy_input = image

            raw_action, action = self.policy.step(policy_input, instruction)

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
                self.policy.reset(instruction)

            if "success" in info and info["success"]:
                success = True

        return success, clean_frames, dirty_frames

    def evaluate(self, dust_params: Optional[np.ndarray], n_episodes: int) -> float:
        """Run n_episodes with dust applied, return mean success rate."""
        successes = []
        for ep in range(n_episodes):
            success, _, _ = self.run_episode(dust_params, record=False)
            successes.append(float(success))
            logger.debug(f"Episode {ep + 1}/{n_episodes}: success={success}")

        mean_sr = float(np.mean(successes))
        logger.info(f"Evaluation: {n_episodes} episodes, success_rate={mean_sr:.3f}")
        return mean_sr
