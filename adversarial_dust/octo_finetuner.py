"""Octo policy fine-tuning utilities.

Contains:
- TrajectoryCollector / OctoGRPO — adversarial GRPO fine-tuning (original).
- DemoCollector — collect clean successful demonstrations for BC.
- OctoBCTrainer — behavioral-cloning fine-tuning using Octo's native
  diffusion loss (freeze backbone, train action head only).

Requires JAX, Octo, and SimplerEnv (GPU environment).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from adversarial_dust.config import (
    DefenderConfig,
    DemoCollectionConfig,
    EnvConfig,
    FinetuningConfig,
    GeneratorConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """A single episode trajectory."""

    observations: List[np.ndarray]  # list of (H, W, 3) images
    actions: List[np.ndarray]  # list of action arrays
    reward: float  # final episode reward (1.0 for success, 0.0 for failure)
    dust_mask: Optional[np.ndarray] = None  # (H, W) alpha mask used


class TrajectoryCollector:
    """Collects trajectories by running the Octo policy under adversarial dust.

    This is the online data collection component — no pre-collected demos needed.
    The policy learns from its own experience under adversarial conditions.
    """

    def __init__(
        self,
        env_config: EnvConfig,
        mask_applicator: Any,  # MaskApplicator instance
    ):
        self.env_config = env_config
        self.mask_applicator = mask_applicator
        self.env = None

    def _ensure_env(self):
        if self.env is None:
            import simpler_env

            self.env = simpler_env.make(
                self.env_config.task_name,
                **self.env_config.make_kwargs(),
            )

    def collect_trajectory(
        self,
        policy: Any,
        dust_mask: Optional[np.ndarray],
        budget_level: float,
    ) -> Trajectory:
        """Run one episode, collecting the full trajectory.

        Args:
            policy: Octo policy wrapper with .reset() and .step().
            dust_mask: (H, W) alpha mask, or None for clean.
            budget_level: Coverage budget for mask projection.

        Returns:
            Trajectory with observations, actions, and reward.
        """
        self._ensure_env()
        from simpler_env.utils.env.observation_utils import (
            get_image_from_maniskill2_obs_dict,
        )

        obs, _ = self.env.reset()
        instruction = self.env.get_language_instruction()
        policy.reset(instruction)

        observations = []
        actions = []
        done = False
        truncated = False
        step_count = 0
        success = False

        while (
            not done
            and not truncated
            and step_count < self.env_config.max_episode_steps
        ):
            image = get_image_from_maniskill2_obs_dict(self.env, obs)

            if dust_mask is not None:
                dirty_image = self.mask_applicator.apply(
                    image, dust_mask, budget_level
                )
                policy_input = dirty_image
            else:
                policy_input = image

            observations.append(policy_input.copy())

            raw_action, action = policy.step(policy_input, instruction)

            action_array = np.concatenate(
                [
                    action["world_vector"],
                    action["rot_axangle"],
                    action["gripper"].flatten(),
                ]
            )
            actions.append(action_array.copy())

            obs, reward, done, truncated, info = self.env.step(action_array)
            step_count += 1

            new_instruction = self.env.get_language_instruction()
            if new_instruction != instruction:
                instruction = new_instruction
                policy.reset(instruction)

            if "success" in info and info["success"]:
                success = True

        return Trajectory(
            observations=observations,
            actions=actions,
            reward=float(success),
            dust_mask=dust_mask,
        )

    def collect_rollouts(
        self,
        policy: Any,
        dust_mask: Optional[np.ndarray],
        budget_level: float,
        n_episodes: int,
    ) -> List[Trajectory]:
        """Collect multiple trajectories.

        Args:
            policy: Octo policy wrapper.
            dust_mask: (H, W) alpha mask, or None for clean.
            budget_level: Coverage budget.
            n_episodes: Number of episodes to collect.

        Returns:
            List of Trajectory objects.
        """
        trajectories = []
        for i in range(n_episodes):
            traj = self.collect_trajectory(policy, dust_mask, budget_level)
            trajectories.append(traj)
            logger.debug(
                f"  Collected trajectory {i + 1}/{n_episodes}: "
                f"reward={traj.reward}, steps={len(traj.actions)}"
            )
        return trajectories


class OctoGRPO:
    """Group Relative Policy Optimization for Octo.

    Fine-tunes the Octo action head (frozen backbone) using online RL:
    1. Collect K episodes under adversarial dust
    2. Compute group-relative advantage: A_k = reward_k - mean(rewards)
    3. Weight Octo's diffusion loss by advantage
    4. Take gradient steps to reinforce successful trajectories

    This uses Octo's model internals:
        OctoModel.load_pretrained() → TrainState → weighted loss_fn →
        jax.value_and_grad → apply_gradients
    """

    def __init__(
        self,
        defender_config: DefenderConfig,
        env_config: EnvConfig,
        mask_applicator: Any,
    ):
        self.config = defender_config
        self.env_config = env_config
        self.mask_applicator = mask_applicator

        self.model = None
        self.train_state = None
        self.policy = None
        self._initialized = False

    def initialize(self):
        """Load Octo model and create TrainState for fine-tuning.

        Must be called on GPU before training.
        """
        if self._initialized:
            return

        import jax
        import jax.numpy as jnp
        from octo.model.octo_model import OctoModel

        logger.info("Loading Octo model for GRPO fine-tuning...")
        self.model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")

        # Create TrainState with frozen backbone
        import optax

        # Freeze everything except the action head
        if self.config.freeze_pattern == "head_only":
            # Only train the action head parameters
            partition_fn = self._make_head_only_partition()
        else:
            partition_fn = None

        tx = optax.adam(self.config.learning_rate)
        if partition_fn is not None:
            tx = optax.masked(tx, partition_fn)

        from flax.training.train_state import TrainState

        self.train_state = TrainState.create(
            apply_fn=self.model.model.apply,
            params=self.model.params,
            tx=tx,
        )

        self._initialized = True
        logger.info("Octo GRPO initialized successfully")

    def _make_head_only_partition(self) -> Callable:
        """Create a mask that is True only for action head parameters."""
        import jax

        def partition_fn(params):
            flat = jax.tree.leaves_with_path(params)
            mask = {}

            def _make_mask(path, leaf):
                # Only train params with "heads" in their path
                path_str = "/".join(str(k) for k in path)
                return "heads" in path_str

            return jax.tree.map_with_path(_make_mask, params)

        return partition_fn

    def compute_advantages(
        self, trajectories: List[Trajectory]
    ) -> np.ndarray:
        """Compute group-relative advantages.

        A_k = reward_k - mean(rewards)

        Successful episodes get positive advantage (reinforced),
        failures get negative advantage (suppressed).
        """
        rewards = np.array([t.reward for t in trajectories])
        mean_reward = rewards.mean()
        advantages = rewards - mean_reward

        # Normalize if there's variance
        std = advantages.std()
        if std > 1e-8:
            advantages = advantages / std

        return advantages

    def train_step(
        self,
        trajectories: List[Trajectory],
        advantages: np.ndarray,
    ) -> float:
        """One GRPO gradient step using weighted diffusion loss.

        Args:
            trajectories: Collected episode trajectories.
            advantages: Per-episode advantages.

        Returns:
            Mean weighted loss value.
        """
        import jax
        import jax.numpy as jnp

        # Prepare batch: sample trajectories weighted by advantage
        batch = self._prepare_batch(trajectories, advantages)
        if batch is None:
            return 0.0

        def loss_fn(params):
            # Use Octo's built-in loss computation
            output = self.train_state.apply_fn(
                {"params": params},
                batch["observations"],
                batch["tasks"],
                batch["actions"],
                train=True,
                rngs={"dropout": jax.random.PRNGKey(0)},
            )
            # Weight the loss by advantages
            loss = output["loss"]
            weighted_loss = loss * batch["weights"]
            return weighted_loss.mean()

        loss_val, grads = jax.value_and_grad(loss_fn)(self.train_state.params)
        self.train_state = self.train_state.apply_gradients(grads=grads)

        # Update model params reference
        self.model = self.model.replace(params=self.train_state.params)

        return float(loss_val)

    def _prepare_batch(
        self,
        trajectories: List[Trajectory],
        advantages: np.ndarray,
    ) -> Optional[Dict]:
        """Prepare a training batch from trajectories.

        Samples trajectory steps and assigns advantage weights.
        Returns None if no valid data.
        """
        import jax.numpy as jnp

        # Filter to trajectories with at least one action
        valid = [
            (t, a)
            for t, a in zip(trajectories, advantages)
            if len(t.actions) > 0
        ]
        if not valid:
            return None

        # For simplicity, take fixed-length windows from each trajectory
        # and weight by the trajectory's advantage
        batch_obs = []
        batch_actions = []
        batch_weights = []

        for traj, adv in valid:
            n_steps = min(len(traj.actions), len(traj.observations))
            for step_idx in range(n_steps):
                batch_obs.append(traj.observations[step_idx])
                batch_actions.append(traj.actions[step_idx])
                batch_weights.append(adv)

        if not batch_obs:
            return None

        # Subsample to batch size
        n_total = len(batch_obs)
        indices = np.random.choice(
            n_total,
            size=min(n_total, self.config.grpo_batch_size),
            replace=False,
        )

        obs_batch = np.stack([batch_obs[i] for i in indices])
        action_batch = np.stack([batch_actions[i] for i in indices])
        weight_batch = np.array([batch_weights[i] for i in indices])

        return {
            "observations": jnp.array(obs_batch),
            "actions": jnp.array(action_batch),
            "weights": jnp.array(weight_batch),
            "tasks": None,  # Will be set from model's task encoding
        }

    def train_round(
        self,
        generator: Any,
        budget_level: float,
    ):
        """Run one defender training round.

        1. Generate dust mask from current generator
        2. Collect rollouts under that dust
        3. Compute group-relative advantages
        4. Run GRPO gradient steps

        Args:
            generator: DustGenerator instance.
            budget_level: Current budget level.
        """
        import torch

        self.initialize()

        # Generate dust mask from current generator
        generator.eval()
        with torch.no_grad():
            # Use a dummy image for conditioning (or could use first obs)
            dummy = torch.zeros(1, 3, 512, 640)
            mask = generator.generate_mask(dummy)
            mask_np = generator.mask_to_numpy(mask)[0]  # (H, W)

        # Collect rollouts
        collector = TrajectoryCollector(self.env_config, self.mask_applicator)
        trajectories = collector.collect_rollouts(
            policy=self.policy,
            dust_mask=mask_np,
            budget_level=budget_level,
            n_episodes=self.config.grpo_episodes_per_round,
        )

        rewards = [t.reward for t in trajectories]
        mean_sr = np.mean(rewards)
        logger.info(
            f"  GRPO rollouts: {len(trajectories)} episodes, "
            f"success_rate={mean_sr:.3f}"
        )

        # Compute advantages
        advantages = self.compute_advantages(trajectories)

        # Run GRPO update steps
        losses = []
        for step in range(self.config.grpo_update_steps):
            loss = self.train_step(trajectories, advantages)
            losses.append(loss)

        mean_loss = np.mean(losses) if losses else 0.0
        logger.info(
            f"  GRPO update: {self.config.grpo_update_steps} steps, "
            f"mean_loss={mean_loss:.4f}"
        )


# ---------------------------------------------------------------------------
# Dust-robustness: DemoCollector + OctoBCTrainer
# ---------------------------------------------------------------------------


class DemoCollector:
    """Collect successful clean demonstrations for behavioral-cloning fine-tuning.

    Runs the baseline Octo policy in SimplerEnv (no dust) and keeps only
    episodes that ended in success.  Each successful trajectory stores
    (observations, actions, task_text) for later BC training.
    """

    def __init__(self, env_config: EnvConfig):
        self.env_config = env_config
        self.env = None

    def _ensure_env(self):
        if self.env is None:
            import simpler_env

            self.env = simpler_env.make(
                self.env_config.task_name,
                **self.env_config.make_kwargs(),
            )

    def collect(
        self,
        policy: Any,
        config: DemoCollectionConfig,
    ) -> List[Trajectory]:
        """Run episodes until *target_demos* successes are collected.

        Args:
            policy: Octo policy wrapper with ``.reset()`` / ``.step()``.
            config: DemoCollectionConfig with max_attempts and target_demos.

        Returns:
            List of successful Trajectory objects (reward == 1.0).
        """
        self._ensure_env()
        from simpler_env.utils.env.observation_utils import (
            get_image_from_maniskill2_obs_dict,
        )

        demos: List[Trajectory] = []
        attempts = 0

        while len(demos) < config.target_demos and attempts < config.max_attempts:
            obs, _ = self.env.reset()
            instruction = self.env.get_language_instruction()
            policy.reset(instruction)

            observations: List[np.ndarray] = []
            actions: List[np.ndarray] = []
            truncated = False
            step_count = 0
            success = False

            while not truncated and step_count < self.env_config.max_episode_steps:
                image = get_image_from_maniskill2_obs_dict(self.env, obs)
                observations.append(image.copy())

                raw_action, action = policy.step(image, instruction)
                action_array = np.concatenate([
                    action["world_vector"],
                    action["rot_axangle"],
                    action["gripper"].flatten(),
                ])

                # Store the RAW model output (euler rotation, continuous gripper)
                # for BC training — this matches the model's internal action space.
                raw_action_array = np.concatenate([
                    raw_action["world_vector"],
                    raw_action["rotation_delta"],
                    raw_action["open_gripper"].flatten(),
                ])
                actions.append(raw_action_array.copy())

                obs, reward, done, truncated, info = self.env.step(action_array)
                step_count += 1

                new_instruction = self.env.get_language_instruction()
                if new_instruction != instruction:
                    instruction = new_instruction
                    policy.reset(instruction)

                if "success" in info and info["success"]:
                    success = True

            attempts += 1

            if success:
                demos.append(Trajectory(
                    observations=observations,
                    actions=actions,
                    reward=1.0,
                    dust_mask=None,
                ))
                logger.info(
                    f"  Demo {len(demos)}/{config.target_demos} collected "
                    f"(attempt {attempts}/{config.max_attempts}, "
                    f"steps={len(actions)})"
                )
            else:
                logger.debug(
                    f"  Attempt {attempts}/{config.max_attempts}: failed "
                    f"({len(actions)} steps)"
                )

        logger.info(
            f"Demo collection complete: {len(demos)} demos from "
            f"{attempts} attempts "
            f"({len(demos) / max(attempts, 1):.1%} success rate)"
        )
        return demos


class OctoBCTrainer:
    """Behavioral-cloning fine-tuning for Octo using its native diffusion loss.

    Freezes the transformer backbone and language encoder, then trains the
    action head (``heads_action/diffusion_model``) on successful demonstrations.
    Uses the same loss function as Octo's official fine-tuning examples.
    """

    def __init__(
        self,
        finetuning_config: FinetuningConfig,
        env_config: EnvConfig,
    ):
        self.config = finetuning_config
        self.env_config = env_config

        self.model = None
        self.train_state = None
        self._rng = None
        self._initialized = False

    # ---- public API --------------------------------------------------------

    def initialize(self):
        """Load Octo model and create TrainState.  Must be called on GPU."""
        if self._initialized:
            return

        import jax
        import jax.numpy as jnp
        import optax
        from flax.training.train_state import TrainState
        from octo.model.octo_model import OctoModel
        from octo.utils.train_utils import freeze_weights

        logger.info("Loading Octo model for BC fine-tuning...")
        self.model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")

        tx = optax.adam(self.config.learning_rate)

        if self.config.freeze_transformer:
            frozen_keys = ["*hf_model*", "BlockTransformer_0"]
            tx = freeze_weights(tx, self.model.params, frozen_keys)

        self.train_state = TrainState.create(
            apply_fn=self.model.module.apply,
            params=self.model.params,
            tx=tx,
        )
        self._rng = jax.random.PRNGKey(0)
        self._initialized = True
        logger.info("Octo BC trainer initialized")

    def prepare_batch(
        self,
        trajectories: List[Trajectory],
        task_text: str,
        batch_size: int,
        action_horizon: int = 4,
    ) -> dict:
        """Sample a training batch from collected demonstrations.

        Formats data into the dict layout that Octo's model expects:
            batch["observation"]["image_primary"]: (B, 1, 256, 256, 3) uint8
            batch["observation"]["pad_mask"]: (B, 1) bool
            batch["task"]: from model.create_tasks(texts=[task_text])
            batch["action"]: (B, action_horizon, 7) float32

        Octo's diffusion action head predicts ``action_horizon`` future
        actions (default 4), so the ground-truth must match.

        Args:
            trajectories: Successful demonstration trajectories.
            task_text: Language instruction (e.g. "pick coke can").
            batch_size: Number of transitions to sample.
            action_horizon: Number of consecutive future actions per sample.

        Returns:
            Batch dict ready for Octo's loss function.
        """
        import jax
        import jax.numpy as jnp

        # Build (obs_index, trajectory) windows where each sample has
        # action_horizon consecutive future actions available.
        windows: List[tuple] = []  # (traj_idx, step_idx)
        for t_idx, traj in enumerate(trajectories):
            n = min(len(traj.observations), len(traj.actions))
            for s in range(n - action_horizon + 1):
                windows.append((t_idx, s))

        if not windows:
            raise ValueError(
                f"No valid windows: need trajectories with >= {action_horizon} steps"
            )

        n_total = len(windows)
        chosen = np.random.choice(
            n_total, size=min(batch_size, n_total), replace=n_total < batch_size
        )

        # Resize images to 256x256 (Octo's expected input size)
        import cv2

        images = []
        action_chunks = []
        for idx in chosen:
            t_idx, s = windows[idx]
            traj = trajectories[t_idx]
            img = traj.observations[s]
            img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            images.append(img_resized)
            # Consecutive action_horizon actions starting at step s
            action_chunks.append(
                np.stack(traj.actions[s : s + action_horizon])
            )

        images_np = np.stack(images)[:, np.newaxis, ...]  # (B, 1, 256, 256, 3)
        actions_np = np.stack(action_chunks)  # (B, action_horizon, 7)

        # Normalize actions to model's internal space.
        # Demo actions are in denormalized space (world_vector + rot + gripper).
        # Octo's loss expects normalized actions.
        dataset_id = list(self.model.dataset_statistics.keys())[0]
        action_mean = np.array(self.model.dataset_statistics[dataset_id]["action"]["mean"])
        action_std = np.array(self.model.dataset_statistics[dataset_id]["action"]["std"])
        actions_np = (actions_np - action_mean) / np.clip(action_std, 1e-6, None)

        pad_mask = np.ones((len(chosen), 1), dtype=bool)

        task = self.model.create_tasks(texts=[task_text])
        # Tile task to batch size
        task = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (len(chosen), *x.shape[1:])),
            task,
        )

        return {
            "observation": {
                "image_primary": jnp.array(images_np, dtype=jnp.uint8),
                "pad_mask": jnp.array(pad_mask),
            },
            "task": task,
            "action": jnp.array(actions_np, dtype=jnp.float32),
        }

    def train(
        self,
        trajectories: List[Trajectory],
        task_text: str,
    ) -> List[float]:
        """Run behavioral-cloning training loop.

        Args:
            trajectories: Successful demonstration trajectories.
            task_text: Language instruction for the task.

        Returns:
            List of loss values (one per training step).
        """
        import jax
        import jax.numpy as jnp

        self.initialize()

        def loss_fn(params, batch, rng):
            bound = self.model.module.bind({"params": params}, rngs={"dropout": rng})
            embeddings = bound.octo_transformer(
                batch["observation"],
                batch["task"],
                batch["observation"]["pad_mask"],
                train=True,
            )
            loss, metrics = bound.heads["action"].loss(
                embeddings,
                batch["action"],
                pad_mask=batch["observation"]["pad_mask"],
                train=True,
            )
            return loss, metrics

        @jax.jit
        def train_step(state, batch, rng):
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params, batch, rng
            )
            state = state.apply_gradients(grads=grads)
            return state, loss, metrics

        losses: List[float] = []
        logger.info(
            f"Starting BC training: {self.config.n_steps} steps, "
            f"batch_size={self.config.batch_size}, lr={self.config.learning_rate}"
        )

        for step in range(self.config.n_steps):
            self._rng, step_rng = jax.random.split(self._rng)
            batch = self.prepare_batch(
                trajectories, task_text, self.config.batch_size
            )
            self.train_state, loss_val, metrics = train_step(
                self.train_state, batch, step_rng
            )
            losses.append(float(loss_val))

            if (step + 1) % 100 == 0 or step == 0:
                logger.info(f"  Step {step + 1}/{self.config.n_steps}: loss={loss_val:.4f}")

        # Sync model params
        self.model = self.model.replace(params=self.train_state.params)
        logger.info(f"BC training complete. Final loss: {losses[-1]:.4f}")
        return losses

    def save(self, path: str):
        """Save the fine-tuned model checkpoint."""
        abs_path = str(Path(path).resolve())
        self.model.save_pretrained(step=0, checkpoint_path=abs_path)
        logger.info(f"Saved fine-tuned checkpoint to {abs_path}")

    def get_policy(self):
        """Return an OctoInference-compatible policy from the fine-tuned model.

        Returns a wrapper that has ``.reset()`` and ``.step()`` matching
        the SimplerEnv OctoInference interface.
        """
        from simpler_env.policies.octo.octo_model import OctoInference

        task_name = self.env_config.task_name
        policy_setup = "widowx_bridge" if task_name.startswith("widowx") else "google_robot"

        dataset_id = "bridge_dataset" if policy_setup == "widowx_bridge" else "fractal20220817_data"
        policy = OctoInference(
            model=self.model,
            dataset_id=dataset_id,
            policy_setup=policy_setup,
        )
        logger.info(f"Created policy from fine-tuned model (dataset: {dataset_id})")
        return policy
