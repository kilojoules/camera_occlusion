"""Smoke tests for evaluator with mock policy (no GPU/SimplerEnv needed)."""

import numpy as np
import pytest

from adversarial_dust.config import DustGridConfig, EnvConfig
from adversarial_dust.dust_model import AdversarialDustModel
from adversarial_dust.evaluator import PolicyEvaluator


class MockPolicy:
    """Mock Octo policy that returns fixed actions."""

    def reset(self, task_description: str):
        self.task_description = task_description

    def step(self, image: np.ndarray, instruction=None):
        raw_action = {
            "world_vector": np.zeros(3),
            "rotation_delta": np.zeros(3),
            "open_gripper": np.array([0.5]),
        }
        action = {
            "world_vector": np.zeros(3),
            "rot_axangle": np.zeros(3),
            "gripper": np.array([0.0]),
            "terminate_episode": np.array([0.0]),
        }
        return raw_action, action


class MockEnv:
    """Mock SimplerEnv environment."""

    def __init__(self):
        self._step_count = 0
        self._image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    def reset(self):
        self._step_count = 0
        obs = {"image": {"overhead_camera": {"rgb": self._image.copy()}}}
        return obs, {}

    def get_language_instruction(self):
        return "pick coke can"

    def step(self, action):
        self._step_count += 1
        obs = {"image": {"overhead_camera": {"rgb": self._image.copy()}}}
        done = self._step_count >= 5
        info = {"success": self._step_count >= 3}  # succeeds after 3 steps
        return obs, 0.0, done, False, info

    def close(self):
        pass


class TestPolicyEvaluator:
    def test_evaluate_with_mock(self, monkeypatch):
        """Test that evaluator correctly runs episodes with mocked env."""
        dust_config = DustGridConfig()
        env_config = EnvConfig(max_episode_steps=10)
        image_shape = (256, 256, 3)
        dust_model = AdversarialDustModel(dust_config, image_shape, budget_level=0.2)
        policy = MockPolicy()

        evaluator = PolicyEvaluator(env_config, policy, dust_model)

        # Inject mock env and mock the import
        mock_env = MockEnv()
        evaluator.env = mock_env

        # Mock the SimplerEnv observation utility
        def mock_get_image(env, obs):
            return obs["image"]["overhead_camera"]["rgb"]

        monkeypatch.setattr(
            "adversarial_dust.evaluator.get_image_from_maniskill2_obs_dict",
            mock_get_image,
            raising=False,
        )

        # Need to also patch the import inside evaluate()
        import types
        mock_module = types.ModuleType("simpler_env.utils.env.observation_utils")
        mock_module.get_image_from_maniskill2_obs_dict = mock_get_image
        monkeypatch.setitem(
            __import__("sys").modules,
            "simpler_env.utils.env.observation_utils",
            mock_module,
        )

        params = np.random.uniform(0, 0.3, size=64)
        sr = evaluator.evaluate(params, n_episodes=3)

        assert 0.0 <= sr <= 1.0

    def test_evaluate_clean(self, monkeypatch):
        """Test clean evaluation (no dust params)."""
        dust_config = DustGridConfig()
        env_config = EnvConfig(max_episode_steps=10)
        image_shape = (256, 256, 3)
        dust_model = AdversarialDustModel(dust_config, image_shape, budget_level=0.0)
        policy = MockPolicy()

        evaluator = PolicyEvaluator(env_config, policy, dust_model)
        evaluator.env = MockEnv()

        import types
        def mock_get_image(env, obs):
            return obs["image"]["overhead_camera"]["rgb"]

        mock_module = types.ModuleType("simpler_env.utils.env.observation_utils")
        mock_module.get_image_from_maniskill2_obs_dict = mock_get_image
        monkeypatch.setitem(
            __import__("sys").modules,
            "simpler_env.utils.env.observation_utils",
            mock_module,
        )

        sr = evaluator.evaluate(dust_params=None, n_episodes=2)
        assert 0.0 <= sr <= 1.0
