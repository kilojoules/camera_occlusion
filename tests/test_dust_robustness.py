"""Tests for dust robustness evaluation pipeline.

Tests severity presets, batch formatting, demo collection, config loading,
and the visualization function — all without GPU or SimplerEnv.
"""

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Mock helpers (reused from test_evaluator.py pattern)
# ---------------------------------------------------------------------------


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
    """Mock SimplerEnv environment that succeeds after N steps."""

    def __init__(self, succeed_after: int = 3, max_steps: int = 5):
        self._step_count = 0
        self._succeed_after = succeed_after
        self._max_steps = max_steps
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
        done = self._step_count >= self._max_steps
        info = {"success": self._step_count >= self._succeed_after}
        return obs, 0.0, done, False, info

    def close(self):
        pass


def _mock_get_image(env, obs):
    return obs["image"]["overhead_camera"]["rgb"]


def _install_simpler_env_mock(monkeypatch):
    """Patch simpler_env imports so they resolve without the real package."""
    mock_obs_utils = types.ModuleType("simpler_env.utils.env.observation_utils")
    mock_obs_utils.get_image_from_maniskill2_obs_dict = _mock_get_image
    monkeypatch.setitem(
        sys.modules, "simpler_env.utils.env.observation_utils", mock_obs_utils
    )

    mock_simpler_env = types.ModuleType("simpler_env")
    mock_simpler_env.make = MagicMock(return_value=MockEnv())
    monkeypatch.setitem(sys.modules, "simpler_env", mock_simpler_env)


class MockDustEffect:
    """Mock camera_occlusion.Dust effect that adds noise."""

    def __init__(self, intensity: float = 0.1):
        self.intensity = intensity

    def apply(self, image):
        noise = np.random.normal(0, self.intensity * 50, image.shape)
        return np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)

    def __call__(self, image):
        return self.apply(image)


def _make_mock_presets():
    """Create mock dust presets for testing without camera_occlusion."""
    return {
        "clean": None,
        "light": MockDustEffect(intensity=0.1),
        "moderate": MockDustEffect(intensity=0.3),
        "heavy": MockDustEffect(intensity=0.6),
    }


# ---------------------------------------------------------------------------
# Test: severity presets (mocked)
# ---------------------------------------------------------------------------


class TestDustPresets:
    def test_presets_have_correct_keys(self):
        presets = _make_mock_presets()
        assert "clean" in presets
        assert "light" in presets
        assert "moderate" in presets
        assert "heavy" in presets

    def test_clean_is_none(self):
        presets = _make_mock_presets()
        assert presets["clean"] is None

    def test_dust_effects_are_callable(self):
        presets = _make_mock_presets()
        for name, effect in presets.items():
            if effect is not None:
                assert callable(effect), f"{name} preset is not callable"

    def test_dust_apply_preserves_shape(self):
        presets = _make_mock_presets()
        image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        for name, effect in presets.items():
            if effect is not None:
                result = effect(image)
                assert result.shape == image.shape, (
                    f"{name}: output shape {result.shape} != input {image.shape}"
                )
                assert result.dtype == np.uint8

    def test_dust_severity_ordering(self):
        """Heavier dust should modify the image more than lighter dust."""
        presets = _make_mock_presets()
        np.random.seed(42)
        image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        diffs = {}
        n_trials = 5
        for name in ["light", "moderate", "heavy"]:
            effect = presets[name]
            total_diff = 0.0
            for _ in range(n_trials):
                result = effect(image)
                total_diff += np.mean(np.abs(result.astype(float) - image.astype(float)))
            diffs[name] = total_diff / n_trials

        assert diffs["heavy"] > diffs["light"], (
            f"Expected heavy ({diffs['heavy']:.2f}) > light ({diffs['light']:.2f})"
        )

    def test_make_dust_presets_function_exists(self):
        """The make_dust_presets function is importable."""
        from adversarial_dust.dust_robustness_eval import make_dust_presets

        assert callable(make_dust_presets)


# ---------------------------------------------------------------------------
# Test: DustRobustnessEvaluator
# ---------------------------------------------------------------------------


class TestDustRobustnessEvaluator:
    def _make_evaluator(self, monkeypatch, severities=None):
        """Helper to create an evaluator with mocked deps."""
        _install_simpler_env_mock(monkeypatch)

        from adversarial_dust.config import DustEvalConfig, EnvConfig
        from adversarial_dust.dust_robustness_eval import DustRobustnessEvaluator

        env_config = EnvConfig(max_episode_steps=10)
        eval_config = DustEvalConfig(
            severity_levels=severities or ["clean", "heavy"],
            episodes_per_eval=2,
        )
        evaluator = DustRobustnessEvaluator(env_config, eval_config)
        evaluator.env = MockEnv(succeed_after=3)
        evaluator.presets = _make_mock_presets()
        return evaluator

    def test_evaluate_model_clean(self, monkeypatch):
        """Evaluator runs clean episodes and returns a valid success rate."""
        evaluator = self._make_evaluator(monkeypatch, severities=["clean"])
        policy = MockPolicy()
        sr, successes = evaluator.evaluate_model(policy, "clean", n_episodes=3)

        assert 0.0 <= sr <= 1.0
        assert len(successes) == 3

    def test_evaluate_model_with_dust(self, monkeypatch):
        """Evaluator runs episodes with dust effect and returns valid results."""
        evaluator = self._make_evaluator(monkeypatch)
        policy = MockPolicy()
        sr, successes = evaluator.evaluate_model(policy, "heavy", n_episodes=2)

        assert 0.0 <= sr <= 1.0
        assert len(successes) == 2

    def test_evaluate_all(self, monkeypatch):
        """evaluate_all sweeps all models and severities."""
        evaluator = self._make_evaluator(
            monkeypatch, severities=["clean", "light"]
        )

        models = {"baseline": MockPolicy(), "finetuned": MockPolicy()}
        results = evaluator.evaluate_all(models)

        assert "baseline" in results
        assert "finetuned" in results
        for model_name in results:
            assert "clean" in results[model_name]
            assert "light" in results[model_name]
            for sev in results[model_name]:
                assert "success_rate" in results[model_name][sev]
                assert "successes" in results[model_name][sev]

    def test_results_serializable(self, monkeypatch):
        """evaluate_all results can be serialized to JSON."""
        evaluator = self._make_evaluator(
            monkeypatch, severities=["clean", "heavy"]
        )
        models = {"baseline": MockPolicy()}
        results = evaluator.evaluate_all(models)

        # Should not raise
        serialized = json.dumps(results, default=str)
        assert len(serialized) > 0


# ---------------------------------------------------------------------------
# Test: DemoCollector
# ---------------------------------------------------------------------------


class TestDemoCollector:
    def test_collect_demos(self, monkeypatch):
        """DemoCollector collects successful trajectories."""
        _install_simpler_env_mock(monkeypatch)

        from adversarial_dust.config import DemoCollectionConfig, EnvConfig
        from adversarial_dust.octo_finetuner import DemoCollector

        env_config = EnvConfig(max_episode_steps=10)
        collector = DemoCollector(env_config)
        collector.env = MockEnv(succeed_after=3)

        demo_config = DemoCollectionConfig(max_attempts=5, target_demos=2)
        demos = collector.collect(MockPolicy(), demo_config)

        assert len(demos) <= 2
        for demo in demos:
            assert demo.reward == 1.0
            assert len(demo.observations) > 0
            assert len(demo.actions) > 0
            assert demo.dust_mask is None

    def test_collect_stops_at_max_attempts(self, monkeypatch):
        """DemoCollector stops when max_attempts is reached."""
        _install_simpler_env_mock(monkeypatch)

        from adversarial_dust.config import DemoCollectionConfig, EnvConfig
        from adversarial_dust.octo_finetuner import DemoCollector

        env_config = EnvConfig(max_episode_steps=10)
        collector = DemoCollector(env_config)
        collector.env = MockEnv(succeed_after=999)

        demo_config = DemoCollectionConfig(max_attempts=3, target_demos=10)
        demos = collector.collect(MockPolicy(), demo_config)

        assert len(demos) == 0

    def test_trajectory_fields(self, monkeypatch):
        """Collected trajectories have proper field shapes."""
        _install_simpler_env_mock(monkeypatch)

        from adversarial_dust.config import DemoCollectionConfig, EnvConfig
        from adversarial_dust.octo_finetuner import DemoCollector

        env_config = EnvConfig(max_episode_steps=10)
        collector = DemoCollector(env_config)
        collector.env = MockEnv(succeed_after=2, max_steps=4)

        demo_config = DemoCollectionConfig(max_attempts=5, target_demos=1)
        demos = collector.collect(MockPolicy(), demo_config)

        assert len(demos) >= 1
        traj = demos[0]
        # Each observation should be an image
        assert traj.observations[0].shape == (256, 256, 3)
        assert traj.observations[0].dtype == np.uint8
        # Each action should be a 7-DOF vector
        assert traj.actions[0].shape == (7,)


# ---------------------------------------------------------------------------
# Test: OctoBCTrainer batch formatting
# ---------------------------------------------------------------------------


class TestBatchFormatting:
    def test_prepare_batch_shapes(self, monkeypatch):
        """prepare_batch produces correctly shaped arrays."""
        _install_simpler_env_mock(monkeypatch)

        # Mock jax for environments without GPU
        mock_jnp = MagicMock()
        mock_jnp.array = lambda x, **kw: np.array(x)
        mock_jnp.uint8 = np.uint8
        mock_jnp.float32 = np.float32
        mock_jnp.broadcast_to = np.broadcast_to

        mock_jax = MagicMock()
        mock_jax.numpy = mock_jnp
        mock_jax.tree.map = lambda fn, tree: {
            k: fn(v) for k, v in tree.items()
        }

        monkeypatch.setitem(sys.modules, "jax", mock_jax)
        monkeypatch.setitem(sys.modules, "jax.numpy", mock_jnp)

        from adversarial_dust.config import EnvConfig, FinetuningConfig
        from adversarial_dust.octo_finetuner import OctoBCTrainer, Trajectory

        trajs = [
            Trajectory(
                observations=[
                    np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
                    for _ in range(5)
                ],
                actions=[np.random.randn(7).astype(np.float32) for _ in range(5)],
                reward=1.0,
            ),
            Trajectory(
                observations=[
                    np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
                    for _ in range(3)
                ],
                actions=[np.random.randn(7).astype(np.float32) for _ in range(3)],
                reward=1.0,
            ),
        ]

        trainer = OctoBCTrainer(
            FinetuningConfig(n_steps=10, batch_size=4),
            EnvConfig(),
        )

        # Mock the model's create_tasks
        mock_task = {"language_instruction": np.zeros((1, 16, 512))}
        mock_model = MagicMock()
        mock_model.create_tasks.return_value = mock_task
        trainer.model = mock_model

        batch = trainer.prepare_batch(trajs, "pick coke can", batch_size=4)

        obs_images = batch["observation"]["image_primary"]
        assert obs_images.shape[1] == 1  # window=1
        assert obs_images.shape[2:] == (256, 256, 3)

        pad_mask = batch["observation"]["pad_mask"]
        assert pad_mask.shape[1] == 1

        actions = batch["action"]
        assert actions.shape[1] == 1  # window=1
        assert actions.shape[2] == 7  # 7-DOF action


# ---------------------------------------------------------------------------
# Test: Config loading
# ---------------------------------------------------------------------------


class TestDustRobustnessConfig:
    def test_default_config(self):
        from adversarial_dust.config import DustRobustnessConfig

        config = DustRobustnessConfig()
        assert config.env.task_name == "google_robot_pick_coke_can"
        assert config.demo_collection.target_demos == 20
        assert config.finetuning.freeze_transformer is True
        assert config.evaluation.episodes_per_eval == 50
        assert "clean" in config.evaluation.severity_levels

    def test_load_from_yaml(self, tmp_path):
        import yaml

        from adversarial_dust.config import load_dust_robustness_config

        config_data = {
            "env": {"task_name": "google_robot_pick_coke_can"},
            "demo_collection": {"max_attempts": 100, "target_demos": 5},
            "finetuning": {"n_steps": 500},
            "evaluation": {
                "severity_levels": ["clean", "heavy"],
                "episodes_per_eval": 10,
            },
            "output_dir": "results/test_run",
        }

        yaml_path = tmp_path / "test_config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_dust_robustness_config(str(yaml_path))
        assert config.demo_collection.max_attempts == 100
        assert config.demo_collection.target_demos == 5
        assert config.finetuning.n_steps == 500
        assert config.evaluation.severity_levels == ["clean", "heavy"]
        assert config.output_dir == "results/test_run"

    def test_load_none_returns_defaults(self):
        from adversarial_dust.config import load_dust_robustness_config

        config = load_dust_robustness_config(None)
        assert config.env.task_name == "google_robot_pick_coke_can"
        assert config.finetuning.learning_rate == 3e-5

    def test_config_env_control_freq(self):
        from adversarial_dust.config import DustRobustnessConfig

        config = DustRobustnessConfig()
        assert config.env.control_freq == 3
        assert config.env.sim_freq == 513


# ---------------------------------------------------------------------------
# Test: Visualization
# ---------------------------------------------------------------------------


class TestVisualization:
    def test_plot_robustness_comparison(self, tmp_path):
        """plot_robustness_comparison creates a plot file."""
        try:
            from adversarial_dust.visualization import plot_robustness_comparison
        except ImportError:
            pytest.skip("matplotlib not available or incompatible")

        results = {
            "baseline": {
                "clean": {"success_rate": 0.8, "successes": [True] * 8 + [False] * 2},
                "light": {"success_rate": 0.6, "successes": [True] * 6 + [False] * 4},
                "heavy": {"success_rate": 0.2, "successes": [True] * 2 + [False] * 8},
            },
            "finetuned": {
                "clean": {"success_rate": 0.9, "successes": [True] * 9 + [False] * 1},
                "light": {"success_rate": 0.7, "successes": [True] * 7 + [False] * 3},
                "heavy": {"success_rate": 0.4, "successes": [True] * 4 + [False] * 6},
            },
        }

        output_path = str(tmp_path / "test_degradation.png")
        plot_robustness_comparison(
            results, output_path, severity_order=["clean", "light", "heavy"]
        )

        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0
