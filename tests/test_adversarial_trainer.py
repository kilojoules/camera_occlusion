"""Unit tests for adversarial_trainer.py (no GPU or simulator needed)."""

import numpy as np
import pytest
import torch

from adversarial_dust.config import (
    AdversarialExperimentConfig,
    AdversarialTrainingConfig,
    GeneratorConfig,
)
from adversarial_dust.adversarial_trainer import (
    AdversarialTrainer,
    MaskApplicator,
    ReinforceTrainer,
    RoundMetrics,
)
from adversarial_dust.dust_generator import DustGenerator

IMAGE_SHAPE = (64, 80, 3)  # Small for fast tests


@pytest.fixture
def gen_config():
    return GeneratorConfig(
        latent_dim=4,
        encoder_channels=[4, 8],
        decoder_channels=[8, 4],
        intermediate_resolution=(4, 4),
        max_opacity=0.6,
        num_blobs=2,
        learning_rate=0.01,
        explore_sigma=0.05,
    )


@pytest.fixture
def generator(gen_config):
    return DustGenerator(gen_config, IMAGE_SHAPE)


@pytest.fixture
def mask_applicator(gen_config):
    return MaskApplicator(gen_config, IMAGE_SHAPE)


@pytest.fixture
def dummy_image():
    return torch.rand(1, 3, IMAGE_SHAPE[0], IMAGE_SHAPE[1])


class TestMaskApplicator:
    def test_compute_coverage(self, mask_applicator):
        mask = np.ones((IMAGE_SHAPE[0], IMAGE_SHAPE[1]), dtype=np.float32) * 0.3
        coverage = mask_applicator.compute_coverage(mask)
        assert coverage == 1.0  # all pixels above threshold

    def test_compute_coverage_empty(self, mask_applicator):
        mask = np.zeros((IMAGE_SHAPE[0], IMAGE_SHAPE[1]), dtype=np.float32)
        coverage = mask_applicator.compute_coverage(mask)
        assert coverage == 0.0

    def test_project_to_budget_no_change(self, mask_applicator):
        mask = np.ones((IMAGE_SHAPE[0], IMAGE_SHAPE[1]), dtype=np.float32) * 0.01
        projected = mask_applicator.project_to_budget(mask, budget_level=0.5)
        np.testing.assert_array_almost_equal(mask, projected)

    def test_project_to_budget_scales_down(self, mask_applicator):
        mask = np.ones((IMAGE_SHAPE[0], IMAGE_SHAPE[1]), dtype=np.float32) * 0.3
        projected = mask_applicator.project_to_budget(mask, budget_level=0.1)
        coverage = mask_applicator.compute_coverage(projected)
        assert coverage <= 0.1 + 0.01  # allow small tolerance from binary search

    def test_project_full_budget(self, mask_applicator):
        mask = np.ones((IMAGE_SHAPE[0], IMAGE_SHAPE[1]), dtype=np.float32) * 0.5
        projected = mask_applicator.project_to_budget(mask, budget_level=1.0)
        np.testing.assert_array_equal(mask, projected)

    def test_apply_uint8(self, mask_applicator):
        image = np.random.randint(0, 256, IMAGE_SHAPE, dtype=np.uint8)
        mask = np.ones((IMAGE_SHAPE[0], IMAGE_SHAPE[1]), dtype=np.float32) * 0.3
        result = mask_applicator.apply(image, mask, budget_level=0.5)
        assert result.dtype == np.uint8
        assert result.shape == IMAGE_SHAPE

    def test_apply_float32(self, mask_applicator):
        image = np.random.rand(*IMAGE_SHAPE).astype(np.float32)
        mask = np.ones((IMAGE_SHAPE[0], IMAGE_SHAPE[1]), dtype=np.float32) * 0.3
        result = mask_applicator.apply(image, mask, budget_level=0.5)
        assert result.dtype == np.float32
        assert result.shape == IMAGE_SHAPE

    def test_apply_with_zero_mask(self, mask_applicator):
        image = np.random.randint(0, 256, IMAGE_SHAPE, dtype=np.uint8)
        mask = np.zeros((IMAGE_SHAPE[0], IMAGE_SHAPE[1]), dtype=np.float32)
        result = mask_applicator.apply(image, mask, budget_level=0.5)
        np.testing.assert_array_equal(image, result)


class TestReinforceTrainer:
    def test_step_returns_metrics(self, generator, dummy_image, gen_config):
        trainer = ReinforceTrainer(generator, gen_config)

        # Mock evaluate: always return fixed success rate
        def mock_eval(mask_np):
            return 0.5

        metrics = trainer.step(
            dummy_image, evaluate_fn=mock_eval, n_samples=2, sigma=0.05
        )

        assert "loss" in metrics
        assert "mean_reward" in metrics
        assert "mean_success_rate" in metrics
        assert metrics["mean_success_rate"] == 0.5
        assert metrics["mean_reward"] == 0.5  # 1.0 - 0.5

    def test_step_updates_params(self, generator, dummy_image, gen_config):
        trainer = ReinforceTrainer(generator, gen_config)
        # Set baseline away from expected rewards so advantages are non-zero
        trainer.baseline = 0.0

        # Get initial params
        params_before = [p.clone() for p in generator.parameters()]

        call_count = {"n": 0}

        def mock_eval(mask_np):
            # Return varying success rates so advantages differ
            call_count["n"] += 1
            return 0.1 * call_count["n"]

        trainer.step(dummy_image, evaluate_fn=mock_eval, n_samples=2, sigma=0.05)

        # Check at least some params changed
        params_changed = False
        for p_before, p_after in zip(params_before, generator.parameters()):
            if not torch.allclose(p_before, p_after):
                params_changed = True
                break
        assert params_changed, "REINFORCE step should update parameters"

    def test_baseline_updates(self, generator, dummy_image, gen_config):
        trainer = ReinforceTrainer(generator, gen_config)
        initial_baseline = trainer.baseline

        def mock_eval(mask_np):
            return 0.0  # complete failure → reward = 1.0

        trainer.step(dummy_image, evaluate_fn=mock_eval, n_samples=2, sigma=0.05)

        # Baseline should move toward 1.0 (the observed reward)
        assert trainer.baseline != initial_baseline

    def test_multiple_steps(self, generator, dummy_image, gen_config):
        trainer = ReinforceTrainer(generator, gen_config)

        def mock_eval(mask_np):
            # Random success rate
            return float(np.random.rand())

        for _ in range(5):
            metrics = trainer.step(
                dummy_image, evaluate_fn=mock_eval, n_samples=2, sigma=0.05
            )
            assert isinstance(metrics["loss"], float)


class TestAdversarialTrainer:
    def test_train_round(self, gen_config, generator, mask_applicator, dummy_image):
        config = AdversarialExperimentConfig(
            generator=gen_config,
            training=AdversarialTrainingConfig(
                n_rounds=1,
                attacker_steps_per_round=2,
                n_samples_per_step=2,
                episodes_per_reinforce_sample=1,
                budget_levels=[0.2],
                eval_episodes=2,
            ),
        )

        call_count = {"n": 0}

        def mock_evaluate(mask_or_none, n_episodes):
            call_count["n"] += 1
            return 0.5

        ref_image = np.random.randint(0, 256, IMAGE_SHAPE, dtype=np.uint8)

        trainer = AdversarialTrainer(
            config=config,
            generator=generator,
            mask_applicator=mask_applicator,
            evaluate_fn=mock_evaluate,
            defender=None,
            reference_image=ref_image,
        )

        metrics = trainer.train_round(round_idx=0, budget_level=0.2)

        assert isinstance(metrics, RoundMetrics)
        assert metrics.round_idx == 0
        assert metrics.budget_level == 0.2
        assert len(metrics.attacker_losses) == 2
        assert 0.0 <= metrics.adversarial_eval_sr <= 1.0
        assert not metrics.defender_updated

    def test_train_full_loop(self, gen_config, generator, mask_applicator):
        config = AdversarialExperimentConfig(
            generator=gen_config,
            training=AdversarialTrainingConfig(
                n_rounds=2,
                attacker_steps_per_round=1,
                n_samples_per_step=2,
                episodes_per_reinforce_sample=1,
                budget_levels=[0.1, 0.2],
                eval_episodes=1,
            ),
        )

        def mock_evaluate(mask_or_none, n_episodes):
            return 0.5

        ref_image = np.random.randint(0, 256, IMAGE_SHAPE, dtype=np.uint8)

        trainer = AdversarialTrainer(
            config=config,
            generator=generator,
            mask_applicator=mask_applicator,
            evaluate_fn=mock_evaluate,
            defender=None,
            reference_image=ref_image,
        )

        all_metrics = trainer.train()

        # 2 rounds × 2 budgets = 4 round metrics
        assert len(all_metrics) == 4

    def test_get_history_dict(self, gen_config, generator, mask_applicator):
        config = AdversarialExperimentConfig(
            generator=gen_config,
            training=AdversarialTrainingConfig(
                n_rounds=1,
                attacker_steps_per_round=1,
                n_samples_per_step=2,
                episodes_per_reinforce_sample=1,
                budget_levels=[0.1],
                eval_episodes=1,
            ),
        )

        def mock_evaluate(mask_or_none, n_episodes):
            return 0.5

        ref_image = np.random.randint(0, 256, IMAGE_SHAPE, dtype=np.uint8)

        trainer = AdversarialTrainer(
            config=config,
            generator=generator,
            mask_applicator=mask_applicator,
            evaluate_fn=mock_evaluate,
            defender=None,
            reference_image=ref_image,
        )

        trainer.train()
        history = trainer.get_history_dict()

        assert "rounds" in history
        assert "budgets" in history
        assert "adversarial_srs" in history
        assert "clean_srs" in history
        assert len(history["rounds"]) == 1
