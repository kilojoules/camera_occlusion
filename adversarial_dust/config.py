"""Configuration dataclasses and YAML loading."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import dacite
import yaml


# ---------------------------------------------------------------------------
# Original CMA-ES experiment configs
# ---------------------------------------------------------------------------


@dataclass
class DustGridConfig:
    grid_resolution: Tuple[int, int] = (8, 8)
    max_cell_opacity: float = 0.6
    dust_color: Tuple[int, int, int] = (180, 160, 140)
    gaussian_blur_sigma: float = 3.0
    dirty_threshold: float = 0.05


@dataclass
class OptimizationConfig:
    population_size: int = 20
    max_generations: int = 50
    sigma0: float = 0.2
    seed: int = 42
    episodes_per_eval: int = 5
    episodes_final_eval: int = 25
    n_random_baselines: int = 10


@dataclass
class EnvConfig:
    task_name: str = "google_robot_pick_coke_can"
    policy_model: str = "octo-base"
    max_episode_steps: int = 80
    control_freq: Optional[int] = None
    sim_freq: Optional[int] = None
    # InternVLA-M1 server settings (only used when policy_model starts with "internvla-m1")
    internvla_m1_port: int = 10093
    internvla_m1_ckpt: str = ""

    def make_kwargs(self) -> dict:
        """Return kwargs dict for simpler_env.make()."""
        kw = {"max_episode_steps": self.max_episode_steps}
        if self.control_freq is not None:
            kw["control_freq"] = self.control_freq
        if self.sim_freq is not None:
            kw["sim_freq"] = self.sim_freq
        return kw


@dataclass
class SweepConfig:
    budget_levels: List[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.20, 0.40]
    )


@dataclass
class BlobConfig:
    """Configuration for the dynamic Gaussian blob dust model."""

    num_blobs: int = 5
    max_opacity: float = 0.6
    dust_color: Tuple[int, int, int] = (180, 160, 140)
    dirty_threshold: float = 0.05
    # Parameter ranges for CMA-ES
    sigma_range: Tuple[float, float] = (0.02, 0.30)
    velocity_range: Tuple[float, float] = (-0.005, 0.005)
    growth_rate_range: Tuple[float, float] = (-0.002, 0.005)


@dataclass
class FingerprintConfig:
    """Configuration for the fingerprint smudge occlusion model."""

    num_prints: int = 3
    max_opacity: float = 0.5
    smudge_color: Tuple[int, int, int] = (200, 190, 170)  # oily/greasy tone
    dirty_threshold: float = 0.05
    scale_range: Tuple[float, float] = (0.05, 0.25)
    freq_range: Tuple[float, float] = (3.0, 15.0)


@dataclass
class GlareConfig:
    """Configuration for the adversarial glare model."""

    max_intensity: float = 0.8
    num_streaks: int = 6
    chromatic_aberration: float = 1.5
    ghost_count: int = 4
    ghost_spacing: float = 0.4
    ghost_decay: float = 0.85
    dirty_threshold: float = 0.05


@dataclass
class ExperimentConfig:
    dust: DustGridConfig = field(default_factory=DustGridConfig)
    blob: BlobConfig = field(default_factory=BlobConfig)
    fingerprint: FingerprintConfig = field(default_factory=FingerprintConfig)
    glare: GlareConfig = field(default_factory=GlareConfig)
    dust_model_type: str = "grid"  # "grid", "blob", "fingerprint", "glare"
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    output_dir: str = "results/default"


@dataclass
class EnvelopeConfig:
    """Configuration for safe operating envelope prediction."""

    occlusion_types: List[str] = field(
        default_factory=lambda: ["fingerprint", "glare"]
    )
    budget_levels: List[float] = field(
        default_factory=lambda: [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    )
    episodes_per_eval: int = 10
    episodes_final_eval: int = 25
    population_size: int = 20
    max_generations: int = 50
    sigma0: float = 0.2
    seed: int = 42
    n_random_baselines: int = 5
    safe_threshold: float = 0.95
    marginal_threshold: float = 0.70


@dataclass
class EnvelopeExperimentConfig:
    """Top-level config for safe operating envelope experiments."""

    envelope: EnvelopeConfig = field(default_factory=EnvelopeConfig)
    dust: DustGridConfig = field(default_factory=DustGridConfig)
    blob: BlobConfig = field(default_factory=BlobConfig)
    fingerprint: FingerprintConfig = field(default_factory=FingerprintConfig)
    glare: GlareConfig = field(default_factory=GlareConfig)
    env: EnvConfig = field(default_factory=lambda: EnvConfig(
        task_name="widowx_put_eggplant_in_basket",
        policy_model="internvla-m1",
        max_episode_steps=200,
    ))
    output_dir: str = "results/envelope"


def load_config(yaml_path: Optional[str] = None) -> ExperimentConfig:
    """Load config from YAML, merging onto defaults."""
    if yaml_path is None:
        return ExperimentConfig()

    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    return dacite.from_dict(
        data_class=ExperimentConfig,
        data=raw,
        config=dacite.Config(strict=False, cast=[tuple]),
    )


# ---------------------------------------------------------------------------
# Adversarial zero-sum training configs
# ---------------------------------------------------------------------------


@dataclass
class GeneratorConfig:
    """Configuration for the PyTorch dust generator network."""

    latent_dim: int = 32
    encoder_channels: List[int] = field(default_factory=lambda: [16, 32, 64])
    decoder_channels: List[int] = field(default_factory=lambda: [64, 32, 16])
    intermediate_resolution: Tuple[int, int] = (16, 16)
    max_opacity: float = 0.6
    num_blobs: int = 5
    learning_rate: float = 1e-3
    explore_sigma: float = 0.05
    dust_color: Tuple[int, int, int] = (180, 160, 140)
    dirty_threshold: float = 0.05


@dataclass
class BlobGeneratorConfig:
    """Configuration for the Gaussian blob generator network."""

    k_max: int = 4
    encoder_channels: List[int] = field(default_factory=lambda: [16, 32, 64])
    mlp_hidden: int = 64
    sigma_min: float = 0.02
    sigma_max: float = 0.30
    max_opacity: float = 0.6
    learning_rate: float = 1e-3
    explore_sigma: float = 0.02
    dust_color: Tuple[int, int, int] = (180, 160, 140)
    dirty_threshold: float = 0.05
    # Temporal blob parameters (area-time budget mode)
    temporal: bool = False
    max_t_width: float = 1.0
    temporal_sharpness: float = 20.0


@dataclass
class DefenderConfig:
    """Configuration for Octo GRPO fine-tuning (defender)."""

    freeze_pattern: str = "head_only"
    learning_rate: float = 3e-4
    grpo_episodes_per_round: int = 20
    grpo_update_steps: int = 50
    grpo_batch_size: int = 8


@dataclass
class AdversarialTrainingConfig:
    """Configuration for the zero-sum training loop."""

    n_rounds: int = 10
    attacker_steps_per_round: int = 50
    episodes_per_reinforce_sample: int = 3
    n_samples_per_step: int = 4
    budget_levels: List[float] = field(
        default_factory=lambda: [0.10, 0.20, 0.40]
    )
    eval_episodes: int = 20
    seed: int = 42


@dataclass
class AdversarialExperimentConfig:
    """Top-level config for adversarial zero-sum training experiments."""

    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    blob_generator: Optional[BlobGeneratorConfig] = None
    defender: DefenderConfig = field(default_factory=DefenderConfig)
    training: AdversarialTrainingConfig = field(
        default_factory=AdversarialTrainingConfig
    )
    env: EnvConfig = field(default_factory=lambda: EnvConfig(
        task_name="widowx_put_eggplant_in_basket",
        policy_model="octo-small",
    ))
    output_dir: str = "results/adversarial"


# ---------------------------------------------------------------------------
# Dust robustness (baseline vs fine-tuned) configs
# ---------------------------------------------------------------------------


@dataclass
class DemoCollectionConfig:
    """Configuration for clean demonstration collection."""

    max_attempts: int = 300
    target_demos: int = 20


@dataclass
class FinetuningConfig:
    """Configuration for Octo behavioral-cloning fine-tuning."""

    freeze_transformer: bool = True
    learning_rate: float = 3e-5
    n_steps: int = 2000
    batch_size: int = 32


@dataclass
class DustEvalConfig:
    """Configuration for dust robustness evaluation."""

    severity_levels: List[str] = field(
        default_factory=lambda: ["clean", "light", "moderate", "heavy"]
    )
    episodes_per_eval: int = 50


@dataclass
class DustRobustnessConfig:
    """Top-level config for baseline vs fine-tuned dust robustness experiments."""

    env: EnvConfig = field(default_factory=lambda: EnvConfig(
        task_name="google_robot_pick_coke_can",
        policy_model="octo-base",
        max_episode_steps=80,
        control_freq=3,
        sim_freq=513,
    ))
    demo_collection: DemoCollectionConfig = field(
        default_factory=DemoCollectionConfig
    )
    finetuning: FinetuningConfig = field(default_factory=FinetuningConfig)
    evaluation: DustEvalConfig = field(default_factory=DustEvalConfig)
    output_dir: str = "results/dust_robustness"


def load_dust_robustness_config(
    yaml_path: Optional[str] = None,
) -> DustRobustnessConfig:
    """Load dust robustness config from YAML, merging onto defaults."""
    if yaml_path is None:
        return DustRobustnessConfig()

    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    return dacite.from_dict(
        data_class=DustRobustnessConfig,
        data=raw,
        config=dacite.Config(strict=False, cast=[tuple]),
    )


def load_adversarial_config(
    yaml_path: Optional[str] = None,
) -> AdversarialExperimentConfig:
    """Load adversarial training config from YAML, merging onto defaults."""
    if yaml_path is None:
        return AdversarialExperimentConfig()

    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    return dacite.from_dict(
        data_class=AdversarialExperimentConfig,
        data=raw,
        config=dacite.Config(strict=False, cast=[tuple]),
    )


def load_envelope_config(
    yaml_path: Optional[str] = None,
) -> EnvelopeExperimentConfig:
    """Load safe operating envelope config from YAML, merging onto defaults."""
    if yaml_path is None:
        return EnvelopeExperimentConfig()

    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    return dacite.from_dict(
        data_class=EnvelopeExperimentConfig,
        data=raw,
        config=dacite.Config(strict=False, cast=[tuple]),
    )
