"""Tests for the safe operating envelope predictor (no GPU/SimplerEnv needed).

Tests the config, model factory, zone classification, serialization,
and visualization helpers.
"""

import json
import tempfile

import numpy as np
import pytest

from adversarial_dust.config import (
    EnvelopeConfig,
    EnvelopeExperimentConfig,
    FingerprintConfig,
    GlareConfig,
    DustGridConfig,
    BlobConfig,
    load_envelope_config,
)
from adversarial_dust.envelope_predictor import (
    make_occlusion_model,
    SafeOperatingEnvelopePredictor,
    _make_serializable,
)
from adversarial_dust.fingerprint_model import FingerprintSmudgeModel
from adversarial_dust.glare_model import AdversarialGlareModel
from adversarial_dust.dust_model import AdversarialDustModel
from adversarial_dust.blob_model import DynamicBlobDustModel


@pytest.fixture
def config():
    return EnvelopeExperimentConfig()


@pytest.fixture
def image_shape():
    return (256, 256, 3)


class TestMakeOcclusionModel:
    def test_fingerprint(self, config, image_shape):
        model = make_occlusion_model("fingerprint", config, image_shape, 0.1)
        assert isinstance(model, FingerprintSmudgeModel)

    def test_glare(self, config, image_shape):
        model = make_occlusion_model("glare", config, image_shape, 0.1)
        assert isinstance(model, AdversarialGlareModel)

    def test_grid(self, config, image_shape):
        model = make_occlusion_model("grid", config, image_shape, 0.1)
        assert isinstance(model, AdversarialDustModel)

    def test_blob(self, config, image_shape):
        model = make_occlusion_model("blob", config, image_shape, 0.1)
        assert isinstance(model, DynamicBlobDustModel)

    def test_unknown_raises(self, config, image_shape):
        with pytest.raises(ValueError, match="Unknown occlusion type"):
            make_occlusion_model("waterfall", config, image_shape, 0.1)


class TestZoneClassification:
    def test_safe(self, config, image_shape):
        predictor = SafeOperatingEnvelopePredictor(config, None, image_shape)
        assert predictor._classify_zone(1.0) == "safe"
        assert predictor._classify_zone(0.95) == "safe"

    def test_marginal(self, config, image_shape):
        predictor = SafeOperatingEnvelopePredictor(config, None, image_shape)
        assert predictor._classify_zone(0.80) == "marginal"
        assert predictor._classify_zone(0.70) == "marginal"

    def test_unsafe(self, config, image_shape):
        predictor = SafeOperatingEnvelopePredictor(config, None, image_shape)
        assert predictor._classify_zone(0.69) == "unsafe"
        assert predictor._classify_zone(0.0) == "unsafe"


class TestEnvelopeSummary:
    def test_summary_computation(self, config, image_shape):
        predictor = SafeOperatingEnvelopePredictor(config, None, image_shape)
        type_results = {
            "0.01": {"zone": "safe"},
            "0.05": {"zone": "safe"},
            "0.1": {"zone": "marginal"},
            "0.2": {"zone": "unsafe"},
        }
        summary = predictor._compute_envelope_summary(type_results)
        assert summary["max_safe_budget"] == 0.05
        assert summary["breakdown_budget"] == 0.2

    def test_all_safe(self, config, image_shape):
        predictor = SafeOperatingEnvelopePredictor(config, None, image_shape)
        type_results = {
            "0.1": {"zone": "safe"},
            "0.2": {"zone": "safe"},
        }
        summary = predictor._compute_envelope_summary(type_results)
        assert summary["max_safe_budget"] == 0.2
        assert summary["breakdown_budget"] == float("inf")

    def test_all_unsafe(self, config, image_shape):
        predictor = SafeOperatingEnvelopePredictor(config, None, image_shape)
        type_results = {
            "0.01": {"zone": "unsafe"},
        }
        summary = predictor._compute_envelope_summary(type_results)
        assert summary["max_safe_budget"] == 0.0
        assert summary["breakdown_budget"] == 0.01


class TestSerialization:
    def test_numpy_arrays(self):
        data = {"arr": np.array([1.0, 2.0]), "val": np.float64(3.14)}
        result = _make_serializable(data)
        assert isinstance(result["arr"], list)
        assert isinstance(result["val"], float)
        # Should be JSON-serializable
        json.dumps(result)

    def test_nested(self):
        data = {
            "a": {"b": np.int64(5)},
            "c": [np.float32(1.0), np.float32(2.0)],
        }
        result = _make_serializable(data)
        json.dumps(result)

    def test_inf(self):
        data = {"x": float("inf")}
        result = _make_serializable(data)
        assert result["x"] == "inf"
        json.dumps(result)


class TestLoadConfig:
    def test_default(self):
        config = load_envelope_config(None)
        assert isinstance(config, EnvelopeExperimentConfig)
        assert config.env.task_name == "widowx_put_eggplant_in_basket"
        assert "fingerprint" in config.envelope.occlusion_types
        assert "glare" in config.envelope.occlusion_types

    def test_from_yaml(self, tmp_path):
        yaml_content = """
env:
  task_name: "google_robot_pick_coke_can"
  policy_model: "octo-base"
envelope:
  occlusion_types:
    - glare
  budget_levels:
    - 0.05
    - 0.10
  safe_threshold: 0.90
output_dir: "results/test"
"""
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml_content)
        config = load_envelope_config(str(yaml_path))
        assert config.env.task_name == "google_robot_pick_coke_can"
        assert config.envelope.occlusion_types == ["glare"]
        assert config.envelope.safe_threshold == 0.90


class TestModelAPIConsistency:
    """Verify all occlusion models expose the same API surface."""

    @pytest.mark.parametrize("model_type", ["fingerprint", "glare", "grid", "blob"])
    def test_has_required_methods(self, config, image_shape, model_type):
        model = make_occlusion_model(model_type, config, image_shape, 0.1)
        assert hasattr(model, "apply")
        assert hasattr(model, "get_alpha_mask")
        assert hasattr(model, "compute_coverage")
        assert hasattr(model, "get_cma_bounds")
        assert hasattr(model, "get_cma_x0")
        assert hasattr(model, "get_random_params")

    @pytest.mark.parametrize("model_type", ["fingerprint", "glare", "grid", "blob"])
    def test_apply_returns_correct_shape(self, config, image_shape, model_type):
        model = make_occlusion_model(model_type, config, image_shape, 0.1)
        sample = np.random.default_rng(0).integers(0, 256, size=image_shape, dtype=np.uint8)
        params = model.get_cma_x0()
        result = model.apply(sample, params)
        assert result.shape == image_shape
        assert result.dtype == np.uint8
