"""Tests for configuration module.

Tests configuration loading, validation, and serialization.
"""

import yaml

from surgical_segmentation.utils.config import (
    AugmentationConfig,
    ClassWeightsConfig,
    ColorJitterConfig,
    Config,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    HardwareConfig,
    LoggingConfig,
    ModelConfig,
    NormalizeConfig,
    PathsConfig,
    TrainingConfig,
    get_config,
    load_config,
    save_config,
)


class TestClassWeightsConfig:
    """Test ClassWeightsConfig dataclass."""

    def test_default_values(self):
        """Verify default class weights."""
        config = ClassWeightsConfig()
        assert config.background == 1.0
        assert config.instrument == 3.0

    def test_to_list_binary(self):
        """Verify to_list returns correct format for binary segmentation."""
        config = ClassWeightsConfig(background=1.0, instrument=5.0)
        weights = config.to_list(num_classes=2)

        assert len(weights) == 2
        assert weights[0] == 1.0
        assert weights[1] == 5.0

    def test_to_list_multiclass(self):
        """Verify to_list extends weights for multiclass."""
        config = ClassWeightsConfig(background=1.0, instrument=3.0)
        weights = config.to_list(num_classes=4)

        assert len(weights) == 4
        assert weights[0] == 1.0
        assert all(w == 3.0 for w in weights[1:])


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_values(self):
        """Verify default training hyperparameters."""
        config = TrainingConfig()

        assert config.epochs == 15
        assert config.batch_size == 4
        assert config.learning_rate == 0.0001
        assert config.weight_decay == 0.0001
        assert config.seed == 42

    def test_custom_values(self):
        """Verify custom values are stored correctly."""
        config = TrainingConfig(
            epochs=50,
            batch_size=8,
            learning_rate=0.001,
        )

        assert config.epochs == 50
        assert config.batch_size == 8
        assert config.learning_rate == 0.001


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_default_architecture(self):
        """Verify default architecture is DeepLabV3."""
        config = ModelConfig()
        assert config.architecture == "deeplabv3_resnet50"

    def test_default_num_classes(self):
        """Verify default is binary segmentation."""
        config = ModelConfig()
        assert config.num_classes == 2


class TestDataConfig:
    """Test DataConfig dataclass."""

    def test_default_image_size(self):
        """Verify default image size."""
        config = DataConfig()
        assert config.image_size == 256

    def test_default_normalization(self):
        """Verify ImageNet normalization values via NormalizeConfig."""
        config = DataConfig()
        assert len(config.normalize.mean) == 3
        assert len(config.normalize.std) == 3
        # ImageNet standard values
        assert config.normalize.mean == [0.485, 0.456, 0.406]
        assert config.normalize.std == [0.229, 0.224, 0.225]


class TestAugmentationConfig:
    """Test AugmentationConfig dataclass."""

    def test_horizontal_flip_probability(self):
        """Verify default horizontal flip probability."""
        config = AugmentationConfig()
        assert config.horizontal_flip_prob == 0.5

    def test_augmentation_parameters(self):
        """Verify default augmentation parameters."""
        config = AugmentationConfig()
        assert config.horizontal_flip_prob == 0.5
        assert config.vertical_flip_prob == 0.3
        assert config.rotation_degrees == 15
        assert config.gaussian_noise_std == 0.02

    def test_color_jitter_defaults(self):
        """Verify default color jitter parameters."""
        config = AugmentationConfig()
        assert config.color_jitter.brightness == 0.2
        assert config.color_jitter.contrast == 0.2
        assert config.color_jitter.saturation == 0.2
        assert config.color_jitter.hue == 0.1


class TestColorJitterConfig:
    """Test ColorJitterConfig dataclass."""

    def test_default_values(self):
        """Verify default color jitter values."""
        config = ColorJitterConfig()
        assert config.brightness == 0.2
        assert config.contrast == 0.2
        assert config.saturation == 0.2
        assert config.hue == 0.1

    def test_custom_values(self):
        """Verify custom color jitter values."""
        config = ColorJitterConfig(brightness=0.5, contrast=0.5)
        assert config.brightness == 0.5
        assert config.contrast == 0.5


class TestNormalizeConfig:
    """Test NormalizeConfig dataclass."""

    def test_imagenet_defaults(self):
        """Verify ImageNet normalization defaults."""
        config = NormalizeConfig()
        assert config.mean == [0.485, 0.456, 0.406]
        assert config.std == [0.229, 0.224, 0.225]


class TestHardwareConfig:
    """Test HardwareConfig dataclass."""

    def test_default_uses_gpu_if_available(self):
        """Verify default doesn't force CPU."""
        config = HardwareConfig()
        assert config.force_cpu is False

    def test_get_device_with_force_cpu(self):
        """Verify force_cpu returns cpu device."""
        config = HardwareConfig(force_cpu=True)
        assert config.get_device() == "cpu"

    def test_get_device_with_specific_cuda(self):
        """Verify specific CUDA device selection."""
        import torch

        config = HardwareConfig(cuda_device=0)

        if torch.cuda.is_available():
            assert config.get_device() == "cuda:0"
        else:
            assert config.get_device() == "cpu"


class TestConfig:
    """Test master Config dataclass."""

    def test_default_initialization(self):
        """Verify Config initializes with all defaults."""
        config = Config()

        assert config.training is not None
        assert config.model is not None
        assert config.data is not None
        assert config.augmentation is not None
        assert config.paths is not None
        assert config.evaluation is not None
        assert config.logging is not None
        assert config.hardware is not None
        assert config.experiment is not None

    def test_nested_config_access(self):
        """Verify nested config values are accessible."""
        config = Config()

        assert config.training.epochs == 15
        assert config.model.num_classes == 2
        assert config.data.image_size == 256


class TestLoadConfig:
    """Test load_config function."""

    def test_load_from_yaml_file(self, tmp_path):
        """Verify loading config from YAML file."""
        config_content = """
training:
  epochs: 25
  batch_size: 8
model:
  architecture: unet
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        config = load_config(config_file)

        assert config.training.epochs == 25
        assert config.training.batch_size == 8
        assert config.model.architecture == "unet"

    def test_load_nonexistent_file_returns_defaults(self, tmp_path):
        """Verify missing config file returns default config."""
        config = load_config(tmp_path / "nonexistent.yaml")
        # Should return default config, not raise
        assert isinstance(config, Config)
        assert config.training.epochs == 15  # Default value

    def test_load_default_config(self):
        """Verify loading default config from config/default.yaml."""
        from pathlib import Path

        default_path = Path("config/default.yaml")
        if default_path.exists():
            config = load_config(default_path)
            assert config is not None
            assert isinstance(config, Config)


class TestSaveConfig:
    """Test save_config function."""

    def test_save_creates_yaml_file(self, tmp_path):
        """Verify save_config creates a valid YAML file."""
        config = Config()
        output_path = tmp_path / "saved_config.yaml"

        save_config(config, output_path)

        assert output_path.exists()
        loaded = yaml.safe_load(output_path.read_text())
        assert "training" in loaded
        assert "model" in loaded

    def test_round_trip_preserves_values(self, tmp_path):
        """Verify save/load round trip preserves config values."""
        original = Config()
        original.training.epochs = 100
        original.model.num_classes = 5

        output_path = tmp_path / "roundtrip.yaml"
        save_config(original, output_path)
        loaded = load_config(output_path)

        assert loaded.training.epochs == 100
        assert loaded.model.num_classes == 5


class TestGetConfig:
    """Test get_config global configuration access."""

    def test_returns_config_instance(self):
        """Verify get_config returns a Config instance."""
        config = get_config()
        assert isinstance(config, Config)

    def test_returns_same_instance(self):
        """Verify get_config returns the same global instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2


class TestExperimentConfig:
    """Test ExperimentConfig dataclass."""

    def test_default_name(self):
        """Verify default experiment name."""
        config = ExperimentConfig()
        assert "surgical" in config.name.lower()

    def test_default_tags(self):
        """Verify default tags include relevant keywords."""
        config = ExperimentConfig()
        assert "medical-imaging" in config.tags
        assert "segmentation" in config.tags


class TestPathsConfig:
    """Test PathsConfig dataclass."""

    def test_default_output_dir(self):
        """Verify default output directory."""
        config = PathsConfig()
        assert config.output_dir is not None

    def test_model_path_property(self):
        """Verify model_path property returns correct path."""
        from pathlib import Path

        config = PathsConfig(model_dir="models", model_filename="test.pth")
        assert config.model_path == Path("models") / "test.pth"

    def test_ensure_dirs_exist_creates_directories(self, tmp_path):
        """Verify ensure_dirs_exist creates all necessary directories."""
        config = PathsConfig(
            output_dir=str(tmp_path / "outputs"),
            model_dir=str(tmp_path / "outputs/models"),
            figures_dir=str(tmp_path / "outputs/figures"),
            predictions_dir=str(tmp_path / "preds"),
        )

        config.ensure_dirs_exist()

        assert (tmp_path / "outputs").exists()
        assert (tmp_path / "outputs/models").exists()
        assert (tmp_path / "outputs/figures").exists()
        assert (tmp_path / "preds").exists()


class TestEvaluationConfig:
    """Test EvaluationConfig dataclass."""

    def test_default_metrics(self):
        """Verify default evaluation metrics."""
        config = EvaluationConfig()
        assert "iou" in config.metrics
        assert "dice" in config.metrics


class TestLoggingConfig:
    """Test LoggingConfig dataclass."""

    def test_default_log_level(self):
        """Verify default log level."""
        config = LoggingConfig()
        assert config.level == "INFO"


class TestDictToDataclass:
    """Test _dict_to_dataclass helper function."""

    def test_non_dataclass_returns_data(self):
        """Verify non-dataclass input returns data unchanged."""
        from surgical_segmentation.utils.config import _dict_to_dataclass

        data = {"key": "value"}
        result = _dict_to_dataclass(data, dict)
        assert result == data

    def test_dataclass_conversion(self):
        """Verify dict is converted to dataclass."""
        from surgical_segmentation.utils.config import _dict_to_dataclass

        data = {"epochs": 50, "batch_size": 16}
        result = _dict_to_dataclass(data, TrainingConfig)
        assert result.epochs == 50
        assert result.batch_size == 16

    def test_nested_dataclass(self):
        """Verify _dict_to_dataclass handles simple dataclass."""
        from surgical_segmentation.utils.config import NormalizeConfig, _dict_to_dataclass

        # Test converting directly to NormalizeConfig
        data = {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.25, 0.25, 0.25],
        }
        result = _dict_to_dataclass(data, NormalizeConfig)
        assert result.mean == [0.5, 0.5, 0.5]
        assert result.std == [0.25, 0.25, 0.25]


class TestLoadConfigWithOverrides:
    """Test load_config with override dictionary."""

    def test_nested_key_override(self, tmp_path):
        """Verify nested key override works."""
        from surgical_segmentation.utils.config import load_config

        config = load_config(override={"training.epochs": 100})
        assert config.training.epochs == 100

    def test_override_preserves_other_values(self, tmp_path):
        """Verify override doesn't affect other values."""
        from surgical_segmentation.utils.config import load_config

        config = load_config(override={"training.batch_size": 32})
        assert config.training.batch_size == 32
        # Other values should still have defaults
        assert config.training.learning_rate is not None
