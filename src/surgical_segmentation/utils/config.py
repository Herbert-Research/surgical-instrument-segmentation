"""
Configuration management for surgical instrument segmentation.

This module provides utilities for loading, validating, and accessing
configuration parameters from YAML files. Centralizes all hyperparameters
and settings for reproducible training and evaluation.

Example:
    >>> from surgical_segmentation.utils.config import load_config, get_config
    >>> config = load_config("config/default.yaml")
    >>> print(config.training.epochs)
    15
    >>> print(config.model.architecture)
    deeplabv3_resnet50
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


@dataclass
class ClassWeightsConfig:
    """Class weight configuration for handling class imbalance."""

    background: float = 1.0
    instrument: float = 3.0

    def to_list(self, num_classes: int = 2) -> list[float]:
        """Convert to list format for CrossEntropyLoss weight parameter."""
        if num_classes == 2:
            return [self.background, self.instrument]
        # For multi-class, extend with instrument weight
        return [self.background] + [self.instrument] * (num_classes - 1)


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""

    epochs: int = 15
    batch_size: int = 4
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    seed: int = 42
    class_weights: ClassWeightsConfig = field(default_factory=ClassWeightsConfig)
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    architecture: str = "deeplabv3_resnet50"
    num_classes: int = 2
    pretrained: bool = True
    dropout: float = 0.1


@dataclass
class NormalizeConfig:
    """Image normalization configuration (ImageNet statistics)."""

    mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""

    image_size: int = 256
    augment: bool = True
    train_split: float = 0.8
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    cholec_instrument_classes: list[int] = field(default_factory=lambda: [31, 32])


@dataclass
class ColorJitterConfig:
    """Color jitter augmentation parameters."""

    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""

    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.3
    rotation_degrees: int = 15
    color_jitter: ColorJitterConfig = field(default_factory=ColorJitterConfig)
    gaussian_noise_std: float = 0.02


@dataclass
class PathsConfig:
    """File and directory paths configuration."""

    frame_dir: str = "data/sample_frames"
    mask_dir: str = "data/masks"
    output_dir: str = "outputs"
    model_dir: str = "outputs/models"
    figures_dir: str = "outputs/figures"
    predictions_dir: str = "data/preds"
    model_filename: str = "instrument_segmentation_model.pth"

    @property
    def model_path(self) -> Path:
        """Full path to the model checkpoint file."""
        return Path(self.model_dir) / self.model_filename

    def ensure_dirs_exist(self) -> None:
        """Create all output directories if they don't exist."""
        for dir_path in [self.output_dir, self.model_dir, self.figures_dir, self.predictions_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@dataclass
class EvaluationConfig:
    """Evaluation settings configuration."""

    metrics: list[str] = field(
        default_factory=lambda: ["accuracy", "iou", "dice", "precision", "recall"]
    )
    threshold: float = 0.5
    save_predictions: bool = True
    generate_figures: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "outputs/training.log"
    log_interval: int = 10


@dataclass
class HardwareConfig:
    """Hardware and compute configuration."""

    force_cpu: bool = False
    cuda_device: int = -1
    mixed_precision: bool = False

    def get_device(self) -> str:
        """Determine the compute device based on configuration and availability."""
        import torch

        if self.force_cpu:
            return "cpu"

        if not torch.cuda.is_available():
            return "cpu"

        if self.cuda_device >= 0:
            return f"cuda:{self.cuda_device}"

        return "cuda"


@dataclass
class ExperimentConfig:
    """Experiment metadata configuration."""

    name: str = "surgical_instrument_segmentation"
    description: str = "Binary segmentation of surgical instruments in laparoscopic video"
    tags: list[str] = field(
        default_factory=lambda: [
            "medical-imaging",
            "segmentation",
            "cholecystectomy",
            "deep-learning",
        ]
    )


@dataclass
class Config:
    """
    Master configuration container for all settings.

    Aggregates all configuration sections into a single, validated
    configuration object. Supports loading from YAML files and
    provides sensible defaults for all parameters.

    Attributes:
        training: Training hyperparameters (epochs, batch_size, lr, etc.)
        model: Model architecture settings
        data: Data loading and preprocessing settings
        augmentation: Data augmentation parameters
        paths: File and directory paths
        evaluation: Evaluation settings
        logging: Logging configuration
        hardware: Hardware and compute settings
        experiment: Experiment metadata

    Example:
        >>> config = Config()  # Use all defaults
        >>> config.training.epochs
        15
        >>> config.model.architecture
        'deeplabv3_resnet50'
    """

    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


# Global configuration instance (lazy loaded)
_global_config: Config | None = None


def _dict_to_dataclass(data: dict[str, Any], cls: type) -> Any:
    """
    Recursively convert a dictionary to a dataclass instance.

    Handles nested dataclasses by inspecting field types and recursively
    converting nested dictionaries to their corresponding dataclass types.

    Args:
        data: Dictionary containing configuration values
        cls: Target dataclass type

    Returns:
        Instance of the target dataclass populated with values from data
    """
    from dataclasses import fields, is_dataclass

    if not is_dataclass(cls):
        return data

    field_types = {f.name: f.type for f in fields(cls)}
    kwargs = {}

    for field_name, field_type in field_types.items():
        if field_name in data:
            value = data[field_name]

            # Handle nested dataclasses
            if is_dataclass(field_type) and isinstance(value, dict):
                kwargs[field_name] = _dict_to_dataclass(value, field_type)
            else:
                kwargs[field_name] = value

    return cls(**kwargs)


def load_config(
    config_path: str | Path | None = None,
    override: dict[str, Any] | None = None,
) -> Config:
    """
    Load configuration from a YAML file.

    Reads a YAML configuration file and returns a validated Config object.
    Missing values are filled with sensible defaults. Optionally applies
    override values from a dictionary.

    Args:
        config_path: Path to YAML configuration file. If None, uses
                     default path "config/default.yaml". If file doesn't
                     exist, returns default configuration.
        override: Optional dictionary of values to override after loading.
                  Supports nested keys using dot notation (e.g.,
                  "training.epochs": 20).

    Returns:
        Config: Validated configuration object with all parameters.

    Raises:
        yaml.YAMLError: If the configuration file contains invalid YAML.

    Example:
        >>> # Load default configuration
        >>> config = load_config()

        >>> # Load custom configuration
        >>> config = load_config("config/experiment_01.yaml")

        >>> # Load with overrides
        >>> config = load_config(override={"training.epochs": 30})
    """
    global _global_config

    # Determine config path
    if config_path is None:
        config_path = Path("config/default.yaml")
    else:
        config_path = Path(config_path)

    # Start with default config
    config_dict: dict[str, Any] = {}

    # Load from file if it exists
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if loaded:
                config_dict = loaded

    # Apply overrides
    if override:
        for key, value in override.items():
            keys = key.split(".")
            target = config_dict
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            target[keys[-1]] = value

    # Convert to Config dataclass
    config = Config()

    if "training" in config_dict:
        training_dict = config_dict["training"]
        if "class_weights" in training_dict:
            training_dict["class_weights"] = ClassWeightsConfig(**training_dict["class_weights"])
        config.training = TrainingConfig(
            **{k: v for k, v in training_dict.items() if k in TrainingConfig.__dataclass_fields__}
        )

    if "model" in config_dict:
        config.model = ModelConfig(
            **{
                k: v
                for k, v in config_dict["model"].items()
                if k in ModelConfig.__dataclass_fields__
            }
        )

    if "data" in config_dict:
        data_dict = config_dict["data"]
        if "normalize" in data_dict:
            data_dict["normalize"] = NormalizeConfig(**data_dict["normalize"])
        config.data = DataConfig(
            **{k: v for k, v in data_dict.items() if k in DataConfig.__dataclass_fields__}
        )

    if "augmentation" in config_dict:
        aug_dict = config_dict["augmentation"]
        if "color_jitter" in aug_dict:
            aug_dict["color_jitter"] = ColorJitterConfig(**aug_dict["color_jitter"])
        config.augmentation = AugmentationConfig(
            **{k: v for k, v in aug_dict.items() if k in AugmentationConfig.__dataclass_fields__}
        )

    if "paths" in config_dict:
        config.paths = PathsConfig(
            **{
                k: v
                for k, v in config_dict["paths"].items()
                if k in PathsConfig.__dataclass_fields__
            }
        )

    if "evaluation" in config_dict:
        config.evaluation = EvaluationConfig(
            **{
                k: v
                for k, v in config_dict["evaluation"].items()
                if k in EvaluationConfig.__dataclass_fields__
            }
        )

    if "logging" in config_dict:
        config.logging = LoggingConfig(
            **{
                k: v
                for k, v in config_dict["logging"].items()
                if k in LoggingConfig.__dataclass_fields__
            }
        )

    if "hardware" in config_dict:
        config.hardware = HardwareConfig(
            **{
                k: v
                for k, v in config_dict["hardware"].items()
                if k in HardwareConfig.__dataclass_fields__
            }
        )

    if "experiment" in config_dict:
        config.experiment = ExperimentConfig(
            **{
                k: v
                for k, v in config_dict["experiment"].items()
                if k in ExperimentConfig.__dataclass_fields__
            }
        )

    # Cache globally
    _global_config = config

    return config


def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns the cached configuration if load_config() has been called,
    otherwise loads the default configuration.

    Returns:
        Config: Global configuration instance.

    Example:
        >>> config = get_config()
        >>> print(config.training.epochs)
    """
    global _global_config

    if _global_config is None:
        _global_config = load_config()

    return _global_config


def save_config(config: Config, path: str | Path) -> None:
    """
    Save configuration to a YAML file.

    Serializes the configuration object to YAML format for experiment
    reproducibility and documentation.

    Args:
        config: Configuration object to save.
        path: Output file path for the YAML configuration.

    Example:
        >>> config = load_config()
        >>> config.training.epochs = 30
        >>> save_config(config, "config/experiment_30epochs.yaml")
    """
    from dataclasses import asdict

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = asdict(config)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


__all__ = [
    "Config",
    "TrainingConfig",
    "ModelConfig",
    "DataConfig",
    "AugmentationConfig",
    "PathsConfig",
    "EvaluationConfig",
    "LoggingConfig",
    "HardwareConfig",
    "ExperimentConfig",
    "load_config",
    "get_config",
    "save_config",
]
