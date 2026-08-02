"""
Quantization Configuration.

Defines available quantization recipes and their parameters for the
PyTorch-to-TFLite conversion step. Each recipe specifies the weight
quantization method, activation data type, and target precision.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, Optional


class QuantizationMethod(str, Enum):
    """Supported quantization methods."""
    NONE = "none"
    WEIGHT_ONLY = "weight_only"
    DYNAMIC = "dynamic"
    FULL = "full"
    FLOAT8 = "float8"


class QuantizationDType(str, Enum):
    """Target quantization data types."""
    INT4 = "int4"
    INT8 = "int8"
    FP8 = "fp8"
    FP16 = "fp16"


class QuantizationConfig:
    """
    Configuration for model quantization during TFLite conversion.

    Attributes:
        method: The quantization method to apply.
        weight_dtype: Target data type for weight quantization.
        activation_dtype: Data type for activations during inference.
        per_channel: Whether to use per-channel quantization.
        group_size: Quantization group size (for grouped quantization).
        custom_recipe_path: Optional path to a custom quantization recipe JSON.
    """

    # Predefined recipe catalog
    _RECIPES: Dict[str, Dict[str, Any]] = {
        "none": {
            "method": QuantizationMethod.NONE,
        },
        "dynamic_wi8_afp32": {
            "method": QuantizationMethod.WEIGHT_ONLY,
            "weight_dtype": QuantizationDType.INT8,
            "activation_dtype": "fp32",
            "per_channel": True,
        },
        "dynamic_wi4_afp32": {
            "method": QuantizationMethod.WEIGHT_ONLY,
            "weight_dtype": QuantizationDType.INT4,
            "activation_dtype": "fp32",
            "per_channel": True,
        },
        "full_int8": {
            "method": QuantizationMethod.FULL,
            "weight_dtype": QuantizationDType.INT8,
            "activation_dtype": "int8",
            "per_channel": True,
        },
        "float8": {
            "method": QuantizationMethod.FLOAT8,
            "weight_dtype": QuantizationDType.FP8,
            "activation_dtype": "fp16",
            "per_channel": False,
        },
    }

    def __init__(
        self,
        method: QuantizationMethod = QuantizationMethod.WEIGHT_ONLY,
        weight_dtype: QuantizationDType = QuantizationDType.INT8,
        activation_dtype: str = "fp32",
        per_channel: bool = True,
        group_size: int = 32,
        custom_recipe_path: Optional[str] = None,
    ):
        self.method = method
        self.weight_dtype = weight_dtype
        self.activation_dtype = activation_dtype
        self.per_channel = per_channel
        self.group_size = group_size
        self.custom_recipe_path = custom_recipe_path

    @classmethod
    def from_recipe(cls, recipe_name: str) -> "QuantizationConfig":
        """
        Create a QuantizationConfig from a named recipe.

        Args:
            recipe_name: Name of the quantization recipe.

        Returns:
            A populated QuantizationConfig instance.

        Raises:
            ValueError: If the recipe name is not recognized.
        """
        if recipe_name not in cls._RECIPES:
            raise ValueError(
                f"Unknown quantization recipe: '{recipe_name}'. "
                f"Available: {list(cls._RECIPES.keys())}"
            )
        params = cls._RECIPES[recipe_name]
        return cls(**params)

    @classmethod
    def from_file(cls, file_path: str) -> "QuantizationConfig":
        """
        Load a quantization recipe from a JSON file.

        Args:
            file_path: Path to the JSON recipe file.

        Returns:
            A QuantizationConfig instance loaded from the file.
        """
        with open(file_path, "r") as f:
            params = json.load(f)
        return cls(**params)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            "method": self.method.value if isinstance(self.method, Enum) else self.method,
            "weight_dtype": self.weight_dtype.value if isinstance(self.weight_dtype, Enum) else self.weight_dtype,
            "activation_dtype": self.activation_dtype,
            "per_channel": self.per_channel,
            "group_size": self.group_size,
        }

    @classmethod
    def list_recipes(cls) -> list:
        """Return all available recipe names."""
        return list(cls._RECIPES.keys())

    def __repr__(self) -> str:
        return (
            f"QuantizationConfig(method={self.method}, "
            f"weight_dtype={self.weight_dtype}, "
            f"activation_dtype={self.activation_dtype})"
        )
