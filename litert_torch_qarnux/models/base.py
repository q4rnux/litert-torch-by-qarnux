"""
Base Model Definition and Registry.

Provides the abstract base class for all supported model architectures
and a registry that maps GGUF architecture strings to their corresponding
PyTorch model classes. Each architecture implementation must subclass
BaseModel and register itself for automatic discovery.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, Optional, Type

import numpy as np

logger = logging.getLogger(__name__)


class BaseModel(abc.ABC):
    """
    Abstract base class for all supported model architectures.

    Each concrete model class must implement methods to build the model
    architecture from GGUF metadata and load dequantized weights into
    the appropriate parameters.
    """

    # Architecture identifiers this class handles
    ARCHITECTURES: list = []

    def __init__(self, metadata: Any):
        """
        Initialize the model with GGUF-extracted metadata.

        Args:
            metadata: A GGUFMetadata instance containing architecture and
                     hyperparameter information.
        """
        self.metadata = metadata
        self.model = None

    @abc.abstractmethod
    def build(self) -> Any:
        """
        Construct the PyTorch nn.Module architecture.

        Returns:
            A fully constructed PyTorch model instance (not yet loaded with weights).
        """
        ...

    @abc.abstractmethod
    def load_weights(
        self,
        tensor_map: Dict[str, str],
        dequantized_tensors: Dict[str, np.ndarray],
    ) -> None:
        """
        Load dequantized weight tensors into the PyTorch model.

        Args:
            tensor_map: Mapping from GGUF tensor names to PyTorch parameter names.
            dequantized_tensors: Dictionary mapping PyTorch parameter names to
                               dequantized numpy arrays.
        """
        ...

    def get_model(self) -> Any:
        """
        Return the constructed PyTorch model.

        Returns:
            The PyTorch nn.Module instance.
        """
        if self.model is None:
            self.model = self.build()
        return self.model


class ModelRegistry:
    """
    Central registry that maps GGUF architecture strings to model classes.

    New architectures can be added by calling register() or by using
    the ARCHITECTURES class attribute on BaseModel subclasses.
    """

    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, architecture: str, model_class: Type[BaseModel]) -> None:
        """
        Register a model class for a given architecture string.

        Args:
            architecture: The GGUF architecture identifier (e.g., "llama").
            model_class: The BaseModel subclass that handles this architecture.
        """
        cls._registry[architecture.lower()] = model_class
        logger.debug("Registered model class %s for architecture '%s'", model_class.__name__, architecture)

    @classmethod
    def get_model_class(cls, architecture: str) -> Optional[Type[BaseModel]]:
        """
        Look up the model class for a given architecture.

        Args:
            architecture: The GGUF architecture identifier.

        Returns:
            The corresponding BaseModel subclass, or None if not registered.
        """
        return cls._registry.get(architecture.lower())

    @classmethod
    def create_model(cls, architecture: str, metadata: Any) -> Optional[BaseModel]:
        """
        Instantiate a model for the given architecture.

        Args:
            architecture: The GGUF architecture identifier.
            metadata: GGUFMetadata instance with model hyperparameters.

        Returns:
            A BaseModel instance, or None if the architecture is unsupported.
        """
        model_class = cls.get_model_class(architecture)
        if model_class is not None:
            return model_class(metadata)
        logger.warning("Unsupported architecture: %s", architecture)
        return None

    @classmethod
    def list_architectures(cls) -> list:
        """Return all registered architecture strings."""
        return sorted(cls._registry.keys())
