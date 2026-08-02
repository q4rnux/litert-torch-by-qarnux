"""
ModelAuthoringAgent - PyTorch Model Construction and Weight Loading.

Takes dequantized weights and architecture metadata to construct a
complete PyTorch nn.Module with all weights loaded. This agent
selects the appropriate model class from the registry and orchestrates
the build and weight-loading process.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from litert_torch_qarnux.orchestrator.base_agent import AgentMessage, BaseAgent
from litert_torch_qarnux.models.base import ModelRegistry

logger = logging.getLogger(__name__)


class ModelAuthoringAgent(BaseAgent):
    """
    Constructs a PyTorch model from dequantized GGUF weights.

    Uses the ModelRegistry to find the appropriate architecture class,
    builds the model structure, and loads the dequantized weight tensors.
    """

    def __init__(self):
        super().__init__("model_authoring")

    def execute(self, message: AgentMessage) -> AgentMessage:
        """
        Build and load the PyTorch model.

        Returns:
            AgentMessage containing the constructed PyTorch model and
            associated metadata.
        """
        metadata = message.data["metadata"]
        dequantized_weights = message.data["dequantized_weights"]
        tensor_map = message.data["tensor_map"]

        architecture = metadata.architecture
        self.logger.info(
            "Authoring model for architecture: %s (%d weights to load)",
            architecture,
            len(dequantized_weights),
        )

        # Look up the model class
        model_instance = ModelRegistry.create_model(architecture, metadata)

        if model_instance is None:
            raise ValueError(
                f"Unsupported architecture: '{architecture}'. "
                f"Supported architectures: {ModelRegistry.list_architectures()}"
            )

        # Build the model architecture
        model = model_instance.get_model()

        # Load dequantized weights
        model_instance.load_weights(tensor_map, dequantized_weights)

        self.logger.info("Model authoring complete for architecture: %s", architecture)
        self.logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

        return AgentMessage(
            source=self.agent_id,
            target="conversion",
            data={
                "metadata": metadata,
                "model": model,
                "model_instance": model_instance,
                "architecture": architecture,
                "weight_count": len(dequantized_weights),
            },
        )
