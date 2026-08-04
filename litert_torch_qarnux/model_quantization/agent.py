"""
Quantization Agent.

A pipeline agent that wraps ModelQuantizer for use within the
multi-agent orchestration framework.
"""
from __future__ import annotations

import logging
from typing import Any

from litert_torch_qarnux.model_quantization.config import QuantizationProfile
from litert_torch_qarnux.model_quantization.quantizer import ModelQuantizer
from litert_torch_qarnux.orchestrator.base_agent import BaseAgent, AgentMessage, AgentStatus

logger = logging.getLogger(__name__)


class QuantizationAgent(BaseAgent):
    """
    Agent that performs model quantization within the orchestration pipeline.

    Receives a model path and quantization profile via AgentMessage,
    runs the quantization pipeline, and returns results.
    """

    def __init__(self, profile: QuantizationProfile):
        super().__init__("quantization_agent")
        self.profile = profile
        self.quantizer = ModelQuantizer(profile)

    def execute(self, message: AgentMessage) -> AgentMessage:
        """
        Execute the quantization step.

        The input message should contain:
            - data["model_path"]: Path to the input model file
            - data["behavior_profile"]: Optional BehaviorProfile dict
            - data["template_config"]: Optional TemplateConfig dict
        """
        # Override input path if provided in message
        if "model_path" in message.data:
            self.profile.input_path = message.data["model_path"]

        # Merge behavior data if provided
        if "behavior_profile" in message.data:
            from litert_torch_qarnux.model_quantization.config import BehaviorProfile
            bp = BehaviorProfile.from_dict(message.data["behavior_profile"])
            self.profile.behavior = bp

        # Merge template data if provided
        if "template_config" in message.data:
            from litert_torch_qarnux.model_quantization.config import TemplateConfig
            tc = TemplateConfig.from_dict(message.data["template_config"])
            self.profile.template = tc

        self.logger.info("Starting quantization for %s", self.profile.input_path)

        try:
            result = self.quantizer.quantize()
            return AgentMessage(
                source=self.agent_id,
                target="orchestrator",
                data={
                    "quantized_path": result.get("output_path"),
                    "format": result.get("format"),
                    "file_size_bytes": result.get("file_size_bytes", 0),
                    "metadata": result.get("metadata", {}),
                },
                success=True,
            )
        except Exception as e:
            self.logger.error("Quantization failed: %s", e)
            return AgentMessage(
                source=self.agent_id,
                target="orchestrator",
                success=False,
                error_message=f"Quantization failed: {e}",
            )
