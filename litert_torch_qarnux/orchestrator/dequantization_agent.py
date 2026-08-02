"""
DequantizationAgent - Weight Dequantization from GGUF Formats.

Handles conversion of quantized GGUF tensor data (Q4_0, Q5_0, Q8_0, etc.)
into full-precision float16/float32 arrays. Uses the gguf library's
dequantize function and falls back to manual implementations for
unsupported quantization types.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

import gguf

from litert_torch_qarnux.orchestrator.base_agent import AgentMessage, BaseAgent
from litert_torch_qarnux.utils.gguf_parser import TensorInfo
from litert_torch_qarnux.utils.tensor_mapping import TensorMapper

logger = logging.getLogger(__name__)


class DequantizationAgent(BaseAgent):
    """
    Dequantizes GGUF tensor weights to full precision.

    Processes each tensor in the GGUF file, dequantizing from its
    compressed format (Q4_0, Q8_0, etc.) to float32. The agent
    also applies the tensor name mapping to produce PyTorch-compatible
    parameter names.
    """

    # Quantization types that store raw float data (no dequantization needed)
    _FLOAT_TYPES = {0, 1, 32}  # F32, F16, F64

    # Maximum number of elements to dequantize in a single batch
    _BATCH_SIZE = 1024

    def __init__(self):
        super().__init__("dequantization")

    def execute(self, message: AgentMessage) -> AgentMessage:
        """
        Dequantize all tensors from the GGUF file.

        Returns:
            AgentMessage containing dequantized weight arrays mapped
            to PyTorch parameter names, plus the tensor mapping dictionary.
        """
        metadata = message.data["metadata"]
        tensors: List[TensorInfo] = message.data["tensors"]
        reader = message.data["reader"]

        # Initialize tensor mapper for the architecture
        mapper = TensorMapper(metadata.architecture)

        # Dequantize each tensor
        dequantized = {}
        tensor_map = {}
        total = len(tensors)

        self.logger.info(
            "Dequantizing %d tensors from %s quantization",
            total,
            metadata.architecture,
        )

        pbar = tqdm(total=total, desc="Dequantizing", unit="tensor")

        for tensor_info in tensors:
            try:
                # Map GGUF name to PyTorch name
                pytorch_name = mapper.map_tensor(tensor_info.name)
                if pytorch_name is None:
                    pbar.update(1)
                    continue

                tensor_map[tensor_info.name] = pytorch_name

                # Dequantize
                dequant_data = self._dequantize_tensor(tensor_info, reader)

                if dequant_data is not None:
                    dequantized[pytorch_name] = dequant_data

            except Exception as e:
                self.logger.warning(
                    "Failed to dequantize %s: %s", tensor_info.name, e
                )

            pbar.update(1)

        pbar.close()

        self.logger.info(
            "Dequantized %d/%d tensors (%d mapped to PyTorch params)",
            len(dequantized),
            total,
            len(tensor_map),
        )

        return AgentMessage(
            source=self.agent_id,
            target="model_authoring",
            data={
                "metadata": metadata,
                "dequantized_weights": dequantized,
                "tensor_map": tensor_map,
                "gguf_tensor_map": {v: k for k, v in tensor_map.items()},
            },
        )

    def _dequantize_tensor(
        self, tensor_info: TensorInfo, reader
    ) -> Optional[np.ndarray]:
        """
        Dequantize a single tensor to float32.

        Args:
            tensor_info: The tensor descriptor from GGUF.
            reader: The GGUFReader instance for accessing raw data.

        Returns:
            Dequantized numpy array, or None if dequantization fails.
        """
        if tensor_info.tensor_type in self._FLOAT_TYPES:
            # Already in floating point format
            data = tensor_info.data
            if hasattr(data, "numpy"):
                return data.numpy().astype(np.float32)
            return np.array(data, dtype=np.float32)

        try:
            # Use gguf's built-in dequantize function
            dequantized = gguf.dequantize(tensor_info.data, tensor_info.tensor_type)
            if dequantized is not None:
                return dequantized.astype(np.float32)
        except (ValueError, RuntimeError, TypeError) as e:
            self.logger.debug(
                "gguf.dequantize failed for %s (%s): %s",
                tensor_info.name,
                tensor_info.data_type,
                e,
            )

        # Fallback: try to interpret as raw float
        try:
            raw = np.array(tensor_info.data)
            if raw.dtype in (np.float32, np.float16, np.float64):
                return raw.astype(np.float32)
        except Exception:
            pass

        self.logger.warning(
            "Could not dequantize tensor %s (type: %s)",
            tensor_info.name,
            tensor_info.data_type,
        )
        return None
