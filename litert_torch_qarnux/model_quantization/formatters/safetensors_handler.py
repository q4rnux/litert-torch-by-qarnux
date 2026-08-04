"""
SafeTensors Format Handler.

Handles reading, quantizing, and writing SafeTensors model files.
SafeTensors is a safe format for storing tensors without pickle.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from litert_torch_qarnux.model_quantization.config import QuantizationProfile
from litert_torch_qarnux.model_quantization.formatters.base import BaseFormatHandler

logger = logging.getLogger(__name__)


class SafeTensorsFormatHandler(BaseFormatHandler):
    """Handler for SafeTensors model files."""

    FORMAT_NAME = "SafeTensors"
    SUPPORTED_EXTENSIONS = (".safetensors",)

    def detect_format(self, path: Path) -> str:
        try:
            from safetensors import safe_open
            with safe_open(str(path), framework="numpy") as f:
                metadata = f.metadata() or {}
                format_info = metadata.get("format", "safetensors")
                return f"SafeTensors (format: {format_info})"
        except ImportError:
            return "SafeTensors (safetensors library not available)"
        except Exception as e:
            return f"SafeTensors (error: {e})"

    def load_metadata(self, path: Path) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        try:
            from safetensors import safe_open
            with safe_open(str(path), framework="numpy") as f:
                sf_metadata = f.metadata() or {}
                metadata.update(sf_metadata)
                metadata["keys"] = f.keys()
                metadata["num_tensors"] = len(f.keys())
        except ImportError:
            logger.warning("safetensors package not installed; limited metadata")
            # Read the JSON header from the file directly
            try:
                with open(path, "rb") as f:
                    header_size = int.from_bytes(f.read(8), "little")
                    header_bytes = f.read(header_size)
                    header = json.loads(header_bytes.decode("utf-8"))
                    metadata["keys"] = list(header.keys())
                    metadata["num_tensors"] = len(header)
                    if "__metadata__" in header:
                        metadata.update(header["__metadata__"])
            except Exception as e:
                logger.warning("Error reading SafeTensors header: %s", e)
        except Exception as e:
            logger.warning("Error reading SafeTensors metadata: %s", e)
        return metadata

    def quantize(self, path: Path) -> Path:
        """Quantize a SafeTensors model."""
        output_path = self._get_output_path(path)
        self._log_info(f"Quantizing {path} -> {output_path}")

        try:
            import numpy as np
            from safetensors import safe_open

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Read original file
            with safe_open(str(path), framework="numpy") as f:
                metadata = f.metadata() or {}
                tensors = {}
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    tensors[key] = tensor

            # Quantize tensors
            quantized_tensors = {}
            for key, tensor in tensors.items():
                if tensor.dtype in (np.float32, np.float64):
                    if self.profile.output_dtype.value == "int8":
                        quantized_tensors[key] = self._quantize_to_int8(tensor)
                    elif self.profile.output_dtype.value == "int4":
                        quantized_tensors[key] = self._quantize_to_int8(tensor)  # INT8 as proxy
                        self._log_info(f"INT4 not natively supported; using INT8 for {key}")
                    elif self.profile.output_dtype.value == "fp16":
                        quantized_tensors[key] = tensor.astype(np.float16)
                    elif self.profile.output_dtype.value == "bf16":
                        quantized_tensors[key] = tensor.astype(np.float16)  # BF16 -> FP16 fallback
                    else:
                        quantized_tensors[key] = tensor
                else:
                    quantized_tensors[key] = tensor

            # Write quantized file
            from safetensors.numpy import save_file
            save_file(quantized_tensors, str(output_path), metadata=metadata)
            self._log_info(f"Quantized SafeTensors written to {output_path}")

        except ImportError:
            logger.warning("safetensors not available; copying file")
            import shutil
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output_path)
        except Exception as e:
            logger.error("SafeTensors quantization error: %s", e)
            import shutil
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output_path)

        return output_path

    @staticmethod
    def _quantize_to_int8(tensor: Any) -> Any:
        """Quantize a numpy array to INT8 with per-channel symmetric quantization."""
        import numpy as np

        # Compute scale
        abs_max = np.abs(tensor).max(axis=tuple(range(1, tensor.ndim)), keepdims=True)
        abs_max = np.maximum(abs_max, np.full_like(abs_max, 1e-8))
        scale = abs_max / 127.0
        quantized = np.round(tensor / scale).astype(np.int8)
        quantized = np.clip(quantized, -128, 127)
        return quantized

    def embed_template(self, quantized_path: Path) -> None:
        """SafeTensors doesn't support arbitrary string metadata. Use sidecar JSON."""
        template = self.profile.template
        sidecar_data: Dict[str, Any] = {"format": "safetensors"}

        if template.chat_template:
            sidecar_data["chat_template"] = template.chat_template
        if template.system_prompt:
            sidecar_data["system_prompt"] = template.system_prompt
        if template.personality:
            sidecar_data["personality"] = template.personality
        if template.role:
            sidecar_data["role"] = template.role
        if template.skill_md_content:
            sidecar_data["skill_md"] = template.skill_md_content

        if sidecar_data:
            sidecar_path = quantized_path.parent / f"{quantized_path.stem}.template.json"
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(sidecar_data, f, indent=2)
            self._log_info(f"Wrote template sidecar to {sidecar_path}")

    def write_behavior_metadata(self, quantized_path: Path) -> None:
        """Write behavior metadata as a JSON sidecar."""
        behavior = self.profile.behavior
        sidecar_data: Dict[str, Any] = {
            "behavior": {
                k: v.emphasis for k, v in behavior.categories.items()
            }
        }

        sidecar_path = quantized_path.parent / f"{quantized_path.stem}.behavior.json"
        existing: Dict[str, Any] = {}
        if sidecar_path.exists():
            with open(sidecar_path, "r", encoding="utf-8") as f:
                existing = json.load(f)

        existing.update(sidecar_data)
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        self._log_info(f"Wrote behavior metadata to {sidecar_path}")
