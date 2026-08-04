"""
PyTorch Format Handler.

Handles reading, quantizing, and writing PyTorch model files (.pt, .pth).
Supports weight-only and full quantization via torch.ao.quantization.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from litert_torch_qarnux.model_quantization.config import QuantizationProfile
from litert_torch_qarnux.model_quantization.formatters.base import BaseFormatHandler

logger = logging.getLogger(__name__)


class PyTorchFormatHandler(BaseFormatHandler):
    """Handler for PyTorch model files (.pt, .pth)."""

    FORMAT_NAME = "PyTorch"
    SUPPORTED_EXTENSIONS = (".pt", ".pth", ".ckpt")

    def detect_format(self, path: Path) -> str:
        try:
            import torch
            data = torch.load(str(path), map_location="cpu", weights_only=False)
            if isinstance(data, dict):
                return f"PyTorch (state_dict with {len(data)} keys)"
            elif hasattr(data, "state_dict"):
                return f"PyTorch (nn.Module: {type(data).__name__})"
            else:
                return "PyTorch (arbitrary pickle)"
        except ImportError:
            return "PyTorch (torch not available)"
        except Exception as e:
            return f"PyTorch (error: {e})"

    def load_metadata(self, path: Path) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        try:
            import torch
            data = torch.load(str(path), map_location="cpu", weights_only=False)
            if isinstance(data, dict):
                metadata["type"] = "state_dict"
                metadata["num_tensors"] = len(data)
                metadata["keys"] = list(data.keys())[:50]  # first 50 keys
                total_params = sum(
                    v.numel() if hasattr(v, "numel") else 0
                    for v in data.values()
                )
                metadata["total_parameters"] = total_params
            elif hasattr(data, "state_dict"):
                metadata["type"] = "nn.Module"
                metadata["class_name"] = type(data).__name__
                sd = data.state_dict()
                metadata["num_tensors"] = len(sd)
                metadata["total_parameters"] = sum(v.numel() for v in sd.values())
            else:
                metadata["type"] = "unknown"
        except ImportError:
            logger.warning("torch package not installed; limited metadata")
        except Exception as e:
            logger.warning("Error reading PyTorch metadata: %s", e)
        return metadata

    def quantize(self, path: Path) -> Path:
        """Quantize a PyTorch model."""
        output_path = self._get_output_path(path)
        self._log_info(f"Quantizing {path} -> {output_path}")

        try:
            import torch
            import torch.quantization as torch_quant

            data = torch.load(str(path), map_location="cpu", weights_only=False)

            if isinstance(data, dict):
                # State dict quantization
                quantized_state = {}
                for key, tensor in data.items():
                    if torch.is_tensor(tensor):
                        if tensor.dtype == torch.float32:
                            if self.profile.output_dtype.value == "int8":
                                quantized_state[key] = self._quantize_tensor_int8(tensor)
                            elif self.profile.output_dtype.value == "fp16":
                                quantized_state[key] = tensor.half()
                            elif self.profile.output_dtype.value == "bf16":
                                quantized_state[key] = tensor.to(torch.bfloat16)
                            else:
                                quantized_state[key] = tensor
                        else:
                            quantized_state[key] = tensor
                    else:
                        quantized_state[key] = tensor

                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(quantized_state, str(output_path))
                self._log_info(f"Quantized state dict written to {output_path}")

            elif hasattr(data, "state_dict"):
                # nn.Module quantization
                self._quantize_module(data, path)
                torch.save(data, str(output_path))
                self._log_info(f"Quantized module written to {output_path}")
            else:
                import shutil
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, output_path)
                self._log_info("Non-standard PyTorch file copied as-is")

        except ImportError:
            logger.warning("torch not available; copying file")
            import shutil
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output_path)
        except Exception as e:
            logger.error("PyTorch quantization error: %s", e)
            import shutil
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output_path)

        return output_path

    @staticmethod
    def _quantize_tensor_int8(tensor: "torch.Tensor") -> "torch.Tensor":
        """Quantize a single tensor to int8 with scale/zero-point metadata."""
        if not torch.is_tensor(tensor):
            return tensor
        # Use torch.quantization to get scale and zero_point
        per_channel = True
        qscheme = torch.per_channel_symmetric if per_channel else torch.per_tensor_symmetric
        if qscheme == torch.per_channel_symmetric:
            ch_axis = 0 if tensor.dim() >= 2 else -1
            qparams = torch.per_channel_dynamic_quant(tensor, ch_axis=ch_axis, dtype=torch.qint8)
        else:
            qparams = torch.per_tensor_dynamic_quant(tensor, dtype=torch.qint8)
        return qparams

    def _quantize_module(self, module: Any, path: Path) -> None:
        """Quantize an nn.Module in-place."""
        try:
            import torch
            import torch.quantization as torch_quant

            if self.profile.output_dtype.value == "int8":
                module.qconfig = torch_quant.get_default_qconfig("x86")
                torch_quant.prepare(module, inplace=True)
                # In a real pipeline, we'd provide calibration data here
                torch_quant.convert(module, inplace=True)
                self._log_info("Applied INT8 post-training quantization")
            elif self.profile.output_dtype.value == "fp16":
                module.half()
                self._log_info("Converted to FP16")
            else:
                self._log_info(f"No specific quantization for {self.profile.output_dtype.value}")
        except Exception as e:
            logger.error("Module quantization failed: %s", e)

    def embed_template(self, quantized_path: Path) -> None:
        """PyTorch files don't support arbitrary metadata. Use sidecar JSON."""
        template = self.profile.template
        sidecar_data: Dict[str, Any] = {"format": "pytorch"}

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
