"""
ONNX Format Handler.

Handles reading, quantizing, and writing ONNX model files.
Supports dynamic and static quantization via onnxruntime or onnx tools.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from litert_torch_qarnux.model_quantization.config import QuantizationProfile
from litert_torch_qarnux.model_quantization.formatters.base import BaseFormatHandler

logger = logging.getLogger(__name__)


class ONNXFormatHandler(BaseFormatHandler):
    """Handler for ONNX model files."""

    FORMAT_NAME = "ONNX"
    SUPPORTED_EXTENSIONS = (".onnx",)

    def detect_format(self, path: Path) -> str:
        try:
            import onnx
            model = onnx.load(str(path))
            opset = ""
            for imp in model.opset_import:
                opset = f" (opset {imp.version})"
            return f"ONNX{opset}"
        except ImportError:
            return "ONNX (onnx library not available)"
        except Exception as e:
            return f"ONNX (error: {e})"

    def load_metadata(self, path: Path) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        try:
            import onnx
            model = onnx.load(str(path))
            metadata["producer_name"] = model.producer_name
            metadata["producer_version"] = model.producer_version
            metadata["domain"] = model.domain
            metadata["model_version"] = model.model_version
            metadata["graph_name"] = model.graph.name

            # Input/output info
            inputs = [inp.name for inp in model.graph.input]
            outputs = [out.name for out in model.graph.output]
            metadata["inputs"] = inputs
            metadata["outputs"] = outputs

            # Node count
            metadata["node_count"] = len(model.graph.node)

            # Opset
            for imp in model.opset_import:
                metadata[f"opset_{imp.domain or 'default'}"] = imp.version

        except ImportError:
            logger.warning("onnx package not installed; limited metadata extraction")
        except Exception as e:
            logger.warning("Error reading ONNX metadata: %s", e)
        return metadata

    def quantize(self, path: Path) -> Path:
        """Quantize an ONNX model."""
        output_path = self._get_output_path(path)
        self._log_info(f"Quantizing {path} -> {output_path}")

        try:
            import onnx
            from onnxruntime.quantization import quantize_dynamic, QuantType

            output_path.parent.mkdir(parents=True, exist_ok=True)

            if self.profile.output_dtype.value == "int8":
                quant_type = QuantType.QInt8
            elif self.profile.output_dtype.value == "uint8":
                quant_type = QuantType.QUInt8
            elif self.profile.output_dtype.value == "int4":
                quant_type = QuantType.QInt8  # ONNX quant doesn't directly support INT4
                self._log_info("ONNX doesn't support INT4 directly; using INT8 as closest")
            else:
                self._log_info(f"ONNX quantization to {self.profile.output_dtype.value} not directly supported")
                import shutil
                shutil.copy2(path, output_path)
                return output_path

            self._log_info(f"Using dynamic quantization with {quant_type}")
            quantize_dynamic(str(path), str(output_path), weight_type=quant_type)
            self._log_info(f"ONNX quantized output written to {output_path}")

        except ImportError:
            logger.warning("onnxruntime not available; copying file as-is")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(path, output_path)
            self._log_info("onnxruntime not available; output is unquantized copy")

        return output_path

    def embed_template(self, quantized_path: Path) -> None:
        """ONNX doesn't support metadata embedding in the same way as GGUF.
        Write a JSON sidecar file with template data."""
        template = self.profile.template
        sidecar_data: Dict[str, Any] = {"format": "onnx"}

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
