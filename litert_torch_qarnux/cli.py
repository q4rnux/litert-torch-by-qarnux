"""
Command-Line Interface for litert-torch.

Provides the `litert-torch` CLI with subcommands for converting
GGUF models to LiteRT-LM format. The primary subcommand is `export_hf`
which drives the multi-agent orchestration pipeline.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from litert_torch_qarnux.configs.quantization import QuantizationConfig

logger = logging.getLogger("litert_torch_qarnux")


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the litert-torch CLI."""
    parser = argparse.ArgumentParser(
        prog="litert-torch",
        description=(
            "Convert GGUF models to LiteRT-LM (.litertlm) format "
            "using multi-agent orchestration."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  litert-torch export_hf --model=model.gguf --output_dir=./output
  litert-torch export_hf --model=llama.gguf --output_dir=./litert_output --quantize
  litert-torch export_hf --model=phi3.gguf --output_dir=./out --quantization_recipe=dynamic_wi4_afp32
  litert-torch export_hf --model=gemma3.gguf --output_dir=./out --backend=cpu
  litert-torch list_architectures
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # export_hf subcommand
    export_parser = subparsers.add_parser(
        "export_hf",
        help="Convert a local GGUF file to LiteRT-LM (.litertlm) format",
        description=(
            "Converts a GGUF model file to the LiteRT-LM container format. "
            "The conversion pipeline uses multi-agent orchestration to parse, "
            "dequantize, author, convert, and package the model."
        ),
    )
    export_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the local GGUF model file (e.g., model.gguf)",
    )
    export_parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the .litertlm file and intermediate artifacts",
    )
    export_parser.add_argument(
        "--quantize",
        action="store_true",
        default=True,
        help="Apply quantization during TFLite conversion (default: enabled)",
    )
    export_parser.add_argument(
        "--no-quantize",
        action="store_true",
        default=False,
        help="Disable quantization during conversion",
    )
    export_parser.add_argument(
        "--quantization_recipe",
        type=str,
        default="dynamic_wi8_afp32",
        help=(
            "Quantization recipe to use. Options: none, dynamic_wi8_afp32, "
            "dynamic_wi4_afp32, full_int8, float8, or path to custom JSON recipe. "
            "(default: dynamic_wi8_afp32)"
        ),
    )
    export_parser.add_argument(
        "--backend",
        type=str,
        default="CPU",
        choices=["CPU", "GPU", "NPU"],
        help=(
            "Target backend for the model. Controls the TargetBackend "
            "metadata in the .litertlm container. (default: CPU)"
        ),
    )
    export_parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose/debug logging output",
    )

    # list_architectures subcommand
    list_parser = subparsers.add_parser(
        "list_architectures",
        help="List all supported model architectures",
    )

    return parser


def cmd_export_hf(args: argparse.Namespace) -> int:
    """
    Execute the export_hf subcommand.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    from litert_torch_qarnux.orchestrator import OrchestratorAgent

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error("Model file not found: %s", model_path)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine quantization
    quantize = not args.no_quantize
    recipe = args.quantization_recipe

    # Determine backend
    target_backend = args.backend.upper()

    # Check if recipe is a file path
    recipe_path = Path(recipe)
    if recipe_path.exists() and recipe_path.suffix == ".json":
        quant_config = QuantizationConfig.from_file(str(recipe_path))
        recipe_name = recipe
    else:
        try:
            quant_config = QuantizationConfig.from_recipe(recipe)
            recipe_name = recipe
        except ValueError:
            logger.error("Unknown quantization recipe: %s", recipe)
            logger.info("Available recipes: %s", QuantizationConfig.list_recipes())
            return 1

    logger.info("Configuration:")
    logger.info("  Model: %s", model_path)
    logger.info("  Output: %s", output_dir)
    logger.info("  Quantize: %s", quantize)
    logger.info("  Recipe: %s", recipe_name)
    logger.info("  Backend: %s", target_backend)

    # Run the pipeline
    orchestrator = OrchestratorAgent(
        model_path=model_path,
        output_dir=output_dir,
        quantize=quantize,
        quantization_recipe=recipe_name,
        target_backend=target_backend,
    )

    try:
        results = orchestrator.run()
        logger.info("Conversion successful!")
        logger.info("  Output file: %s", results.get("litertlm_path"))
        logger.info("  File size: %.2f MB", results.get("file_size_bytes", 0) / (1024 * 1024))
        logger.info("  Time: %.1f seconds", results.get("elapsed_seconds", 0))
        return 0

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return 1
    except RuntimeError as e:
        logger.error("Conversion failed: %s", e)
        return 1
    except ImportError as e:
        logger.error(
            "Missing dependency: %s\n"
            "Install required packages with: pip install litert-lm-builder onnx2tf",
            e,
        )
        return 1
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        if args.verbose:
            logger.exception("Full traceback:")
        return 1


def cmd_list_architectures(args: argparse.Namespace) -> int:
    """
    List all supported model architectures.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code.
    """
    from litert_torch_qarnux.models.base import ModelRegistry
    from litert_torch_qarnux.configs.model_configs import list_supported_architectures

    print("Supported architectures:")
    print("-" * 40)
    for arch in list_supported_architectures():
        model_class = ModelRegistry.get_model_class(arch)
        class_name = model_class.__name__ if model_class else "N/A"
        print(f"  {arch:15s} -> {class_name}")
    print()
    print("GGUF architecture identifiers:")
    print("  llama, gemma, gemma2, gemma3, gemma3n, gemma4,")
    print("  mistral, qwen2, qwen3, phi, phi2, phi3, smollm3")
    return 0


def main(argv: Optional[list] = None) -> None:
    """
    Main entry point for the litert-torch CLI.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Dispatch to the appropriate command handler
    if args.command == "export_hf":
        exit_code = cmd_export_hf(args)
    elif args.command == "list_architectures":
        exit_code = cmd_list_architectures(args)
    else:
        parser.print_help()
        exit_code = 1

    sys.exit(exit_code)
