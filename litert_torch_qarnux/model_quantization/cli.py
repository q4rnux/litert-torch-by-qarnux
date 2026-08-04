"""
Model Quantization CLI.

Command-line interface for the model quantization module.
Supports quantizing models, managing behavior profiles, and embedding templates.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from litert_torch_qarnux.model_quantization.config import (
    BehaviorCategory,
    BehaviorProfile,
    QuantDType,
    QuantMethod,
    QuantizationProfile,
    SourceFormat,
    TemplateConfig,
)
from litert_torch_qarnux.model_quantization.formatters.registry import FormatRegistry
from litert_torch_qarnux.model_quantization.templates import CHAT_TEMPLATES, SKILL_MD_TEMPLATES

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="litert-quantize",
        description="Model Quantization Tool - Quantize models with behavior profiles and embedded templates",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- quantize ----
    quantize_parser = subparsers.add_parser(
        "quantize",
        help="Quantize a model file",
    )
    quantize_parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input model file",
    )
    quantize_parser.add_argument(
        "--output", "-o",
        default="",
        help="Path for the quantized output file",
    )
    quantize_parser.add_argument(
        "--method", "-m",
        default="post_training_static",
        choices=[m.value for m in QuantMethod],
        help="Quantization method (default: post_training_static)",
    )
    quantize_parser.add_argument(
        "--dtype", "-d",
        default="int8",
        choices=[d.value for d in QuantDType],
        help="Target quantization data type (default: int8)",
    )
    quantize_parser.add_argument(
        "--format", "-f",
        default="auto",
        choices=["auto"] + [s.value for s in SourceFormat if s != SourceFormat.AUTO],
        help="Source model format (default: auto-detect)",
    )
    quantize_parser.add_argument(
        "--output-format",
        default="gguf",
        choices=[s.value for s in SourceFormat if s != SourceFormat.AUTO],
        help="Output format (default: gguf)",
    )
    quantize_parser.add_argument(
        "--group-size",
        type=int,
        default=32,
        help="Quantization group size (default: 32)",
    )
    quantize_parser.add_argument(
        "--config", "-c",
        default="",
        help="Path to a YAML/JSON configuration file",
    )
    quantize_parser.add_argument(
        "--skill-md",
        default="",
        help="Path to a skill.md file to embed",
    )
    quantize_parser.add_argument(
        "--chat-template",
        default="",
        choices=list(CHAT_TEMPLATES.keys()),
        help="Built-in chat template to embed",
    )
    quantize_parser.add_argument(
        "--system-prompt",
        default="",
        help="System prompt to embed",
    )
    quantize_parser.add_argument(
        "--personality",
        default="",
        help="Personality description to embed",
    )
    quantize_parser.add_argument(
        "--role",
        default="",
        help="Model role to embed (e.g., assistant, code-reviewer)",
    )
    quantize_parser.add_argument(
        "--calib-data",
        default="",
        help="Path to calibration data for post-training quantization",
    )
    quantize_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    quantize_parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output results as JSON",
    )

    # ---- behavior ----
    behavior_parser = subparsers.add_parser(
        "behavior",
        help="Configure behavior categories for a quantization profile",
    )
    behavior_sub = behavior_parser.add_subparsers(dest="behavior_command")

    # list categories
    list_cats = behavior_sub.add_parser("list", help="List all behavior categories")

    # set emphasis
    set_parser = behavior_sub.add_parser("set", help="Set emphasis for a category")
    set_parser.add_argument("name", help="Category name")
    set_parser.add_argument("value", type=float, help="Emphasis value (-1.0 to 1.0)")
    set_parser.add_argument("--output", "-o", required=True, help="Output config file path")

    # profile
    profile_parser = behavior_sub.add_parser("profile", help="Create a behavior profile from preset")
    profile_parser.add_argument("--preset", required=True, help="Preset name (e.g., safe_coder, creative_writer)")
    profile_parser.add_argument("--output", "-o", required=True, help="Output config file path")

    # ---- template ----
    template_parser = subparsers.add_parser(
        "template",
        help="Manage chat templates and skill.md files",
    )
    template_sub = template_parser.add_subparsers(dest="template_command")

    # list templates
    list_templates = template_sub.add_parser("list", help="List built-in templates")

    # show template
    show_parser = template_sub.add_parser("show", help="Show a template's content")
    show_parser.add_argument("name", help="Template name")
    show_parser.add_argument("--type", "-t", choices=["chat", "skill"], default="chat")

    # ---- info ----
    info_parser = subparsers.add_parser(
        "info",
        help="Display information about a model file",
    )
    info_parser.add_argument("--input", "-i", required=True, help="Path to the model file")

    # ---- init-config ----
    init_parser = subparsers.add_parser(
        "init-config",
        help="Generate a sample configuration file",
    )
    init_parser.add_argument("--output", "-o", default="quantization_config.yaml")
    init_parser.add_argument("--format", "-f", choices=["yaml", "json"], default="yaml")

    return parser


def cmd_quantize(args: argparse.Namespace) -> int:
    """Execute the quantize command."""
    from litert_torch_qarnux.model_quantization.quantizer import ModelQuantizer

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input model not found: %s", input_path)
        return 1

    # Load from config file if provided
    if args.config:
        profile = QuantizationProfile.from_file(args.config)
        # Override with CLI args if provided
        if args.input:
            profile.input_path = args.input
        if args.output:
            profile.output_path = args.output
    else:
        profile = QuantizationProfile(
            input_path=str(input_path),
            output_path=args.output,
            method=QuantMethod(args.method),
            output_dtype=QuantDType(args.dtype),
            source_format=SourceFormat(args.format),
            output_format=SourceFormat(args.output_format),
            group_size=args.group_size,
            calib_data_path=args.calib_data if args.calib_data else None,
        )

    # Embed skill.md
    if args.skill_md:
        try:
            profile.template.load_skill_md(args.skill_md)
        except FileNotFoundError as e:
            logger.error(str(e))
            return 1

    # Embed chat template
    if args.chat_template:
        profile.template.chat_template = CHAT_TEMPLATES[args.chat_template]

    # Embed system prompt / personality / role
    if args.system_prompt:
        profile.template.system_prompt = args.system_prompt
    if args.personality:
        profile.template.personality = args.personality
    if args.role:
        profile.template.role = args.role

    # Run quantization
    quantizer = ModelQuantizer(profile)
    try:
        result = quantizer.quantize()
        if args.json_output:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"Quantization complete!")
            print(f"  Output: {result.get('output_path', 'N/A')}")
            print(f"  Format: {result.get('format', 'N/A')}")
            print(f"  Size: {result.get('file_size_bytes', 0) / (1024*1024):.2f} MB")
            print(f"  Time: {result.get('elapsed_seconds', 0):.1f} seconds")
        return 0
    except Exception as e:
        logger.error("Quantization failed: %s", e)
        if args.verbose:
            logger.exception("Full traceback:")
        return 1


def cmd_behavior(args: argparse.Namespace) -> int:
    """Execute the behavior command."""
    if args.behavior_command == "list":
        print("Available Behavior Categories:")
        print("-" * 70)
        from litert_torch_qarnux.model_quantization.config import DEFAULT_CATEGORIES
        for name, desc in DEFAULT_CATEGORIES.items():
            print(f"  {name:20s}  {desc}")
        return 0

    elif args.behavior_command == "set":
        profile = QuantizationProfile()
        if args.name:
            profile.behavior.set_emphasis(args.name, args.value)
        profile.save_yaml(args.output)
        print(f"Saved behavior profile to {args.output}")
        return 0

    elif args.behavior_command == "profile":
        preset_profiles = _get_preset_profiles()
        if args.preset not in preset_profiles:
            logger.error(f"Unknown preset: {args.preset}")
            logger.info("Available presets: %s", list(preset_profiles.keys()))
            return 1
        profile = preset_profiles[args.preset]
        profile.save_yaml(args.output)
        print(f"Saved preset profile '{args.preset}' to {args.output}")
        return 0

    else:
        print("Usage: litert-quantize behavior {list|set|profile}")
        return 1


def cmd_template(args: argparse.Namespace) -> int:
    """Execute the template command."""
    if args.template_command == "list":
        print("Built-in Chat Templates:")
        print("-" * 40)
        for name in CHAT_TEMPLATES:
            print(f"  chat:{name}")
        print()
        print("Built-in Skill.md Templates:")
        print("-" * 40)
        for name in SKILL_MD_TEMPLATES:
            print(f"  skill:{name}")
        return 0

    elif args.template_command == "show":
        if args.type == "chat":
            if args.name in CHAT_TEMPLATES:
                print(f"Chat Template: {args.name}")
                print("=" * 60)
                print(CHAT_TEMPLATES[args.name])
            else:
                logger.error("Unknown chat template: %s", args.name)
                return 1
        elif args.type == "skill":
            if args.name in SKILL_MD_TEMPLATES:
                print(f"Skill.md Template: {args.name}")
                print("=" * 60)
                print(SKILL_MD_TEMPLATES[args.name])
            else:
                logger.error("Unknown skill template: %s", args.name)
                return 1
        return 0

    else:
        print("Usage: litert-quantize template {list|show}")
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Execute the info command."""
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("File not found: %s", input_path)
        return 1

    handler = FormatRegistry.detect_and_get_handler(input_path, QuantizationProfile())
    fmt = handler.detect_format(input_path)
    metadata = handler.load_metadata(input_path)

    print(f"Model Information")
    print("=" * 60)
    print(f"Path:    {input_path}")
    print(f"Format:  {fmt}")
    print(f"Size:    {input_path.stat().st_size / (1024*1024):.2f} MB")
    print()
    print("Metadata:")
    print("-" * 60)
    for k, v in metadata.items():
        val_str = str(v)
        if len(val_str) > 100:
            val_str = val_str[:97] + "..."
        print(f"  {k}: {val_str}")
    return 0


def cmd_init_config(args: argparse.Namespace) -> int:
    """Generate a sample configuration file."""
    profile = QuantizationProfile(
        input_path="model.gguf",
        output_path="model_quantized.gguf",
        method=QuantMethod.POST_TRAINING_STATIC,
        output_dtype=QuantDType.INT8,
        source_format=SourceFormat.AUTO,
        output_format=SourceFormat.GGUF,
    )

    # Add sample behavior emphasis
    profile.behavior.set_emphasis("coding", 0.5)
    profile.behavior.set_emphasis("reasoning", 0.7)
    profile.behavior.set_emphasis("hallucination", -0.8)
    profile.behavior.set_emphasis("helpfulness", 0.6)

    # Add sample template
    profile.template.chat_template = CHAT_TEMPLATES["default"]
    profile.template.system_prompt = "You are a helpful AI assistant."

    if args.format == "yaml":
        profile.save_yaml(args.output)
        print(f"Generated YAML config: {args.output}")
    else:
        profile.save_json(args.output)
        print(f"Generated JSON config: {args.output}")
    return 0


def _get_preset_profiles() -> dict:
    """Return preset behavior profiles."""
    presets = {}

    # Safe coder
    p = QuantizationProfile()
    p.behavior.set_emphasis("coding", 0.8)
    p.behavior.set_emphasis("reasoning", 0.5)
    p.behavior.set_emphasis("hallucination", -0.9)
    p.behavior.set_emphasis("fabricating", -0.9)
    p.behavior.set_emphasis("helpfulness", 0.7)
    p.behavior.set_emphasis("conciseness", 0.4)
    presets["safe_coder"] = p

    # Creative writer
    p = QuantizationProfile()
    p.behavior.set_emphasis("brainstorming", 0.8)
    p.behavior.set_emphasis("creativity", 0.9)
    p.behavior.set_emphasis("humor", 0.5)
    p.behavior.set_emphasis("empathy", 0.6)
    p.behavior.set_emphasis("examples", 0.7)
    p.behavior.set_emphasis("over_reacting", -0.5)
    presets["creative_writer"] = p

    # Research analyst
    p = QuantizationProfile()
    p.behavior.set_emphasis("reasoning", 0.9)
    p.behavior.set_emphasis("detail", 0.8)
    p.behavior.set_emphasis("honesty", 0.8)
    p.behavior.set_emphasis("hallucination", -0.9)
    p.behavior.set_emphasis("fabricating", -0.9)
    p.behavior.set_emphasis("examples", 0.6)
    p.behavior.set_emphasis("formatting", 0.5)
    presets["research_analyst"] = p

    # Friendly assistant
    p = QuantizationProfile()
    p.behavior.set_emphasis("helpfulness", 0.9)
    p.behavior.set_emphasis("empathy", 0.7)
    p.behavior.set_emphasis("humor", 0.4)
    p.behavior.set_emphasis("safety", 0.8)
    p.behavior.set_emphasis("gaslighting", -0.9)
    p.behavior.set_emphasis("over_reacting", -0.6)
    presets["friendly_assistant"] = p

    return presets


def main(argv: Optional[list] = None) -> None:
    """Main entry point for the CLI."""
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

    # Dispatch
    command_map = {
        "quantize": cmd_quantize,
        "behavior": cmd_behavior,
        "template": cmd_template,
        "info": cmd_info,
        "init-config": cmd_init_config,
    }

    handler_fn = command_map.get(args.command)
    if handler_fn:
        exit_code = handler_fn(args)
    else:
        parser.print_help()
        exit_code = 1

    sys.exit(exit_code)
