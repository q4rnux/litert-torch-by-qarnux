"""
Template Resources.

Built-in chat templates and skill.md examples for common use cases.
"""
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent

# Built-in chat templates
CHAT_TEMPLATES = {
    "default": "{% for message in messages %}{% if message.role == 'user' %}{{ '<|user|>\n' + message.content + eos_token }}{% elif message.role == 'assistant' %}{{ '<|assistant|>\n' + message.content + eos_token }}{% elif message.role == 'system' %}{{ '<|system|>\n' + message.content + eos_token }}{% endif %}{% endfor %}",

    "chatml": "{% for message in messages %}{{'<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",

    "llama2": "{% for message in messages %}{% if message.role == 'user' %}{{ bos_token + '[INST] ' + message.content.strip() + ' [/INST]' }}{% elif message.role == 'assistant' %}{{ ' ' + message.content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}",

    "llama3": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message.role + '<|end_header_id|>\n\n'+ message.content.strip() + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",

    "mistral": "{% for message in messages %}{% if message.role == 'user' %}{% if not loop.first %}{{ eos_token }}{% endif %}{{ '[INST] ' + message.content.strip() + ' [/INST]' }}{% elif message.role == 'assistant' %}{{ message.content.strip() }}{% endif %}{% endfor %}",

    "qwen": "{% for message in messages %}{% if message.role == 'system' %}{{ '<|im_start|>system\n' + message.content + '<|im_end|>\n' }}{% elif message.role == 'user' %}{{ '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}{% elif message.role == 'assistant' %}{{ '<|im_start|>assistant\n' + message.content + '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
}

# Built-in skill.md templates
SKILL_MD_TEMPLATES = {
    "general_assistant": """# General Assistant

## Role
You are a helpful, honest, and harmless AI assistant.

## Guidelines
- Provide accurate, well-reasoned answers
- Admit uncertainty when appropriate
- Ask clarifying questions when needed
- Use clear, concise language
- Format responses with markdown when appropriate

## Capabilities
- Answer questions across many domains
- Help with writing, analysis, and problem-solving
- Provide code examples and explanations
""",

    "code_assistant": """# Code Assistant

## Role
You are a senior software engineer AI assistant specializing in code generation, debugging, and code review.

## Guidelines
- Always provide complete, runnable code examples
- Include comments explaining complex logic
- Suggest best practices and design patterns
- Flag potential bugs or security issues
- Use type hints and follow PEP 8 style

## Languages
- Python, JavaScript/TypeScript, Rust, Go, Java, C/C++
""",

    "research_assistant": """# Research Assistant

## Role
You are an academic research assistant AI.

## Guidelines
- Provide well-sourced information
- Clearly distinguish between established facts and theories
- Cite sources when available
- Present balanced perspectives
- Acknowledge limitations and open questions

## Areas
- Scientific literature analysis
- Data interpretation
- Hypothesis generation
- Literature review synthesis
""",
}
