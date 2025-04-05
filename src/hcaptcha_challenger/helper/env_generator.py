#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Description: Generate .env.example file from AgentConfig

import inspect
import re
import textwrap
from pathlib import Path
from typing import Type, Optional, get_type_hints, Literal, get_origin, get_args

from pydantic import SecretStr
from pydantic_settings import BaseSettings


def generate_env_example(
    config_class: Type[BaseSettings], output_dir: Optional[Path] = None
) -> Path:
    """
    Generate a .env.example file based on a pydantic BaseSettings class.

    Args:
        config_class: A pydantic BaseSettings class (like AgentConfig)
        output_dir: Optional directory to save the .env.example file (default: current directory)

    Returns:
        Path to the generated .env.example file
    """
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / ".env.example"

    # Get model schema to extract field metadata
    model_schema = config_class.model_json_schema()
    properties = model_schema.get("properties", {})

    # Get field order from class definition
    source_code = inspect.getsource(config_class)

    # Find all field definitions in the class
    field_pattern = r'(\w+)(?:\s*:|\s*=\s*Field\()'
    field_matches = re.findall(field_pattern, source_code)

    # Create an ordered list of fields, preserving the original order
    field_order = []
    for field_name in field_matches:
        if (
            field_name not in field_order
            and not field_name.startswith('_')
            and field_name in properties
        ):
            field_order.append(field_name)

    # Add any fields that were missed by the regex
    for field_name in properties:
        if field_name not in field_order and not field_name.startswith('_'):
            field_order.append(field_name)

    env_lines = []
    type_hints = get_type_hints(config_class)

    # Process fields in the original order
    for field_name in field_order:
        # Skip non-environment variables (typically lowercase or Path objects)
        # In environment variables, we typically want uppercase variables only
        if not field_name.isupper():
            continue

        # Get field properties from schema
        field_props = properties.get(field_name, {})

        # Skip complex types that shouldn't be in .env
        field_type = type_hints.get(field_name)
        if field_type and (
            field_type == Path or getattr(field_type, "__origin__", None) in (list, dict)
        ):
            continue

        # Extract description and default value
        description = field_props.get("description", "")

        # Format the entry
        if description:
            # Add description as comment with proper wrapping
            for description_line in description.split("\n"):
                if not description_line.strip():  # Skip empty lines in description
                    continue

                # Wrap long lines (80 characters max, accounting for "# " prefix)
                wrapped_lines = textwrap.wrap(
                    description_line.strip(),
                    width=77,  # 80 - 3 ("# " prefix and space)
                    break_long_words=False,
                    break_on_hyphens=True,
                )

                for wrapped_line in wrapped_lines:
                    env_lines.append(f"# {wrapped_line}")

        # Check if field type is Literal and add choices to description
        if field_type and get_origin(field_type) is Literal:
            choices = get_args(field_type)
            env_lines.append("# Available choices:")
            for choice in choices:
                # Remove quotes for string values in output
                choice_str = str(choice)
                env_lines.append(f"# - {choice_str}")

        # Get default value
        default_value = field_props.get("default")

        # Special handling for SecretStr
        if field_name in type_hints and type_hints[field_name] == SecretStr:
            # For SecretStr, we'll provide an empty string as example
            default_value = ""

        # Add the environment variable
        if default_value is not None:
            # Convert boolean values to lowercase strings
            if isinstance(default_value, bool):
                default_value = str(default_value).lower()
            env_lines.append(f"{field_name}={default_value}")
        else:
            env_lines.append(f"{field_name}=")

        # Add a blank line between entries
        env_lines.append("")

    # Remove trailing empty line if exists
    if env_lines and env_lines[-1] == "":
        env_lines.pop()

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(env_lines))

    print(f"Generated .env.example file at {output_file}")
    return output_file
