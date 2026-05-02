"""Configuration files and utilities for mini-SWE-agent."""

import os
import re
from pathlib import Path
from typing import Any

import yaml

builtin_config_dir = Path(__file__).parent


def get_config_path(config_spec: str | Path) -> Path:
    """Get the path to a config file."""
    config_spec = Path(config_spec)
    if config_spec.suffix != ".yaml":
        config_spec = config_spec.with_suffix(".yaml")
    candidates = [
        Path(config_spec),
        Path(os.getenv("MSWEA_CONFIG_DIR", ".")) / config_spec,
        builtin_config_dir / config_spec,
        builtin_config_dir / "extra" / config_spec,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find config file for {config_spec} (tried: {candidates})")


def _substitute_env_vars(value: Any) -> Any:
    """Recursively substitute environment variables in config values.
    
    Supports ${VAR} and $VAR syntax. If the variable is not found, returns the original string.
    Also handles cases where quotes are included in the string like $"${VAR}".
    """
    if isinstance(value, str):
        # First, handle ${VAR} syntax (most common)
        def replace_env(match):
            var_name = match.group(1)
            env_value = os.getenv(var_name)
            if env_value is not None:
                return env_value
            return match.group(0)
        value = re.sub(r'\$\{([^}]+)\}', replace_env, value)
        
        # Handle edge case: $"${VAR}" or $"{VAR}" (quotes included)
        # This handles cases where YAML might have parsed it incorrectly
        value = re.sub(r'\$"\{([^}]+)\}"', lambda m: os.getenv(m.group(1), m.group(0)), value)
        
        # Replace $VAR syntax (but not $$VAR which is escaped)
        def replace_simple_env(match):
            var_name = match.group(1)
            if var_name:
                env_value = os.getenv(var_name)
                if env_value is not None:
                    return env_value
            return match.group(0)
        value = re.sub(r'\$([A-Za-z_][A-Za-z0-9_]*)', replace_simple_env, value)
        return value
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    else:
        return value


def load_config(config_path: Path) -> dict:
    """Load a YAML config file and substitute environment variables.
    
    Environment variables can be referenced using ${VAR} or $VAR syntax.
    Variables are loaded from .env file (via dotenv) and system environment.
    """
    config = yaml.safe_load(config_path.read_text())
    return _substitute_env_vars(config)


__all__ = ["builtin_config_dir", "get_config_path", "load_config"]
