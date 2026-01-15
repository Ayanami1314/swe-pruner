"""
This file provides:

- Path settings for global config file & relative directories
- Version numbering
- Protocols for the core components of mini-swe-agent.
  By the magic of protocols & duck typing, you can pretty much ignore them,
  unless you want the static type checking.
"""

__version__ = "1.16.0"

import os
from pathlib import Path
from typing import Any, Protocol

import dotenv
from platformdirs import user_config_dir
from rich.console import Console
from minisweagent.utils.log import logger

package_dir = Path(__file__).resolve().parent

global_config_dir = Path(os.getenv("MSWEA_GLOBAL_CONFIG_DIR") or "/data/harry/SWE-Agent-C/mini-swe-agent--with-pruning")
global_config_dir.mkdir(parents=True, exist_ok=True)
global_config_file = Path(global_config_dir) / ".env"

if not os.getenv("MSWEA_SILENT_STARTUP"):
    Console().print(
        f"👋 This is [bold green]mini-swe-agent[/bold green] version [bold green]{__version__}[/bold green].\n"
        f"Loading global config from [bold green]'{global_config_file}'[/bold green]"
    )
dotenv.load_dotenv(dotenv_path=global_config_file)

# Set HuggingFace cache directories if specified in environment or .env
cache_dir = os.getenv("HF_HOME") or os.getenv("HF_DATASETS_CACHE")
if cache_dir:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    # Set environment variables for HuggingFace libraries
    if not os.getenv("HF_HOME"):
        os.environ["HF_HOME"] = str(cache_path)
    if not os.getenv("HF_DATASETS_CACHE"):
        os.environ["HF_DATASETS_CACHE"] = str(cache_path / "datasets")
    if not os.getenv("TRANSFORMERS_CACHE"):
        os.environ["TRANSFORMERS_CACHE"] = str(cache_path / "transformers")
    if not os.getenv("HF_HUB_CACHE"):
        os.environ["HF_HUB_CACHE"] = str(cache_path / "hub")
else:
    # Set default cache directory if not specified
    default_cache_dir = global_config_dir / "cache" / "huggingface"
    default_cache_dir.mkdir(parents=True, exist_ok=True)
    if not os.getenv("HF_HOME"):
        os.environ["HF_HOME"] = str(default_cache_dir)
    if not os.getenv("HF_DATASETS_CACHE"):
        os.environ["HF_DATASETS_CACHE"] = str(default_cache_dir / "datasets")
    if not os.getenv("TRANSFORMERS_CACHE"):
        os.environ["TRANSFORMERS_CACHE"] = str(default_cache_dir / "transformers")
    if not os.getenv("HF_HUB_CACHE"):
        os.environ["HF_HUB_CACHE"] = str(default_cache_dir / "hub")

if global_config_file.exists():
    with open(global_config_file, 'r') as f:
        print(f"Loading Config File: {f.name}")
        print(f"{f.read()}")
# === Protocols ===
# You can ignore them unless you want static type checking.


class Model(Protocol):
    """Protocol for language models."""

    config: Any
    cost: float
    n_calls: int

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict: ...

    def get_template_vars(self) -> dict[str, Any]: ...


class Environment(Protocol):
    """Protocol for execution environments."""

    config: Any

    def execute(self, command: str, cwd: str = "") -> dict[str, str]: ...

    def get_template_vars(self) -> dict[str, Any]: ...


class Agent(Protocol):
    """Protocol for agents."""

    model: Model
    env: Environment
    messages: list[dict[str, str]]
    config: Any

    def run(self, task: str, **kwargs) -> tuple[str, str]: ...


__all__ = [
    "Agent",
    "Model",
    "Environment",
    "package_dir",
    "__version__",
    "global_config_file",
    "global_config_dir",
    "logger",
]
