import time
import torch
from transformers import PreTrainedTokenizer
from typing import Dict, List, Any, Optional
import numpy as np
from loguru import logger


class MetricsLogger:
    """
    A utility class to log and manage performance metrics for a single evaluation run.
    It tracks timing, token counts, and GPU memory usage.
    """

    def __init__(self, tokenizer: Optional[PreTrainedTokenizer] = None):
        self.metrics: Dict[str, Any] = {}
        self.tokenizer = tokenizer
        self._timers: Dict[str, float] = {}

    def set_tokenizer(self, tokenizer: PreTrainedTokenizer):
        """Sets the tokenizer for token counting."""
        self.tokenizer = tokenizer

    def start_timer(self, name: str):
        """Starts a timer for a specific metric."""
        self._timers[name] = time.monotonic()

    def stop_timer(self, name: str, log_name: Optional[str] = None):
        """Stops a timer and records the duration."""
        if name in self._timers:
            duration = time.monotonic() - self._timers[name]
            self.log(log_name or f"{name}_time_s", duration)
            del self._timers[name]
        else:
            logger.warning(f"Timer '{name}' was not started.")

    def log(self, name: str, value: Any):
        """Logs a metric value."""
        self.metrics[name] = value

    def log_token_count(self, name: str, text: str):
        """Logs the token count of a given text."""
        if self.tokenizer is None:
            logger.warning(
                "Tokenizer not set. Cannot log token count. Use set_tokenizer()."
            )
            return
        if not text or not isinstance(text, str):
            token_count = 0
        else:
            token_count = len(self.tokenizer.encode(text, add_special_tokens=False))
        self.log(name, token_count)

    def measure_gpu_memory(self, func, name: str, *args, **kwargs):
        """
        Executes a function and measures its GPU memory usage.

        Args:
            func: The function to execute.
            name: The base name for the memory metrics.
            *args, **kwargs: Arguments to pass to the function.

        Returns:
            The return value of the function.
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Cannot measure GPU memory.")
            return func(*args, **kwargs)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Measure memory before
        start_mem = torch.cuda.memory_allocated()

        result = func(*args, **kwargs)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Measure memory after
        end_mem = torch.cuda.memory_allocated()

        self.log(f"{name}_mem_allocated_mb", (end_mem - start_mem) / (1024**2))
        return result

    def get_metrics(self) -> Dict[str, Any]:
        """Returns the collected metrics."""
        return self.metrics


def aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregates a list of metric dictionaries into a single dictionary with mean values.
    """
    if not metrics_list:
        return {}

    aggregated: Dict[str, List[Any]] = {}
    for metrics in metrics_list:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key not in aggregated:
                    aggregated[key] = []
                aggregated[key].append(value)

    summary = {
        f"avg_{key}": np.mean(values) for key, values in aggregated.items() if values
    }
    return summary


