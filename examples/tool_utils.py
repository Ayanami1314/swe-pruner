import logging
from typing import List, Optional
import subprocess
import os
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
pruner_url = os.getenv("PRUNER_URL", "http://localhost:8000/prune")


class PruneResponse(BaseModel):
    score: float
    pruned_code: str
    token_scores: List[List[str | float]]  # [[token_str, score], ...]
    kept_frags: List[int]

    class Config:
        extra = "allow"


_httpx_client: Optional[httpx.Client] = None


def _get_httpx_client() -> httpx.Client:
    """Get or create the global httpx client."""
    global _httpx_client
    if _httpx_client is None:
        # Disable proxy for prune service requests (local service should not use proxy)
        _httpx_client = httpx.Client(timeout=60.0, proxy=None)
    return _httpx_client


async def prune_fn(context: str, query: str) -> PruneResponse:
    """Async version of prune function using httpx."""
    logging.info(
        f"Calling prune_fn with query: {query[:100] if query else None}, context length: {len(context)}"
    )
    client = _get_httpx_client()
    try:
        response = await client.post(
            pruner_url,
            json={"code": context, "query": query},
            headers={"Content-Type": "application/json"},
        )
        if response.status_code == 200:
            result = PruneResponse(**response.json())
            logging.info(
                f"Prune completed: score={result.score:.3f}, kept_frags={len(result.kept_frags)}, pruned_code_length={len(result.pruned_code)}"
            )
            return result
        else:
            raise Exception(f"Pruner request failed with status {response.status_code}")
    except httpx.RequestError as e:
        logging.error(f"Pruner request error: {str(e)}")
        raise Exception(f"Pruner request error: {str(e)}")


def _execute_bash_command(command: str) -> str:
    """Execute a bash command and return output as list of lines."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=10
        )
        output = result.stdout.strip() if result.stdout.strip() else "(No output)"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 10 seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"


async def bash(
    command: str,
    query: str | None = None,
) -> str:
    """
    Execute a bash command and return the output.
    If a query is provided, prune the output based on the query.
    """
    result = _execute_bash_command(command)

    if not result or result == "(No output)":
        return result
    if query:
        pruned_result = await prune_fn(result, query)
        return pruned_result.pruned_code
    return result


async def cat(
    file_path: str,
    query: str | None = None,
) -> str:
    """
    Read file contents and return the output.
    If a query is provided, prune the output based on the query.
    """
    if not Path(file_path).is_file():
        return f"Error: File not found: {file_path}"
    with open(file_path, "r") as file:
        content = file.read()
    if query:
        pruned_result = await prune_fn(content, query)
        return pruned_result.pruned_code
    return content
