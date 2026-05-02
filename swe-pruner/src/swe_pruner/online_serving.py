import os
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
import logging
import uvicorn
import typer
from .prune_wrapper import SwePrunerForCodePruning, PruneRequest, PruneResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Code Pruning Service")

# Global model and tokenizer
model: Optional[SwePrunerForCodePruning] = None

# Create Typer app
cli = typer.Typer(help="SwePruner code pruning service")


def check_model_path(model_path: str) -> bool:
    """Check if model directory exists and contains required files."""
    model_dir = Path(model_path)
    if not model_dir.exists():
        return False

    # Check for essential model files
    required_files = ["config.json", "model.safetensors"]
    for file in required_files:
        if not (model_dir / file).exists():
            return False
    return True


@app.on_event("startup")
async def startup_event():
    try:
        global model
        model_name_or_path = os.getenv("SWEPRUNER_MODEL_PATH", "./model")

        if not check_model_path(model_name_or_path):
            error_msg = (
                f"Model not found at {model_name_or_path}. "
                "Please download the model or set SWEPRUNER_MODEL_PATH environment variable. "
                "See README.md for instructions."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        model = SwePrunerForCodePruning.from_pretrained(model_name_or_path)
        logger.info(f"Model loaded successfully from {model_name_or_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/prune", response_model=PruneResponse)
async def prune_code(request: PruneRequest) -> PruneResponse | None:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    response = model.prune(request)
    return response


@cli.command()
def serve(
    host: str = typer.Option(
        "0.0.0.0", "--host", "-h", help="Host to bind the server to"
    ),
    port: int = typer.Option(8000, "--port", "-p", help="Port to run the server on"),
    model_path: Optional[str] = typer.Option(
        None,
        "--model-path",
        "-m",
        help="Path to model directory. Overrides SWEPRUNER_MODEL_PATH environment variable.",
    ),
):
    """Start the FastAPI server for code pruning."""
    if model_path:
        os.environ["SWEPRUNER_MODEL_PATH"] = model_path

    # Verify model exists before starting server
    final_model_path = os.getenv("SWEPRUNER_MODEL_PATH", "./model")
    if not check_model_path(final_model_path):
        typer.echo(
            f"Error: Model not found at {final_model_path}",
            err=True,
        )
        typer.echo(
            "Please download the model or set SWEPRUNER_MODEL_PATH environment variable.",
            err=True,
        )
        typer.echo("See README.md for instructions.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Starting server on {host}:{port}")
    typer.echo(f"Model path: {final_model_path}")
    uvicorn.run(app, host=host, port=port)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
