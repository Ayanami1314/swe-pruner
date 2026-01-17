import sys
import os
import argparse
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import uvicorn
from hf.prune_wrapper import SwePrunerForCodePruning, PruneRequest, PruneResponse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Code Pruning Service")

# Global model and tokenizer
model: Optional[SwePrunerForCodePruning] = None


@app.on_event("startup")
async def startup_event():
    try:
        global model
        model_name_or_path = "./swe_pruner"  # HINT: where you download the hf model
        model = SwePrunerForCodePruning.from_pretrained(model_name_or_path)

        logger.info("Model loaded successfully on startup")
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


@app.get("/prune", response_model=PruneResponse)
async def prune_code(request: PruneRequest) -> PruneResponse | None:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    response = model.prune(request)
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FastAPI service for code pruning with token-level scores"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
