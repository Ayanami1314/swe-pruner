## Installation

### From Source

```bash
cd swe-pruner
pip install -e .
```

### Flash Attention Setup

SWE-Pruner requires `flash-attn` which needs to be installed separately based on your system configuration:

1. **Pre-built wheel (recommended)**: Download the appropriate wheel from [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases):

```bash
pip install flash_attn-<version>-<platform>.whl
```

2. **From source**:

```bash
pip install flash-attn --no-build-isolation
```

## Model

Train your own model from the guide in `swe-pruner/README.md`.
Then place it in `./model`

## Usage

### Command Line Interface

Start the FastAPI server:

```bash
swe-pruner --model-path ./model --port 8000
```

Options:
- `--host` / `-h`: Host to bind the server to (default: `0.0.0.0`)
- `--port` / `-p`: Port to run the server on (default: `8000`)
- `--model-path` / `-m`: Path to model directory (overrides `SWEPRUNER_MODEL_PATH` environment variable)

### Python API

```python
from swe_pruner.prune_wrapper import SwePrunerForCodePruning, PruneRequest

model = SwePrunerForCodePruning.from_pretrained("./model")

request = PruneRequest(
    query="Find functions that handle user authentication",
    code="""
def login(username, password):
    if verify_credentials(username, password):
        return create_session(username)
    return None

def logout(session_id):
    invalidate_session(session_id)
    """,
    threshold=0.5,
    always_keep_first_frags=False,
    chunk_overlap_tokens=50
)

response = model.prune(request)

print(f"Relevance score: {response.score}")
print(f"Pruned code:\n{response.pruned_code}")
print(f"Token count: {response.origin_token_cnt} -> {response.left_token_cnt}")
```

### API Response

The `PruneResponse` object contains:

- `score`: Document-level relevance score (float)
- `pruned_code`: Pruned code string with filtered sections marked
- `token_scores`: List of [token, score] pairs
- `kept_frags`: List of kept line numbers
- `origin_token_cnt`: Original token count
- `left_token_cnt`: Remaining token count after pruning
- `model_input_token_cnt`: Total tokens sent to the model
- `error_msg`: Error message if any (optional)

### FastAPI Server

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/prune \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find authentication functions",
    "code": "def login(): ...",
    "threshold": 0.5
  }'
```

## Configuration

### Model Parameters

- `bottleneck`: Bottleneck dimension (default: 256)
- `dropout`: Dropout rate (default: 0.4)
- `num_fusion_layers`: Number of fusion layers (default: 1)
- `num_heads`: Number of attention heads (default: 8)
- `use_multi_layer_fusion`: Whether to use multi-layer fusion (default: True)
- `compression_head_type`: Type of compression head ("ffn", "simple", or "crf")

### Pruning Parameters

- `threshold`: Score threshold for keeping tokens (default: 0.5)
- `always_keep_first_frags`: Always keep the first N fragments (default: False)
- `chunk_overlap_tokens`: Overlap tokens between chunks for long code (default: 50)

## Requirements

- Python >= 3.12
- PyTorch >= 2.8.0
- Transformers >= 4.57.1
- CUDA (for GPU acceleration)
- Flash Attention 2

See `pyproject.toml` for the complete list of dependencies.
