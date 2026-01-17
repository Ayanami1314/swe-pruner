## Install
```
cd inference
```
Download `flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl` from github to current directory, then run
```
uv sync
```
```
uv pip install -e .
```
to install other dependencies.

## Model
model has been placed at `./swe-pruner` and tracked by git-lfs.

## Usage

Run `uv run online_serving.py` to start the fastapi server for downstream tasks.

Another script `calc_latency.py` is used to measure its latency.

After server starting succecessfully, you can run `test-prune.sh 'test' <any file path>` to test if it is working.

Server starts at port 8000 by default, and should be configured in downstream tasks. 