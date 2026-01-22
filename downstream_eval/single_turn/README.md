Step 1:

Download lcc to the single_turn directory. And `DATASET_PATH="${10:-/path/to/longcodeqa_xk.jsonl}"` implies that you can download from `https://huggingface.co/datasets/Steefano/LCB/tree/main -> LongCodeQA.zip`

Step 2:

```bash 
cd /path/to/single_turn
```
Step 3:

Run the script you need.

```bash
sbatch scripts/lcc_4x_qwen.sh
```
Remember to replace the path that specifies /path/to, and it's best to use absolute paths.
Notice that we use slurm, and you can also commit the task using bash. It require 4 gpus to run that, if using Slurm, you need to replace `--partition=h100x` with your cluster name.