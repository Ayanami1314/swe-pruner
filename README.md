# SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents

<p align="center"><b>Semantic Highlight&nbsp; ◦ &nbsp;Coding Agent Native&nbsp; ◦ &nbsp;Flexibly Use&nbsp; ◦ &nbsp;Long Context Tailored</b></p>

<p align="center"><b>Make Claude Tokens 40% Saving!</b></p>

![Tokens Cost Reduction](./images/token_cost.png)
<summary><h2>📢 Latest Updates</h2></summary>

**🔥 Releases:**
- 1/26/2025: Introduce **SWE-Pruner** 
  - paper: https://arxiv.org/abs/2601.16746
  - code: https://github.com/Ayanami1314/swe-pruner
  - pip package: https://pypi.org/project/swe-pruner/
  - huggingface repo: https://huggingface.co/ayanami-kitasan/code-pruner.




## Overview

![overview](./images/overview.jpg)
LLM agents have demonstrated remarkable capabilities in software development, but their performance is hampered by long interaction contexts, which incur high API costs and latency. While various context compression approaches have emerged to tackle this challenge, they typically rely on fixed metrics such as perplexity (PPL), ignoring the task-specific nature of code understanding. As a result, they frequently disrupt syntactic and logical structures and fail to retain critical implementation details. In this paper, we propose SWE-Pruner, a self-adaptive context pruning framework tailored for coding agents. Drawing inspiration from how human programmers "selectively skim" source code during development and debugging, SWE-Pruner performs task-aware adaptive pruning for long contexts. Given the current task, the agent formulates an explicit goal (e.g., "focus on error handling") as a hint to guide the pruning targets. A lightweight neural skimmer (0.6B parameters) is trained to dynamically select relevant lines from the surrounding context given the goal. Evaluations across four benchmarks and multiple models validate SWE-Pruner's effectiveness in various scenarios, achieving 23-54% token reduction on agent tasks like SWE-Bench Verified and up to 14.84x compression on single-turn tasks like LongCodeQA with minimal performance impact.

## Project Structure
```text
.
├── data/                      # Experiment trace archives and hyperparameter configurations
├── downstream_eval/           # Downstream evaluation benchmarks
│   ├── multi_turn/            # Includes: SWE-bench, SWEQA (coming soon)
│   └── single_turn/           # Includes: LongCodeQA, LCC (LongCodeCompletion)
├── swe-pruner/                # Inference code and model utilities
│   └── model/                 # Model files for SWE-Pruner
├── examples                   # Examples for integrating with other agents like claude code and openhands
```

## Quick Start

## Prerequisites

This project uses `uv` for fast and efficient dependency management.

### Installation

Since different modules have different dependencies, please refer to the specific `README` file inside each subfolder for detailed installation instructions.


## Coming Soon
- [ ] Update Training Code
- [ ] Upload full parameters and trajectory files
- [ ] Update HuggingFace model card
- [x] Update agents integrate demo


## Citation
```
@misc{wang2026sweprunerselfadaptivecontextpruning,
      title={SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents}, 
      author={Yuhang Wang and Yuling Shi and Mo Yang and Rongrui Zhang and Shilin He and Heng Lian and Yuting Chen and Siyu Ye and Kai Cai and Xiaodong Gu},
      year={2026},
      eprint={2601.16746},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2601.16746},
}
```


## Acknowledgements
- Bytedance Douyin Team for advises.
- Alibaba Qwen Team for open-source models.