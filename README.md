<h1 align="center">Meta-Learned RL Objective Functions in JAX</h1>

<p align="center">
    <a href= "https://arxiv.org/abs/2310.02782">
        <img src="https://img.shields.io/badge/arXiv-2310.02782-b31b1b.svg" /></a>
    <a href= "https://arxiv.org/abs/2402.05828">
        <img src="https://img.shields.io/badge/arXiv-2402.05828-b31b1b.svg" /></a>
    <a href= "https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
    <a href= "https://github.com/EmptyJackson/groove/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
</p>

GROOVE is the official implementation of the following publications:
1. *Discovering General Reinforcement Learning Algorithms with Adversarial Environment Design, NeurIPS 2023* [[ArXiv](https://arxiv.org/abs/2310.02782) | [NeurIPS](https://neurips.cc/virtual/2023/poster/70658) | [Twitter](https://twitter.com/JacksonMattT/status/1709955868467626058)]
   * Learned Policy Gradient (**LPG**),
   * Prioritized Level Replay (**PLR**),
   * General RL Algorithms Obtained Via Environment Design (**GROOVE**),
   * Grid-World environment from the LPG paper.
2. *Discovering Temporally-Aware Reinforcement Learning Algorithms, ICLR 2024* [[ArXiv](https://arxiv.org/abs/2402.05828)]
   * Temporally-Aware LPG (**TA-LPG**),
   * Evolutionary Strategies (**ES**) with antithetic task sampling.

All scripts are JIT-compiled end-to-end and make extensive use of JAX-based parallelization, enabling meta-training in *under 3 hours* on a single GPU!

**Update (April 2023)**: Misreported LPG ES hyperparameters in repo + paper, specifically initial learning rate (`1e-4` -> `1e-2`) and sigma (`3e-3` -> `1e-1`). Now updated.

[**Setup**](#setup) | [**Running experiments**](#running-experiments) | [**Citation**](#citation)

# Setup

### Requirements

All requirements are found in `setup/`, with [`requirements-base.txt`](https://github.com/EmptyJackson/groove/blob/main/setup/requirements-base.txt) containing the majority of packages, [`requirements-cpu.txt`](https://github.com/EmptyJackson/groove/blob/main/setup/requirements-cpu.txt) containing CPU packages, and [`requirements-gpu.txt`](https://github.com/EmptyJackson/groove/blob/main/setup/requirements-gpu.txt) containing GPU packages.

Some key packages include:
* RL Environments: `gymnax`
* Neural Networks: `flax`
* Optimization: `optax`, `evosax`
* Logging: `wandb`

### Local installation (CPU)
```
pip install $(cat setup/requirements-base.txt setup/requirements-cpu.txt)
```

### Docker installation (GPU)
1. Build docker image
```
cd setup/docker & ./build_gpu.sh & cd ../..
```

2. (To enable WandB logging) Add your [account key](https://wandb.ai/authorize) to `setup/wandb_key`:
```
echo [KEY] > setup/wandb_key
```

# Running experiments
Meta-training is executed with `python3.8 train.py`, with all arguments found in [`experiments/parse_args.py`](https://github.com/EmptyJackson/groove/blob/main/experiments/parse_args.py).
| Argument | Description |
| --- | --- |
| `--env_mode [env_mode]` | Sets the environment mode (below). |
| `--num_agents [agents]` | Sets the meta-training batch size. |
| `--num_mini_batches [mini_batches]` | Computes each update in sequential mini-batches, in order to execute large batches with little memory. *RECOMMENDED: lower this to the smallest value that fits in memory.* |
| `--debug` | Disables JIT compilation. |
| `--log --wandb_entity [entity] --wandb_project [project]` | Enables logging to WandB. |


### Grid-World environments

| Environment mode | Description | Lifetime (# of updates) |
| --- | --- | --- |
|`tabular`|Five tabular levels from [LPG](https://arxiv.org/abs/2007.08794)|Variable|
|`mazes`|Maze levels from [MiniMax](https://github.com/facebookresearch/minimax)|2500|
|`all_shortlife`|Uniformly sampled levels|250|
|`all_vrandlife`|Uniformly sampled levels|10-250 (Log-sampled)|


### Examples
| Experiment | Command | Example run (WandB) |
| --- | --- | --- |
| LPG (meta-gradient) | `python3.8 train.py --num_agents 512 --num_mini_batches 16 --train_steps 5000 --log --wandb_entity [entity] --wandb_project [project]` | [Link](https://api.wandb.ai/links/mjackson/4xbnkrmd) |
| GROOVE | LPG with `--score_function alg_regret` (algorithmic regret is computed every step due to end-to-end compilation, so currently very inefficient) | TBC |
| TA-LPG | LPG with `--num_mini_batches 8 --train_steps 2500 --use_es --lifetime_conditioning --lpg_learning_rate 0.01 --env_mode all_vrandlife` | TBC |


### Docker
To execute CPU or GPU docker containers, run the relevant script (with the GPU index as the first argument for the GPU script).
```
./run_gpu.sh [GPU id] python3.8 train.py [args]
```

# Citation
If you use this implementation in your work, please cite us with the following:
```
@inproceedings{jackson2023discovering,
    author={Jackson, Matthew Thomas and Jiang, Minqi and Parker-Holder, Jack and Vuorio, Risto and Lu, Chris and Farquhar, Gregory and Whiteson, Shimon and Foerster, Jakob Nicolaus},
    booktitle = {Advances in Neural Information Processing Systems},
    title = {Discovering General Reinforcement Learning Algorithms with Adversarial Environment Design},
    volume = {36},
    year = {2023}
}
```
```
@inproceedings{jackson2024discovering,
    author={Jackson, Matthew Thomas and Lu, Chris and Kirsch, Louis and Lange, Robert Tjarko and Whiteson, Shimon and Foerster, Jakob Nicolaus},
    booktitle = {International Conference on Learning Representations},
    title = {Discovering Temporally-Aware Reinforcement Learning Algorithms},
    volume = {12},
    year = {2024}
}
```

# Coming soon

* Speed up GROOVE by removing recomputation of algorithmic regret every step.
* Meta-testing script for checkpointed models.
* Alternative UED metrics (PVL, MaxMC).
