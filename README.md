<h1 align="center">Meta-Learned RL Objective Functions in JAX</h1>

<p align="center">
    <a href= "https://arxiv.org/abs/2310.02782">
        <img src="https://img.shields.io/badge/arXiv-2310.02782-b31b1b.svg" /></a>
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
Meta-training is executed with `python3 train.py`, with all arguments found in [`experiments/parse_args.py`](https://github.com/EmptyJackson/groove/blob/main/experiments/parse_args.py).
* `--log --wandb_entity [entity] --wandb_project [project]` enables logging to WandB.
* `--num_agents [agents]` sets the meta-training batch size.
* `--num_mini_batches [mini_batches]` computes each update in sequential mini-batches, in order to execute large batches with little memory. RECOMMENDED: lower this to the smallest value that fits in memory.
* `--debug` disables JIT compilation.

### Docker
To execute CPU or GPU docker containers, run the relevant script (with the GPU index as the first argument for the GPU script).
```
./run_gpu.sh [GPU id] python3 train.py [args]
```

### Examples
* LPG: `python3 train.py --num_agents 512 --num_mini_batches 16 --log --wandb_entity [entity] --wandb_project [project]`
* GROOVE: LPG with `--score_function alg_regret`
* TA-LPG: LPG with `--num_mini_batches 8 --use_es --lifetime_conditioning`

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

# Coming soon

* Meta-testing script for checkpointed models.
* Alternative UED metrics (PVL, MaxMC).
