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
1. *Discovering General Reinforcement Learning Algorithms with Adversarial Environment Design, NeurIPS 2023* [ [ArXiv](https://arxiv.org/abs/2310.02782) | [NeurIPS](https://neurips.cc/virtual/2023/poster/70658) | [Twitter](https://twitter.com/JacksonMattT/status/1709955868467626058) ]
   * Learned Policy Gradient (**LPG**),
   * Prioritized Level Replay (**PLR**),
   * General RL Algorithms Obtained Via Environment Design (**GROOVE**),
   * Grid-World environment from the LPG paper.
2. *Discovering Temporally-Aware Reinforcement Learning Algorithms, ALOE 2023* [ [OpenReview](https://openreview.net/forum?id=MJJcs3zbmi) ]
   * Temporally-Aware LPG (**TA-LPG**),
   * Evolutionary Strategies (**ES**) with antithetic task sampling.

All scripts are JIT-compiled end-to-end and makes extensive use of JAX-based parallelization, enabling meta-training in *under 3 hours* on a single GPU!

[**Setup**](#setup) | [**Running experiments**](#running-experiments) | [**Citation**](#cite)

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
Meta-training is executed with `python train.py`, with all arguments found in [`experiments/parse_args.py`](https://github.com/EmptyJackson/groove/blob/main/experiments/parse_args.py).

### Docker
To execute CPU or GPU docker containers, run the relevant script (with the GPU index as the first argument for the GPU script).
```
./run_gpu.sh [GPU id] python train.py [args]
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

# Coming soon

* Meta-testing script for checkpointed models.
* Alternative UED metrics (PVL, MaxMC).
