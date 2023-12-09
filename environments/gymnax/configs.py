"""
Environment configurations for built-in Gymnax environments.
"""

configured_envs = [
    # Classic Control
    "CartPole-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    # MinAtar
    "Asterix-MinAtar",
    "Breakout-MinAtar",
    "Freeway-MinAtar",
    "SpaceInvaders-MinAtar",
    # Behaviour Suite
    "Catch-bsuite",
    "DeepSea-bsuite",
    "DiscountingChain-bsuite",
]

_CLASSIC_CONTROL_HYPERS = {
    "actor_net": (64, 64),
    "actor_learning_rate": 5e-4,
    "critic_net": (64, 64),
    "critic_learning_rate": 5e-4,
    "optimizer": "Adam",
    "max_grad_norm": 1.0,
}

# Convolution layers have form (features, kernel_width, stride)
_MIN_ATAR_HYPERS = {
    "actor_net": ((32, (4, 4), (1, 1)), (32, (4, 4), (1, 1)), 64),
    "actor_learning_rate": 5e-4,
    "critic_net": ((32, (4, 4), (1, 1)), (32, (4, 4), (1, 1)), 64),
    "critic_learning_rate": 5e-4,
    "optimizer": "Adam",
    "max_grad_norm": 1.0,
}

_GYM_BLINE_MIN_ATAR_HYPERS = {
    "actor_net": (256, 256),
    "actor_learning_rate": 5e-4,
    "critic_net": (256, 256),
    "critic_learning_rate": 5e-4,
    "optimizer": "Adam",
    "max_grad_norm": 1.0,
}

_GYM_BLINE_BSUITE_HYPERS = {
    "actor_net": (64, 64),
    "actor_learning_rate": 5e-4,
    "critic_net": (64, 64),
    "critic_learning_rate": 5e-4,
    "optimizer": "Adam",
    "max_grad_norm": 1.0,
}

AGENT_HYPERS = {
    # Classic Control
    "CartPole-v1": _CLASSIC_CONTROL_HYPERS,
    "Acrobot-v1": _CLASSIC_CONTROL_HYPERS,
    "MountainCar-v0": _CLASSIC_CONTROL_HYPERS,
    # MinAtar
    "Asterix-MinAtar": _GYM_BLINE_MIN_ATAR_HYPERS,
    "Breakout-MinAtar": _GYM_BLINE_MIN_ATAR_HYPERS,
    "Freeway-MinAtar": _GYM_BLINE_MIN_ATAR_HYPERS,
    "SpaceInvaders-MinAtar": _GYM_BLINE_MIN_ATAR_HYPERS,
    # Behaviour Suite
    "Catch-bsuite": _GYM_BLINE_BSUITE_HYPERS,
    "DeepSea-bsuite": _GYM_BLINE_BSUITE_HYPERS,
    "DiscountingChain-bsuite": _GYM_BLINE_BSUITE_HYPERS,
}

_CLASSIC_CONTROL_LIFETIME = 1000
_MIN_ATAR_LIFETIME = 100000
_BSUITE_LIFETIME = 100

ENV_MODE_LIFETIME = {
    # Classic Control
    "CartPole-v1": _CLASSIC_CONTROL_LIFETIME,
    "Acrobot-v1": _CLASSIC_CONTROL_LIFETIME,
    "MountainCar-v0": _CLASSIC_CONTROL_LIFETIME,
    # MinAtar
    "Asterix-MinAtar": _MIN_ATAR_LIFETIME,
    "Breakout-MinAtar": _MIN_ATAR_LIFETIME,
    "Freeway-MinAtar": _MIN_ATAR_LIFETIME,
    "SpaceInvaders-MinAtar": _MIN_ATAR_LIFETIME,
    # Behaviour Suite
    "Catch-bsuite": _BSUITE_LIFETIME,
    "DeepSea-bsuite": _BSUITE_LIFETIME,
    "DiscountingChain-bsuite": _BSUITE_LIFETIME,
}


def get_agent_hypers(env_name: str):
    """Get agent hyperparameters for a given environment."""
    return AGENT_HYPERS[env_name]


def reset_lifetime(env_name: str):
    return ENV_MODE_LIFETIME[env_name]


def get_max_lifetime(env_name: str):
    return ENV_MODE_LIFETIME[env_name]
