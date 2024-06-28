import jax
import chex
import jax.random as random
import jax.numpy as jnp

from functools import partial

from environments.gridworld.gridworld import EnvParams
from environments.gridworld.custom_mazes import MAZE_DESIGNS


def reset_env_params(rng: chex.PRNGKey, env_mode: str):
    """Sample new gridworld environment parameters for a given mode."""
    params = {}
    mps = ENV_MODE_PARAMS[env_mode]
    kwargs = ENV_MODE_KWARGS[env_mode]

    # --- Sample object parameters ---
    params["obj_ids"] = jnp.array(
        mps["obj_ids"] + [-1] * (kwargs["max_n_objs"] - len(mps["obj_ids"]))
    )
    for obj_param in ["obj_rewards", "obj_p_terminate", "obj_p_respawn"]:
        rng, _rng = random.split(rng)
        params[obj_param] = _sample_obj_param(
            _rng, mps[obj_param], kwargs["max_n_obj_types"]
        )

    # --- Sample remaining parameters ---
    params["auto_collect"] = mps["auto_collect"]
    params["random_respawn"] = not mps["tabular"]
    for other_param in ["max_steps_in_episode", "n_objs", "grid_size"]:
        rng, _rng = random.split(rng)
        params[other_param] = _sample_param(_rng, mps[other_param])

    # --- Sample wall positions ---
    rng, _rng = random.split(rng)
    wall_idxs = _sample_param(_rng, mps["wall_idxs"])
    params["walls"] = (
        jnp.zeros(kwargs["max_grid_size"] ** 2, dtype=jnp.bool_).at[wall_idxs].set(True)
    )

    # --- Sample agent and object positions ---
    all_pos = jnp.arange(kwargs["max_grid_size"] ** 2)
    valid_pos = jnp.logical_and(
        all_pos < params["grid_size"] ** 2,
        jnp.logical_not(jnp.isin(all_pos, wall_idxs)),
    )
    rng, _rng = random.split(rng)
    sampled_pos = random.choice(
        _rng, all_pos, shape=(kwargs["max_n_objs"] + 1,), replace=False, p=valid_pos
    )
    params["start_pos"], params["static_obj_poss"] = sampled_pos[0], sampled_pos[1:]
    return EnvParams(**params)


def reset_lifetime(rng: chex.PRNGKey, env_mode: str):
    return ENV_MODE_LIFETIME[env_mode](rng)


def get_env_spec(mode: str):
    """Returns static environment specification and maximum episode length."""
    return {k: v for k, v in ENV_MODE_KWARGS[mode].items()}, ENV_MODE_EPISODE_LEN[mode]


def get_max_lifetime(mode: str):
    """Returns maximum lifetime length."""
    return ENV_MODE_LIFETIME_MAX[mode]


def get_agent_hypers(mode: str):
    """Returns agent hyperparameters for a given mode."""
    return MODE_AGENT_HYPERS[mode]


def _sample_obj_param(rng: chex.PRNGKey, param, max_n_obj_types: int):
    """Sample and pad object parameters up to max number of object types."""
    if callable(param):
        val = param(rng)
        return jnp.concatenate((val, jnp.zeros(max_n_obj_types - len(val))))
    return jnp.array(param + [0.0] * (max_n_obj_types - len(param)))


def _sample_param(rng: chex.PRNGKey, param):
    """Sample parameter."""
    if callable(param):
        rng, _rng = jax.random.split(rng)
        return param(_rng)
    return param


"""
Gridworld environment configurations
Reference LPG environment modes:
    dense, sparse, long, longer, long_dense, rand_dense, rand_long, rand_small, rand_sparse and rand_very_dense
"""


def uniform_first_pos(key: chex.PRNGKey, n: int, minval: float, maxval: float):
    """Uniformly samples floats in the range [minval, maxval], but ensures the first position is positive."""
    k1, k2 = random.split(key)
    samples = jnp.concatenate(
        (
            random.uniform(k1, shape=(1,), minval=0.0, maxval=maxval),
            random.uniform(k2, shape=(n - 1,), minval=minval, maxval=maxval),
        )
    )
    return samples


def uniform_wall_idxs(key: chex.PRNGKey, n_walls: int, max_grid_size: int):
    """Uniformly samples wall indices."""
    return random.choice(
        key, jnp.arange(max_grid_size**2), shape=(n_walls,), replace=False
    )


def log_uniform(key: chex.PRNGKey, shape: tuple, minval: float, maxval: float):
    """Uniformly samples floats in the range [minval, maxval] on a log scale."""
    return jnp.exp(
        random.uniform(key, shape=shape, minval=jnp.log(minval), maxval=jnp.log(maxval))
    )


def log_uniform_int(key: chex.PRNGKey, shape: tuple, minval: int, maxval: int):
    """Uniformly samples ints in the range [minval, maxval] on a log scale."""
    return jnp.round(log_uniform(key, shape, minval, maxval)).astype(jnp.int32)


def get_maze_params(maze_name: str):
    shared_params = {
        "manual": False,
        "max_steps_in_episode": partial(
            log_uniform_int, shape=(), minval=25, maxval=50
        ),
        "obj_ids": [0, 1, 2],
        "obj_rewards": partial(random.uniform, shape=(3,), minval=0.0, maxval=1.0),
        "obj_p_terminate": partial(log_uniform, shape=(3,), minval=1e-2, maxval=1.0),
        "obj_p_respawn": partial(log_uniform, shape=(3,), minval=1e-3, maxval=1e-1),
        "n_objs": 3,
        "grid_size": 13,
        "tabular": True,
        "auto_collect": True,
    }
    shared_params["wall_idxs"] = MAZE_DESIGNS[maze_name]
    return shared_params


ENV_MODE_PARAMS = {
    "dense": {
        "manual": False,
        "max_steps_in_episode": 500,
        "obj_ids": [0, 0, 1, 2],
        "obj_rewards": [1.0, -1.0, -1.0],
        "obj_p_terminate": [0.0, 0.5, 0.0],
        "obj_p_respawn": [0.05, 0.1, 0.5],
        "n_objs": 4,
        "grid_size": 11,
        "wall_idxs": jnp.array([], dtype=jnp.int32),
        "tabular": True,
        "auto_collect": True,
    },
    "sparse": {
        "manual": False,
        "max_steps_in_episode": 50,
        "obj_ids": [0, 1],
        "obj_rewards": [1.0, -1.0],
        "obj_p_terminate": [1.0, 1.0],
        "obj_p_respawn": [0.0, 0.0],
        "n_objs": 2,
        "grid_size": 13,
        "wall_idxs": jnp.array([], dtype=jnp.int32),
        "tabular": True,
        "auto_collect": True,
    },
    "long": {
        "manual": False,
        "max_steps_in_episode": 1000,
        "obj_ids": [0, 0, 1, 1],
        "obj_rewards": [1.0, -1.0],
        "obj_p_terminate": [0.0, 0.5],
        "obj_p_respawn": [0.01, 1.0],
        "n_objs": 4,
        "grid_size": 11,
        "wall_idxs": jnp.array([], dtype=jnp.int32),
        "tabular": True,
        "auto_collect": True,
    },
    "longer": {
        "manual": False,
        "max_steps_in_episode": 2000,
        "obj_ids": 2 * [0] + 3 * [1],  # Reference: 2, 5
        "obj_rewards": [1.0, -1.0],
        "obj_p_terminate": [0.1, 0.8],
        "obj_p_respawn": [0.01, 1.0],
        "n_objs": 5,  # Reference: 7
        "grid_size": 9,
        # Vertical wall down centre with two corridors
        "wall_idxs": jnp.argwhere(
            jnp.logical_and(
                jnp.arange(9**2) % 9 == 4,
                jnp.logical_not(
                    jnp.isin(jnp.arange(9**2), jnp.array([(9 * 1) + 4, (9 * 7) + 4]))
                ),
            )
        ),
        "tabular": True,
        "auto_collect": True,
    },
    "long_dense": {
        "manual": False,
        "max_steps_in_episode": 2000,
        "obj_ids": 4 * [0],
        "obj_rewards": [1.0],
        "obj_p_terminate": [0.0],
        "obj_p_respawn": [0.005],
        "n_objs": 4,
        "grid_size": 11,
        # Vertical and horizontal walls, each with two corridors
        "wall_idxs": jnp.argwhere(
            jnp.logical_or(
                jnp.logical_and(
                    jnp.arange(11**2) % 11 == 5,
                    jnp.logical_not(
                        jnp.isin(
                            jnp.arange(11**2), jnp.array([(11 * 0) + 5, (11 * 7) + 5])
                        )
                    ),
                ),
                jnp.logical_and(
                    jnp.arange(11**2) // 11 == 4,
                    jnp.logical_not(
                        jnp.isin(
                            jnp.arange(11**2), jnp.array([(11 * 4) + 2, (11 * 4) + 8])
                        )
                    ),
                ),
            )
        ),
        "tabular": True,
        "auto_collect": True,
    },
    "rand_dense": {
        "manual": False,
        "max_steps_in_episode": 500,
        "obj_ids": [0, 0, 1, 2],
        "obj_rewards": [1.0, -1.0, -1.0],
        "obj_p_terminate": [0.0, 0.5, 0.0],
        "obj_p_respawn": [0.05, 0.1, 0.5],
        "n_objs": 4,
        "grid_size": 11,
        "wall_idxs": jnp.array([], dtype=jnp.int32),
        "tabular": False,
        "auto_collect": True,
    },
    "rand_long": {
        "manual": False,
        "max_steps_in_episode": 1000,
        "obj_ids": [0, 0, 1, 1],
        "obj_rewards": [1.0, -1.0],
        "obj_p_terminate": [0.0, 0.5],
        "obj_p_respawn": [0.01, 1.0],
        "n_objs": 4,
        "grid_size": 11,
        "wall_idxs": jnp.array([], dtype=jnp.int32),
        "tabular": False,
        "auto_collect": True,
    },
    "rand_small": {
        "manual": False,
        "max_steps_in_episode": 500,
        "obj_ids": [0, 0, 1, 1],
        "obj_rewards": [1.0, -1.0],
        "obj_p_terminate": [0.0, 0.5],
        "obj_p_respawn": [0.05, 0.1],
        "n_objs": 4,
        "grid_size": 7,
        "wall_idxs": jnp.array([9, 25]),
        "tabular": False,
        "auto_collect": True,
    },
    "rand_sparse": {
        "manual": False,
        "max_steps_in_episode": 50,
        "obj_ids": [0, 1, 1],
        "obj_rewards": [1.0, -1.0],
        "obj_p_terminate": [1.0, 1.0],
        "obj_p_respawn": [1.0, 1.0],
        "n_objs": 3,
        "grid_size": 7,
        "wall_idxs": jnp.array([], dtype=jnp.int32),
        "tabular": False,
        "auto_collect": True,
    },
    "rand_very_dense": {
        "manual": False,
        "max_steps_in_episode": 2000,
        "obj_ids": [0],
        "obj_rewards": [1.0],
        "obj_p_terminate": [0.0],
        "obj_p_respawn": [1.0],
        "n_objs": 1,
        "grid_size": 11,
        "wall_idxs": jnp.array([], dtype=jnp.int32),
        "tabular": False,
        "auto_collect": True,
    },
    # Custom
    "rand_tiny": {
        "manual": False,
        "max_steps_in_episode": 50,
        "obj_ids": [0, 0],
        "obj_rewards": [1.0],
        "obj_p_terminate": [0.0],
        "obj_p_respawn": [1.0],
        "n_objs": 2,
        "grid_size": 3,
        "wall_idxs": jnp.array([], dtype=jnp.int32),
        "tabular": False,
        "auto_collect": True,
    },
    # Environment distributions
    "tabular": {
        "manual": True,
        "modes": ("dense", "sparse", "long", "longer", "long_dense"),
    },
    "small": {
        "manual": False,
        "max_steps_in_episode": partial(
            log_uniform_int, shape=(), minval=20, maxval=100
        ),
        "obj_ids": [0, 1, 2],
        "obj_rewards": partial(uniform_first_pos, n=3, minval=-1.0, maxval=1.0),
        "obj_p_terminate": partial(log_uniform, shape=(3,), minval=1e-2, maxval=1.0),
        "obj_p_respawn": partial(log_uniform, shape=(3,), minval=1e-3, maxval=1e-1),
        "n_objs": partial(random.choice, a=jnp.arange(1, 4)),
        "grid_size": partial(random.choice, a=jnp.arange(4, 7)),
        "wall_idxs": partial(uniform_wall_idxs, n_walls=7, max_grid_size=6),
        "tabular": True,
        "auto_collect": True,
    },
    "medium": {
        "manual": False,
        "max_steps_in_episode": partial(
            log_uniform_int, shape=(), minval=100, maxval=250
        ),
        "obj_ids": [0, 1, 2, 3],
        "obj_rewards": partial(uniform_first_pos, n=4, minval=-1.0, maxval=1.0),
        "obj_p_terminate": partial(log_uniform, shape=(4,), minval=1e-2, maxval=1.0),
        "obj_p_respawn": partial(log_uniform, shape=(4,), minval=1e-3, maxval=1e-1),
        "n_objs": partial(random.choice, a=jnp.arange(2, 5)),
        "grid_size": partial(random.choice, a=jnp.arange(6, 9)),
        "wall_idxs": partial(uniform_wall_idxs, n_walls=10, max_grid_size=8),
        "tabular": True,
        "auto_collect": True,
    },
    "large": {
        "manual": False,
        "max_steps_in_episode": partial(
            log_uniform_int, shape=(), minval=250, maxval=750
        ),
        "obj_ids": [0, 1, 2, 3, 4],
        "obj_rewards": partial(uniform_first_pos, n=5, minval=-1.0, maxval=1.0),
        "obj_p_terminate": partial(log_uniform, shape=(5,), minval=1e-2, maxval=1.0),
        "obj_p_respawn": partial(log_uniform, shape=(5,), minval=1e-3, maxval=1e-1),
        "n_objs": partial(random.choice, a=jnp.arange(2, 6)),
        "grid_size": partial(random.choice, a=jnp.arange(8, 11)),
        "wall_idxs": partial(uniform_wall_idxs, n_walls=15, max_grid_size=10),
        "tabular": True,
        "auto_collect": True,
    },
    "all": {
        "manual": False,
        "max_steps_in_episode": partial(
            log_uniform_int, shape=(), minval=20, maxval=750
        ),
        "obj_ids": [0, 1, 2, 3, 4],
        "obj_rewards": partial(uniform_first_pos, n=5, minval=-1.0, maxval=1.0),
        "obj_p_terminate": partial(log_uniform, shape=(5,), minval=1e-2, maxval=1.0),
        "obj_p_respawn": partial(log_uniform, shape=(5,), minval=1e-3, maxval=1e-1),
        "n_objs": partial(random.choice, a=jnp.arange(1, 6)),
        "grid_size": partial(random.choice, a=jnp.arange(4, 11)),
        "wall_idxs": partial(uniform_wall_idxs, n_walls=15, max_grid_size=10),
        "tabular": True,
        "auto_collect": True,
    },
    "rand_all": {
        "manual": False,
        "max_steps_in_episode": partial(
            log_uniform_int, shape=(), minval=20, maxval=750
        ),
        "obj_ids": [0, 1, 2, 3, 4],
        "obj_rewards": partial(uniform_first_pos, n=5, minval=-1.0, maxval=1.0),
        "obj_p_terminate": partial(log_uniform, shape=(5,), minval=1e-2, maxval=1.0),
        "obj_p_respawn": partial(log_uniform, shape=(5,), minval=1e-3, maxval=1e-1),
        "n_objs": partial(random.choice, a=jnp.arange(1, 6)),
        "grid_size": partial(random.choice, a=jnp.arange(4, 11)),
        "wall_idxs": partial(uniform_wall_idxs, n_walls=15, max_grid_size=10),
        "tabular": False,
        "auto_collect": True,
    },
    "debug": {
        "manual": False,
        "max_steps_in_episode": partial(log_uniform_int, shape=(), minval=5, maxval=10),
        "obj_ids": [0, 1],
        "obj_rewards": partial(uniform_first_pos, n=2, minval=-1.0, maxval=1.0),
        "obj_p_terminate": partial(log_uniform, shape=(2,), minval=1e-2, maxval=1.0),
        "obj_p_respawn": partial(log_uniform, shape=(2,), minval=1e-3, maxval=1e-1),
        "n_objs": partial(random.choice, a=jnp.arange(1, 3)),
        "grid_size": partial(random.choice, a=jnp.arange(3, 5)),
        "wall_idxs": partial(uniform_wall_idxs, n_walls=4, max_grid_size=4),
        "tabular": True,
        "auto_collect": True,
    },
    # Mazes
    **{maze: get_maze_params(maze) for maze in MAZE_DESIGNS},
    "mazes": {
        "manual": True,
        "modes": tuple(MAZE_DESIGNS),
    },
}


_MAZE_KWARGS = {
    "max_n_objs": 3,
    "max_n_obj_types": 3,
    "max_grid_size": 13,
    "tabular": True,
}

ENV_MODE_KWARGS = {
    "dense": {
        "max_n_objs": 4,
        "max_n_obj_types": 3,
        "max_grid_size": 11,
        "tabular": True,
    },
    "sparse": {
        "max_n_objs": 2,
        "max_n_obj_types": 2,
        "max_grid_size": 13,
        "tabular": True,
    },
    "long": {
        "max_n_objs": 4,
        "max_n_obj_types": 2,
        "max_grid_size": 11,
        "tabular": True,
    },
    "longer": {
        "max_n_objs": 5,
        "max_n_obj_types": 2,
        "max_grid_size": 9,
        "tabular": True,
    },
    "long_dense": {
        "max_n_objs": 4,
        "max_n_obj_types": 1,
        "max_grid_size": 11,
        "tabular": True,
    },
    "rand_dense": {
        "max_n_objs": 4,
        "max_n_obj_types": 3,
        "max_grid_size": 11,
        "tabular": False,
    },
    "rand_long": {
        "max_n_objs": 4,
        "max_n_obj_types": 2,
        "max_grid_size": 11,
        "tabular": False,
    },
    "rand_small": {
        "max_n_objs": 4,
        "max_n_obj_types": 2,
        "max_grid_size": 7,
        "tabular": False,
    },
    "rand_sparse": {
        "max_n_objs": 3,
        "max_n_obj_types": 2,
        "max_grid_size": 7,
        "tabular": False,
    },
    "rand_very_dense": {
        "max_n_objs": 1,
        "max_n_obj_types": 1,
        "max_grid_size": 11,
        "tabular": False,
    },
    # Custom
    "rand_tiny": {
        "max_n_objs": 2,
        "max_n_obj_types": 1,
        "max_grid_size": 3,
        "tabular": False,
    },
    # Environment distributions
    "tabular": {
        "max_n_objs": 5,
        "max_n_obj_types": 3,
        "max_grid_size": 13,
        "tabular": True,
    },
    "small": {
        "max_n_objs": 3,
        "max_n_obj_types": 3,
        "max_grid_size": 6,
        "tabular": True,
    },
    "medium": {
        "max_n_objs": 4,
        "max_n_obj_types": 4,
        "max_grid_size": 8,
        "tabular": True,
    },
    "large": {
        "max_n_objs": 5,
        "max_n_obj_types": 5,
        "max_grid_size": 10,
        "tabular": True,
    },
    "all": {
        "max_n_objs": 5,
        "max_n_obj_types": 5,
        "max_grid_size": 10,
        "tabular": True,
    },
    "rand_all": {
        "max_n_objs": 5,
        "max_n_obj_types": 5,
        "max_grid_size": 10,
        "tabular": False,
    },
    "debug": {
        "max_n_objs": 2,
        "max_n_obj_types": 2,
        "max_grid_size": 4,
        "tabular": True,
    },
    # Mazes
    **{maze: _MAZE_KWARGS for maze in MAZE_DESIGNS},
    "mazes": _MAZE_KWARGS,
}

ENV_MODE_EPISODE_LEN = {
    "dense": 500,
    "sparse": 50,
    "long": 1000,
    "longer": 2000,
    "long_dense": 2000,
    "rand_dense": 500,
    "rand_long": 1000,
    "rand_small": 500,
    "rand_sparse": 50,
    "rand_very_dense": 2000,
    # Custom
    "rand_tiny": 50,
    # Environment distributions
    "tabular": 2000,
    "small": 100,
    "medium": 250,
    "large": 750,
    "all": 750,
    "rand_all": 750,
    "debug": 10,
    # Mazes
    **{maze: 50 for maze in MAZE_DESIGNS},
    "mazes": 50,
}


# Duplicate param settings
ENV_MODE_PARAMS = {
    **ENV_MODE_PARAMS,
    "all_shortlife": ENV_MODE_PARAMS["all"],
    "all_randlife": ENV_MODE_PARAMS["all"],
    "all_vrandlife": ENV_MODE_PARAMS["all"],
}

ENV_MODE_KWARGS = {
    **ENV_MODE_KWARGS,
    "all_shortlife": ENV_MODE_KWARGS["all"],
    "all_randlife": ENV_MODE_KWARGS["all"],
    "all_vrandlife": ENV_MODE_KWARGS["all"],
}

ENV_MODE_EPISODE_LEN = {
    **ENV_MODE_EPISODE_LEN,
    "all_shortlife": ENV_MODE_EPISODE_LEN["all"],
    "all_randlife": ENV_MODE_EPISODE_LEN["all"],
    "all_vrandlife": ENV_MODE_EPISODE_LEN["all"],
}


# Reference: lifetime = int(3e6 / (args.env_workers * args.train_rollout_len))
# Updates per LPG update (K) * LPG updates
_TABULAR_LIFETIME = 5 * 500
_RAND_LIFETIME = 10 * 5 * 500
_SMALL_LIFETIME = 5 * 50
_MEDIUM_LIFETIME = 5 * 200
_LARGE_LIFETIME = 5 * 500
_MAZE_LIFETIME = 5 * 500
_DEBUG_LIFETIME = 4

ENV_MODE_LIFETIME = {
    "dense": lambda _: _TABULAR_LIFETIME,
    "sparse": lambda _: _TABULAR_LIFETIME,
    "long": lambda _: _TABULAR_LIFETIME,
    "longer": lambda _: _TABULAR_LIFETIME,
    "long_dense": lambda _: _TABULAR_LIFETIME,
    "rand_dense": lambda _: _RAND_LIFETIME,
    "rand_long": lambda _: _RAND_LIFETIME,
    "rand_small": lambda _: _RAND_LIFETIME,
    "rand_sparse": lambda _: _RAND_LIFETIME,
    "rand_very_dense": lambda _: _RAND_LIFETIME,
    # Custom
    "rand_tiny": lambda _: _SMALL_LIFETIME,
    # Environment distributions
    "tabular": lambda _: _TABULAR_LIFETIME,
    "small": lambda _: _SMALL_LIFETIME,
    "medium": lambda _: _MEDIUM_LIFETIME,
    "large": lambda _: _LARGE_LIFETIME,
    "all": lambda _: _MEDIUM_LIFETIME,
    "rand_all": lambda _: _RAND_LIFETIME,
    "all_shortlife": lambda _: _SMALL_LIFETIME,
    "all_randlife": partial(
        log_uniform_int, shape=(), minval=_SMALL_LIFETIME // 5, maxval=_SMALL_LIFETIME
    ),
    "all_vrandlife": partial(
        log_uniform_int, shape=(), minval=_SMALL_LIFETIME // 25, maxval=_SMALL_LIFETIME
    ),
    "debug": lambda _: _DEBUG_LIFETIME,
    # Mazes
    **{maze: (lambda _: _MAZE_LIFETIME) for maze in MAZE_DESIGNS},
    "mazes": lambda _: _MAZE_LIFETIME,
}

ENV_MODE_LIFETIME_MAX = {
    "all_randlife": _SMALL_LIFETIME,
    "all_vrandlife": _SMALL_LIFETIME,
}
# Get deterministic lifetimes
ENV_MODE_LIFETIME_MAX.update(
    {
        mode: ENV_MODE_LIFETIME[mode](None)
        for mode in ENV_MODE_LIFETIME
        if mode not in ENV_MODE_LIFETIME_MAX
    }
)

_TABULAR_HYPERS = {
    "actor_net": (),
    "actor_learning_rate": 4e1,
    "critic_net": (),
    "critic_learning_rate": 4e0,  # Reference: 4e+1
    "optimizer": "SGD",
    "max_grad_norm": 0.5,
}

_RAND_HYPERS = {
    "actor_net": (32,),
    "actor_learning_rate": 1e-3,
    "critic_net": (32,),
    "critic_learning_rate": 1e-3,
    "optimizer": "Adam",
    "max_grad_norm": 0.5,
}

# Convolution layers have form (features, kernel_width)
_TINY_HYPERS = {
    "actor_net": (32, 32, 32),
    "actor_learning_rate": 1e-3,
    "critic_net": (32, 32, 32),
    "critic_learning_rate": 1e-3,
    "optimizer": "Adam",
    "max_grad_norm": 0.5,
}

MODE_AGENT_HYPERS = {
    "dense": _TABULAR_HYPERS,
    "sparse": _TABULAR_HYPERS,
    "long": _TABULAR_HYPERS,
    "longer": _TABULAR_HYPERS,
    "long_dense": _TABULAR_HYPERS,
    "rand_dense": _RAND_HYPERS,
    "rand_long": _RAND_HYPERS,
    "rand_small": _RAND_HYPERS,
    "rand_sparse": _RAND_HYPERS,
    "rand_very_dense": _RAND_HYPERS,
    # Custom
    "rand_tiny": _TINY_HYPERS,
    # Environment distributions
    "tabular": _TABULAR_HYPERS,
    "small": _TABULAR_HYPERS,
    "medium": _TABULAR_HYPERS,
    "large": _TABULAR_HYPERS,
    "all": _TABULAR_HYPERS,
    "rand_all": _RAND_HYPERS,
    "all_shortlife": _TABULAR_HYPERS,
    "all_randlife": _TABULAR_HYPERS,
    "all_vrandlife": _TABULAR_HYPERS,
    "debug": _TABULAR_HYPERS,
    # Mazes
    **{maze: _TABULAR_HYPERS for maze in MAZE_DESIGNS},
    "mazes": _TABULAR_HYPERS,
}
