import jax
import chex
import gymnax

import environments.gridworld.gridworld as grid
import environments.gridworld.configs as grid_conf
import environments.gymnax.configs as gym_conf


def get_env(env_name: str, env_kwargs: dict):
    if env_name in gymnax.registered_envs:
        env, _ = gymnax.make(env_name, **env_kwargs)
    elif env_name in grid.registered_envs:
        env = grid.GridWorld(**env_kwargs)
    else:
        raise ValueError(
            f"Environment {env_name} not registered in any environment sources."
        )
    return env


def reset_env_params(rng: chex.PRNGKey, env_name: str, env_mode: str):
    """Reset environment parameters and agent lifetime."""
    if env_name in gymnax.registered_envs:
        env, _ = gymnax.make(env_name)
        params = env.default_params
        lifetime = None
        if env_name in gym_conf.configured_envs:
            # Select lifetime if mode configuration exists
            lifetime = gym_conf.reset_lifetime(env_name=env_name)
    elif env_name in grid.registered_envs:
        p_rng, l_rng = jax.random.split(rng)
        params = grid_conf.reset_env_params(p_rng, env_mode)
        lifetime = grid_conf.reset_lifetime(l_rng, env_mode)
    else:
        raise ValueError(f"Environment {env_name} has no parameter reset method.")
    return params, lifetime


def get_env_spec(env_name: str, env_mode: str):
    """Returns static environment parameters, rollout length and lifetime."""
    if env_name in [*gymnax.registered_envs]:
        kwargs = {}
        env = get_env(env_name, kwargs)
        max_rollout_len = env.default_params.max_steps_in_episode
        if env_name in gym_conf.configured_envs:
            max_lifetime = gym_conf.get_max_lifetime(env_name=env_name)
        else:
            max_lifetime = None
    elif env_name in grid.registered_envs:
        kwargs, max_rollout_len = grid_conf.get_env_spec(env_mode)
        max_lifetime = grid_conf.get_max_lifetime(env_mode)
    else:
        raise ValueError(f"Environment {env_name} has no get env spec method.")
    return kwargs, max_rollout_len, max_lifetime


def get_agent_hypers(env_name: str, env_mode: str = None):
    if env_name in gym_conf.configured_envs:
        return gym_conf.get_agent_hypers(env_name)
    elif env_name in grid.registered_envs:
        return grid_conf.get_agent_hypers(env_mode)
    raise ValueError(f"Environment {env_name} has no get agent hyperparameters method.")
