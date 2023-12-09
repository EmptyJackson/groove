"""
Based on Gymnax experimental/rollout.py
"""
import jax
import jax.numpy as jnp

from typing import Optional
from environments.environments import get_env

from util import *


class RolloutWrapper:
    def __init__(
        self,
        env_name: str = "Pendulum-v1",
        train_rollout_len: Optional[int] = None,
        eval_rollout_len: Optional[int] = None,
        env_kwargs: dict = {},
        return_info: bool = False,
    ):
        """
        env_name (str): Name of environment to use.
        train_rollout_len (int): Number of steps to rollout during training.
        eval_rollout_len (int): Number of steps to rollout during evaluation.
        env_kwargs (dict): Static keyword arguments to pass to environment, same for all agents.
        return_info (bool): Return rollout information.
        """
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        # Define the RL environment & network forward function
        self.env = get_env(env_name, env_kwargs)
        self.train_rollout_len = train_rollout_len
        self.eval_rollout_len = eval_rollout_len
        self.return_info = return_info

    # --- ENVIRONMENT RESET ---
    def batch_reset(self, rng, env_params, num_workers):
        """Reset a single environment for multiple workers, returning initial states and observations."""
        rng = jax.random.split(rng, num_workers)
        batch_reset_fn = jax.vmap(self.env.reset, in_axes=(0, None))
        return batch_reset_fn(rng, env_params)

    # --- ENVIRONMENT ROLLOUT ---
    def batch_rollout(
        self, rng, train_state, env_params, init_obs, init_state, eval=False
    ):
        """Evaluate an agent on a single environment over a batch of workers."""
        rng = jax.random.split(rng, init_obs.shape[0])
        return jax.vmap(self.single_rollout, in_axes=(0, None, None, 0, 0, None))(
            rng, train_state, env_params, init_obs, init_state, eval
        )

    def single_rollout(
        self, rng, train_state, env_params, init_obs, init_state, eval=False
    ):
        """Rollout an episode."""

        def policy_step(state_input, _):
            rng, obs, state, train_state, cum_reward, valid_mask = state_input
            rng, _rng = jax.random.split(rng)
            action_probs = train_state.apply_fn({"params": train_state.params}, obs)
            action = jax.random.choice(_rng, action_probs.shape[-1], p=action_probs)
            rng, _rng = jax.random.split(rng)
            next_obs, next_state, reward, done, info = self.env.step(
                _rng, state, action, env_params
            )
            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry = [
                rng,
                next_obs,
                next_state,
                train_state,
                new_cum_reward,
                new_valid_mask,
            ]
            transition = Transition(obs, action, reward, next_obs, done)
            if self.return_info:
                return carry, (transition, info)
            return carry, transition

        # Scan over episode step loop
        carry_out, rollout = jax.lax.scan(
            policy_step,
            [
                rng,
                init_obs,
                init_state,
                train_state,
                jnp.float32(0.0),
                jnp.float32(1.0),
            ],
            (),
            self.eval_rollout_len if eval else self.train_rollout_len,
        )
        if self.return_info:
            rollout, info = rollout
        end_obs, end_state, cum_return = carry_out[1], carry_out[2], carry_out[4]
        if self.return_info:
            return rollout, end_obs, end_state, cum_return, info
        return rollout, end_obs, end_state, cum_return

    def optimal_return(self, env_params, max_rollout_len, return_all):
        """Return the optimal expected return for the given set of environment parameters."""
        return jax.vmap(self.env.optimal_return, in_axes=(0, None, None))(
            env_params, max_rollout_len, return_all
        )

    @property
    def input_shape(self):
        """Get the shape of the observation."""
        rng = jax.random.PRNGKey(0)
        obs, state = self.env.reset(rng, self.env_params)
        return obs.shape
