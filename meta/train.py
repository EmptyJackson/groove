import jax
import chex
import jax.numpy as jnp

from typing import Any
from functools import partial
from flax.training.train_state import TrainState

from util import *
from agents.agents import eval_agent, compute_advantage
from agents.lpg_agent import train_lpg_agent


def lpg_meta_grad_train_step(
    rng: chex.PRNGKey,
    lpg_train_state: TrainState,
    agent_states: AgentState,
    value_critic_states: TrainState,
    rollout_manager: Any,
    num_mini_batches: int,
    gamma: float,
    gae_lambda: float,
    lpg_hypers: LpgHyperparams,
):
    """
    Update a batch of agents with LPG, then update LPG with regularized final agent loss.
    """
    num_agents = agent_states.env_obs.shape[0]
    agent_train_fn = partial(
        train_lpg_agent,
        rollout_manager=rollout_manager,
        num_train_steps=lpg_hypers.num_agent_updates,
        agent_target_coeff=lpg_hypers.agent_target_coeff,
    )

    def _train_agent(lpg_params, rng, agent_state, value_critic_state):
        """Perform K agent train steps then evaluate an agent."""
        _lpg_train_state = lpg_train_state.replace(params=lpg_params)

        # --- Perform K agent train steps ---
        rng, _rng = jax.random.split(rng)
        agent_state, rollouts, agent_metrics = agent_train_fn(
            _rng, _lpg_train_state, agent_state
        )

        # --- Rollout updated agent ---
        rng, _rng = jax.random.split(rng)
        eval_rollouts, env_obs, env_state, _ = rollout_manager.batch_rollout(
            _rng,
            agent_state.actor_state,
            agent_state.level.env_params,
            agent_state.env_obs,
            agent_state.env_state,
        )
        agent_state = agent_state.replace(
            env_obs=env_obs,
            env_state=env_state,
        )

        # --- Update value function ---
        def _compute_value_loss(critic_params, rollouts):
            value_critic_state.replace(params=critic_params)
            value_loss, adv = jax.vmap(
                compute_advantage, in_axes=(None, 0, None, None)
            )(value_critic_state, rollouts, gamma, gae_lambda)
            return value_loss.mean(), adv

        def _update_critic(value_critic_state, rollouts):
            losses, value_critic_grad = jax.value_and_grad(
                _compute_value_loss, has_aux=True
            )(value_critic_state.params, rollouts)
            return value_critic_state.apply_gradients(grads=value_critic_grad), losses

        # Iteratively update on train rollouts
        value_critic_state, _ = jax.lax.scan(
            _update_critic, value_critic_state, rollouts
        )
        # Update critic on evaluation rollout
        value_critic_state, (value_loss, adv) = _update_critic(
            value_critic_state, eval_rollouts
        )

        # --- Compute regularized LPG loss ---
        # Normalize advantage across batch
        adv = jnp.divide(jnp.subtract(adv, jnp.mean(adv)), jnp.std(adv) + 1e-8)

        def _compute_lpg_loss(rollout, adv):
            actor = agent_state.actor_state
            action_probs = actor.apply_fn({"params": actor.params}, rollout.obs)
            sampled_log_probs = gather(jnp.log(action_probs + 1e-8), rollout.action)
            return -jnp.multiply(sampled_log_probs, adv)

        lpg_loss = jax.vmap(_compute_lpg_loss)(eval_rollouts, adv).mean()
        reg_lpg_loss = (
            lpg_loss
            - lpg_hypers.policy_entropy_coeff * agent_metrics.policy_entropy
            + lpg_hypers.policy_l2_coeff * agent_metrics.policy_l2
            - lpg_hypers.target_entropy_coeff * agent_metrics.critic_entropy
            + lpg_hypers.target_l2_coeff * agent_metrics.critic_l2
        )
        metrics = {
            "lpg_loss": lpg_loss,
            "reg_lpg_loss": reg_lpg_loss,
            "value_loss": value_loss,
            "lpg_agent": agent_metrics.as_dict(),
        }

        # --- Evaluate agent return ---
        rng, _rng = jax.random.split(rng)
        agent_returns = eval_agent(
            _rng,
            rollout_manager,
            agent_state.level.env_params,
            agent_state.actor_state,
            4,
        )
        metrics["lpg_agent_return"] = jnp.mean(agent_returns)
        return reg_lpg_loss, (agent_state, value_critic_state, metrics)

    # --- Compute LPG gradient for each agent ---
    rng = jax.random.split(rng, num_agents)
    _grad_fn = partial(jax.grad(_train_agent, has_aux=True), lpg_train_state.params)
    lpg_grad, (agent_states, value_critic_states, metrics) = mini_batch_vmap(
        _grad_fn, num_mini_batches
    )(rng, agent_states, value_critic_states)

    # --- Accumulate gradients and update LPG ---
    lpg_grad, metrics = jax.tree_map(lambda x: x.mean(axis=0), (lpg_grad, metrics))
    lpg_train_state = lpg_train_state.apply_gradients(grads=lpg_grad)
    return lpg_train_state, agent_states, value_critic_states, metrics


def lpg_es_train_step(
    rng: chex.PRNGKey,
    lpg_train_state: ESTrainState,
    agent_states: AgentState,
    value_critic_states: Any,  # To match the dynamic meta-gradient function template (unused)
    rollout_manager: Any,
    num_mini_batches: int,
    lpg_hypers: LpgHyperparams,
):
    """
    Train a batch of agents with LPG, then update LPG with ES.
    Uses antithetic task sampling, meaning each antithetic pair of ES candidates is evaluated on the same level.
    """
    # --- Generate ES candidates ---
    rng, _rng = jax.random.split(rng)
    candidate_params, es_state = lpg_train_state.strategy.ask(
        _rng, lpg_train_state.es_state, lpg_train_state.es_params
    )
    # Reorder parameters to make antithetic samples adjacent
    idxs = jnp.concatenate(
        [
            jnp.array([i, i + lpg_train_state.strategy.popsize // 2])
            for i in range(lpg_train_state.strategy.popsize // 2)
        ]
    )
    candidate_params = jax.tree_map(lambda x: x[idxs], candidate_params)

    agent_train_fn = partial(
        train_lpg_agent,
        rollout_manager=rollout_manager,
        num_train_steps=lpg_hypers.num_agent_updates,
        agent_target_coeff=lpg_hypers.agent_target_coeff,
    )

    def _compute_candidate_fitness(rng, candidate_params, agent_state):
        """Train and evaluate an agent with an LPG parameter candidate."""
        candidate_train_state = lpg_train_state.train_state.replace(
            params=candidate_params
        )
        rng, _rng = jax.random.split(rng)

        # --- Train an agent using LPG with candidate parameters ---
        agent_state, _, metrics = agent_train_fn(
            rng=_rng,
            lpg_train_state=candidate_train_state,
            agent_state=agent_state,
        )

        # --- Compute return of trained agent ---
        num_env_workers = agent_state.env_obs.shape[0]
        candidate_fitness = eval_agent(
            rng,
            rollout_manager,
            agent_state.level.env_params,
            agent_state.actor_state,
            num_env_workers,
        )
        return agent_state, candidate_fitness, metrics

    # --- Evaluate LPG candidates ---
    repeated_agent_states = jax.tree_map(
        lambda x: jnp.repeat(x, 2, axis=0), agent_states
    )
    rng, _rng = jax.random.split(rng)
    _rng = jax.random.split(_rng, lpg_train_state.strategy.popsize)
    repeated_agent_states, fitness, agent_metrics = mini_batch_vmap(
        _compute_candidate_fitness, num_mini_batches
    )(_rng, candidate_params, repeated_agent_states)

    # --- Compute rank transformation per antithetic pair ---
    first_greater = jnp.greater(fitness[::2], fitness[1::2])
    rank_fitness = jnp.zeros_like(fitness)
    rank_fitness = rank_fitness.at[::2].set(first_greater.astype(float))
    rank_fitness = rank_fitness.at[1::2].set(1.0 - first_greater.astype(float))
    # Return agent from each antithetic pair with higher fitness
    agent_states = jax.tree_map(
        lambda x: jax.vmap(jnp.where)(first_greater, x[::2], x[1::2]),
        repeated_agent_states,
    )

    # --- Update and return ES state ---
    new_es_state = lpg_train_state.strategy.tell(
        candidate_params, rank_fitness, es_state, lpg_train_state.es_params
    )
    lpg_train_state = lpg_train_state.replace(es_state=new_es_state)
    metrics = {
        "fitness": {
            "mean": jnp.mean(fitness),
            "min": jnp.min(fitness),
            "max": jnp.max(fitness),
            "var": jnp.var(fitness),
        },
        "lpg_agent": jax.tree_map(jnp.mean, agent_metrics.as_dict()),
    }
    return lpg_train_state, agent_states, None, metrics
