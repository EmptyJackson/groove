import jax
import chex
import jax.numpy as jnp

from typing import Any
from functools import partial
from flax.training.train_state import TrainState

from util import *


@struct.dataclass
class A2CHyperparams:
    gamma: float
    gae_lambda: float
    entropy_coeff: float


def a2c_agent_train_step(
    actor_state: TrainState,
    critic_state: TrainState,
    rollout: Transition,
    lifetime: int,
    hypers: A2CHyperparams,
):
    """Performs a training step for a single agent with A2C, over a batch of rollouts."""

    # --- Update critic ---
    def _batch_critic_loss_fn(params):
        def _critic_loss_fn(rollout):
            all_obs = jnp.append(
                rollout.obs, jnp.expand_dims(rollout.next_obs[-1], 0), axis=0
            )
            value = critic_state.apply_fn({"params": params}, all_obs)
            adv, target = gae(
                value, rollout.reward, rollout.done, hypers.gamma, hypers.gae_lambda
            )
            adv, target = jax.lax.stop_gradient((adv, target))
            return jnp.mean(jnp.square(target - value[:-1])), adv

        losses, adv = jax.vmap(_critic_loss_fn)(rollout)
        # Normalize advantage
        adv = jnp.divide(jnp.subtract(adv, jnp.mean(adv)), jnp.std(adv) + 1e-8)
        return jnp.mean(losses), adv

    (critic_loss, adv), critic_grads = jax.value_and_grad(
        _batch_critic_loss_fn, has_aux=True
    )(critic_state.params)
    updated_critic_state = critic_state.apply_gradients(grads=critic_grads)

    # --- Update actor ---
    def _batch_actor_loss_fn(params):
        def _actor_loss_fn(rollout, adv):
            all_action_probs = actor_state.apply_fn({"params": params}, rollout.obs)
            all_action_probs += 1e-8
            log_probs = jnp.log(all_action_probs)
            selected_log_probs = gather(log_probs, rollout.action)
            policy_losses = -jnp.multiply(selected_log_probs, adv)
            entropy = -jnp.mean(jnp.multiply(all_action_probs, log_probs).sum(axis=-1))
            return jnp.mean(policy_losses) - hypers.entropy_coeff * entropy

        losses = jax.vmap(_actor_loss_fn)(rollout, adv)
        return jnp.mean(losses)

    actor_loss, actor_grads = jax.value_and_grad(_batch_actor_loss_fn)(
        actor_state.params
    )
    updated_actor_state = actor_state.apply_gradients(grads=actor_grads)

    # --- Discard update if agent has finished training ---
    actor_state, critic_state = jax.tree_map(
        lambda new, old: jnp.where(updated_actor_state.step <= lifetime, new, old),
        (updated_actor_state, updated_critic_state),
        (actor_state, critic_state),
    )
    return actor_state, critic_state, actor_loss, critic_loss


def train_a2c_agent(
    rng: chex.PRNGKey,
    agent_state: AgentState,
    rollout_manager: Any,
    num_train_steps: int,
    hypers: A2CHyperparams,
):
    """Train an agent with A2C for K steps; return trained agent."""

    agent_train_step_fn = partial(
        a2c_agent_train_step,
        lifetime=agent_state.level.lifetime,
        hypers=hypers,
    )

    def _train_step(carry, _):
        """Perform a single A2C train iteration."""
        rng, agent_state = carry
        rng, _rng = jax.random.split(rng)
        rollout, env_obs, env_state, _ = rollout_manager.batch_rollout(
            _rng,
            agent_state.actor_state,
            agent_state.level.env_params,
            agent_state.env_obs,
            agent_state.env_state,
        )
        actor_state, critic_state, actor_loss, critic_loss = agent_train_step_fn(
            agent_state.actor_state, agent_state.critic_state, rollout
        )
        agent_state = agent_state.replace(
            actor_state=actor_state,
            critic_state=critic_state,
            env_obs=env_obs,
            env_state=env_state,
        )
        metrics = {"actor_loss": actor_loss, "critic_loss": critic_loss}
        return (rng, agent_state), metrics

    # --- Perform K agent updates ---
    carry_out, metrics = jax.lax.scan(
        _train_step,
        (rng, agent_state),
        None,
        length=num_train_steps,
    )
    _, agent_state = carry_out
    return agent_state, jax.tree_map(jnp.mean, metrics)
