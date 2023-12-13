import jax
import chex
import jax.numpy as jnp

from typing import Any
from functools import partial
from flax.training.train_state import TrainState

from util import *


@struct.dataclass
class LPGAgentMetrics:
    policy_l2: float
    policy_entropy: float
    critic_loss: float
    critic_l2: float
    critic_entropy: float

    def as_dict(self):
        return {
            "policy_l2": self.policy_l2,
            "policy_entropy": self.policy_entropy,
            "critic_loss": self.critic_loss,
            "critic_l2": self.critic_l2,
            "critic_entropy": self.critic_entropy,
        }


def lpg_agent_train_step(
    actor_state: TrainState,
    critic_state: TrainState,
    rollout: Transition,
    lifetime: int,
    lpg_train_state: TrainState,
    agent_target_coeff: float,
):
    """Performs a training step for a single agent with LPG, over a batch of rollouts."""

    def selected_action_probs(all_action_probs, rollout_action):
        all_action_probs += 1e-8
        return gather(all_action_probs, rollout_action)

    def loss_fn(actor_params, critic_params):
        all_action_probs = actor_state.apply_fn({"params": actor_params}, rollout.obs)
        pi = jax.vmap(selected_action_probs)(all_action_probs, rollout.action)
        y_t = critic_state.apply_fn({"params": critic_params}, rollout.obs)
        y_tp1 = critic_state.apply_fn({"params": critic_params}, rollout.next_obs)
        preds = lpg_train_state.apply_fn(
            {"params": lpg_train_state.params},
            rollout.reward,
            rollout.done,
            jax.lax.stop_gradient(pi),
            jax.lax.stop_gradient(y_t),
            jax.lax.stop_gradient(y_tp1),
            actor_state.step,
            lifetime,
        )
        pi_hat, y_hat = preds
        y_l2 = jnp.mean(jnp.square(y_hat).sum(axis=-1))
        critic_loss = jax.vmap(jax.vmap(kl_divergence))(y_t, y_hat)
        pi_hat = jnp.squeeze(pi_hat, axis=-1)
        actor_loss = jnp.multiply(jnp.log(pi), pi_hat)
        # L2-regularization of LPG targets, used for optimization of LPG
        pi_l2 = jnp.mean(jnp.square(pi_hat))
        loss = jnp.mean(actor_loss) + agent_target_coeff * jnp.mean(critic_loss)
        metrics = (critic_loss, pi_l2, y_l2)
        return loss, metrics

    (actor_grads, critic_grads), metrics = jax.grad(
        loss_fn, argnums=(0, 1), has_aux=True
    )(actor_state.params, critic_state.params)
    updated_actor_state = actor_state.apply_gradients(grads=actor_grads)
    updated_critic_state = critic_state.apply_gradients(grads=critic_grads)

    # --- Discard update if agent has finished training ---
    actor_state, critic_state = jax.tree_map(
        lambda new, old: jnp.where(updated_actor_state.step <= lifetime, new, old),
        (updated_actor_state, updated_critic_state),
        (actor_state, critic_state),
    )
    metrics = jax.tree_map(jnp.mean, metrics)
    critic_loss, pi_l2, y_l2 = metrics
    return actor_state, critic_state, critic_loss, pi_l2, y_l2


def train_lpg_agent(
    rng: chex.PRNGKey,
    lpg_train_state: TrainState,
    agent_state: AgentState,
    rollout_manager: Any,
    num_train_steps: int,
    agent_target_coeff: float,
):
    """Train an agent with LPG for K steps; return trained agent."""

    agent_train_step_fn = partial(
        lpg_agent_train_step,
        lpg_train_state=lpg_train_state,
        agent_target_coeff=agent_target_coeff,
        lifetime=agent_state.level.lifetime,
    )

    def _train_step(carry, _):
        """Perform a single LPG train iteration."""
        rng, agent_state = carry
        rng, _rng = jax.random.split(rng)
        rollout, env_obs, env_state, _ = rollout_manager.batch_rollout(
            _rng,
            agent_state.actor_state,
            agent_state.level.env_params,
            agent_state.env_obs,
            agent_state.env_state,
        )
        actor_state, critic_state, critic_loss, pi_l2, y_l2 = agent_train_step_fn(
            agent_state.actor_state, agent_state.critic_state, rollout
        )
        actor_entropy, _ = batch_rollout_entropy(actor_state, rollout.obs)
        critic_entropy, _ = batch_rollout_entropy(critic_state, rollout.obs)
        agent_state = agent_state.replace(
            actor_state=actor_state,
            critic_state=critic_state,
            env_obs=env_obs,
            env_state=env_state,
        )
        metrics = LPGAgentMetrics(
            pi_l2, actor_entropy, critic_loss, y_l2, critic_entropy
        )
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
