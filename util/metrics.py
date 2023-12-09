import jax.numpy as jnp
from flax.training.train_state import TrainState


def batch_rollout_entropy(train_state: TrainState, x: jnp.ndarray):
    """Computes the entropy of the policy/target over a batch of rollouts."""
    probs = train_state.apply_fn({"params": train_state.params}, x)
    probs += 1e-8
    return -jnp.mean(jnp.multiply(probs, jnp.log(probs)).sum(axis=-1)), probs


def kl_divergence(p: jnp.array, q: jnp.array, eps: float = 1e-8):
    """Calculates the Kullback-Leibler divergence between arrays p and q."""
    return p.dot(jnp.log(p + eps) - jnp.log(q + eps))


def gae(
    value: jnp.array,
    reward: jnp.array,
    done: jnp.array,
    discount: float,
    gae_lambda: float,
):
    """
    Lifted from Gymnax-blines
    Value has length T+1, reward and done have length T
    Returns advantages and value targets
    """
    advantages = []
    gae = 0.0
    for t in reversed(range(len(value) - 1)):
        value_diff = discount * value[t + 1] * (1 - done[t]) - value[t]
        delta = reward[t] + value_diff
        gae = delta + discount * gae_lambda * (1 - done[t]) * gae
        advantages.append(gae)
    advantages = advantages[::-1]
    advantages = jnp.array(advantages)
    return advantages, advantages + value[:-1]
