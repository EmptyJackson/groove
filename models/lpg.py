import jax.numpy as jnp

from flax import linen as nn
from flax.linen.initializers import zeros
from functools import partial
from typing import Tuple

from models.common import MLP


class LPGGRU(nn.Module):
    """Reverse GRU layer for LPG model"""

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=1,
        out_axes=1,
        split_rngs={"params": False},
        reverse=True,
    )
    @nn.compact
    def __call__(
        self, gru_state: jnp.array, x: jnp.array, done: jnp.array
    ) -> Tuple[jnp.array, jnp.array]:
        """Applies the module."""
        # Reset GRU at terminal states
        gru_state = jnp.where(done, jnp.zeros_like(gru_state), gru_state)
        new_gru_state, y = nn.GRUCell()(gru_state, x)
        return new_gru_state, y

    @staticmethod
    def initialize_carry(rng, batch_dims, size, init_fn=zeros):
        """Initialize the GRU carry."""
        mem_shape = batch_dims + (size,)
        return init_fn(rng, mem_shape)


class LPG(nn.Module):
    """LPG model"""

    embedding_net_width: int
    gru_width: int
    target_width: int
    lifetime_conditioning: bool

    @nn.compact
    def __call__(self, r, d, pi, yt, yt1, step, lifetime):
        """
        Args:
            r: Reward (Batch, Seq)
            d: Done (Batch, Seq)
            pi: Selected action probability (Batch, Seq)
            yt: Predicted value vector at time t (Batch, Seq, TargetWidth)
            yt1: Predicted value vector at time t+1 (Batch, Seq, TargetWidth)
            step: Current step (float)
            lifetime: Lifetime of the agent (float)

        Returns:
            pi_hat: Policy adjustment at time t (Batch, Seq)
            y_hat: Target value vector at time t (Batch, Seq, StateEmbeddingWidth)
        """
        r = jnp.expand_dims(r, axis=-1)
        d = jnp.expand_dims(d, axis=-1)
        pi = jnp.expand_dims(pi, axis=-1)
        embedding_net = MLP([self.embedding_net_width, 1])
        pyt = embedding_net(yt)
        pyt1 = embedding_net(yt1)
        pyt1 = jnp.where(d, jnp.zeros_like(pyt1), pyt1)
        if self.lifetime_conditioning:
            relative_step = jnp.full_like(r, step, dtype=jnp.float32)
            log_lifetime = jnp.full_like(r, lifetime, dtype=jnp.float32)
            x = jnp.concatenate(
                (r, d, pi, pyt, pyt1, relative_step, log_lifetime), axis=-1
            )
        else:
            x = jnp.concatenate((r, d, pi, pyt, pyt1), axis=-1)

        carry = jnp.zeros((*d.shape[:-2], self.gru_width))
        _, x = LPGGRU()(carry, x, d)

        x = nn.relu(x)
        pi_hat = nn.Dense(1)(x)
        y_hat = nn.softmax(nn.Dense(self.target_width)(x))
        return pi_hat, y_hat

    def get_init_vector(self):
        return (
            jnp.ones([1, 1]),
            jnp.ones([1, 1]),
            jnp.ones([1, 1]),
            jnp.ones([1, 1, self.target_width]),
            jnp.ones([1, 1, self.target_width]),
            1.0,
            1.0,
        )
