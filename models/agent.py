from typing import Sequence, Any
from flax import linen as nn

from models.common import MLP, CNN


class Actor(nn.Module):
    layers: Sequence[Any]
    n_actions: int

    @nn.compact
    def __call__(self, x):
        if self.layers:
            x = MLP((*self.layers, self.n_actions))(x)
        else:
            x = nn.Dense(self.n_actions, use_bias=False)(x)
        return nn.softmax(x)


class ConvActor(nn.Module):
    layers: Sequence[Any]
    n_actions: int
    convert_nchw: bool

    @nn.compact
    def __call__(self, x):
        x = CNN(self.layers, self.convert_nchw)(x)
        x = nn.Dense(self.n_actions)(x)
        return nn.softmax(x)


class Critic(nn.Module):
    layers: Sequence[Any]
    # LPG critic if target width > 1, value critic otherwise
    critic_dims: int

    @nn.compact
    def __call__(self, x):
        if self.layers:
            x = MLP((*self.layers, self.critic_dims))(x)
        else:
            x = nn.Dense(self.critic_dims, use_bias=False)(x)
        if self.critic_dims > 1:
            return nn.softmax(x)
        return x


class ConvCritic(nn.Module):
    layers: Sequence[Any]
    critic_dims: int
    convert_nchw: bool

    @nn.compact
    def __call__(self, x):
        x = CNN(self.layers, self.convert_nchw)(x)
        x = nn.Dense(self.critic_dims)(x)
        if self.critic_dims > 1:
            return nn.softmax(x)
        return x
