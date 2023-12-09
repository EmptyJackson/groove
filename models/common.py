import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


class MLP(nn.Module):
    """
    ReLU activated MLP, with no activation on output.
    """

    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x


class CNN(nn.Module):
    """
    Generic CNN, with activation on output.
    Assumes a sequence of conv layers followed by dense layers.
    Also normalizes input, assuming a (0, 255) input range.
    """

    features: Sequence
    convert_nchw: bool

    @nn.compact
    def __call__(self, x):
        if self.convert_nchw:
            # Convert ..CHW input to ..HWC
            x = jnp.swapaxes(x, -1, -3)
            x = jnp.swapaxes(x, -2, -3)
        x = x / 255.0
        for conv_layer_id in range(len(self.features)):
            # Convolutional layers have form (features, kernel_width, stride)
            if type(self.features[conv_layer_id]) != tuple:
                break
            features, kernel_width, stride = self.features[conv_layer_id]
            x = nn.relu(nn.Conv(features, kernel_width, stride, padding="VALID")(x))
        # Flatten output from final conv layer
        x = jnp.reshape(x, (*x.shape[:-3], -1))
        for dense_layer_idx in range(conv_layer_id, len(self.features)):
            x = nn.relu(nn.Dense(self.features[dense_layer_idx])(x))
        return x
