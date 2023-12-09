import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState
from evosax import Strategy, EvoParams, EvoState
from typing import Any


@struct.dataclass
class LpgHyperparams:
    """
    num_agent_updates: Number of inner-loop agent updates per LPG update.
    agent_target_coeff (alpha_y): Agent target KL divergence.
    policy_entropy_coeff (beta_0): Trained agent policy entropy.
    target_entropy_coeff (beta_1): Trained agent target entropy.
    policy_l2_coeff (beta_2): Policy update (pi_hat) L2 regularization.
    target_l2_coeff (beta_3): Target update (y_hat) L2 regularization.
    """

    num_agent_updates: int
    agent_target_coeff: float
    policy_entropy_coeff: float
    target_entropy_coeff: float
    policy_l2_coeff: float
    target_l2_coeff: float

    @staticmethod
    def from_run_args(args):
        return LpgHyperparams(
            num_agent_updates=args.num_agent_updates,
            agent_target_coeff=args.lpg_agent_target_coeff,
            policy_entropy_coeff=args.lpg_policy_entropy_coeff,
            target_entropy_coeff=args.lpg_target_entropy_coeff,
            policy_l2_coeff=args.lpg_policy_l2_coeff,
            target_l2_coeff=args.lpg_target_l2_coeff,
        )


@struct.dataclass
class Transition:
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray


@struct.dataclass
class Level:
    env_params: Any
    lifetime: int
    buffer_id: int


@struct.dataclass
class AgentState:
    actor_state: TrainState
    critic_state: TrainState
    level: Level
    env_obs: jnp.array
    env_state: Any


# class ESTrainState(NamedTuple):
class ESTrainState(struct.PyTreeNode):
    """Extension of the Flax TrainState class for EvoSax agents"""

    train_state: TrainState = struct.field(pytree_node=True)
    strategy: Strategy = struct.field(pytree_node=False)
    es_params: EvoParams = struct.field(pytree_node=True)
    es_state: EvoState = struct.field(pytree_node=True)
