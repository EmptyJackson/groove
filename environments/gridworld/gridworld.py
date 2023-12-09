import jax
import chex
import itertools
import jax.numpy as jnp

from flax import struct
from typing import Tuple, Optional
from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState:
    time: int
    pos: int
    # obj_poss is incremented by obj_id * number_of_grid_cells
    obj_poss: chex.Array
    obj_existss: chex.Array
    early_term: bool


@struct.dataclass
class EnvParams:
    max_steps_in_episode: int
    random_respawn: bool        # If true, objects respawn at random positions, else they respawn at static_obj_poss
    auto_collect: bool       # If true, agent automatically collects objects when it moves over them
    grid_size: int
    walls: chex.Array
    start_pos: int
    n_objs: int
    obj_ids: chex.Array
    static_obj_poss: chex.Array       # Object positions when random respawn disabled
    # Properties of each object type
    obj_rewards: chex.Array
    obj_p_terminate: chex.Array
    obj_p_respawn: chex.Array


class GridWorld(environment.Environment):
    """
    Gridworld environment, as defined in https://arxiv.org/abs/2007.08794.

    For JIT compatibility, we define and use the maximum grid size and number of objects,
    masking them out with environment parameters when necessary.
    """

    def __init__(self, max_grid_size: int = 11, max_n_objs: int = 4, max_n_obj_types: int = 3, tabular: bool = True):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.max_n_objs = max_n_objs
        self.max_n_obj_types = max_n_obj_types
        self.tabular = tabular

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters for GridWorld: Tabular and Dense with automatic collection
        return EnvParams(
            max_steps_in_episode = 500,
            random_respawn = False,
            auto_collect = True,
            grid_size = 11,
            walls = jnp.zeros((11*11,), dtype=jnp.bool_),
            start_pos = 0,
            n_objs = 4,
            obj_ids = jnp.array([0, 0, 1, 2]),
            static_obj_poss = jnp.array([1*11+3, 3*11+7, 8*11+7, 9*11+2]),
            # each object type,
            obj_rewards = jnp.array([1.0, -1.0, -1.0]),
            obj_p_terminate = jnp.array([0.0, 0.5, 0.0]),
            obj_p_respawn = jnp.array([0.05, 0.1, 0.5]),
        )

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        term_key, respawn_key, obj_key = jax.random.split(key, 3)

        # Update position
        pos = self._get_next_pos(state.pos, action, params)

        # Check if any existing objects collected, TODO: Handle explicit collection
        # state.obj_poss stores object position plus obj_id * number_of_grid_cells, see get_obs for explaination
        old_obj_poss = state.obj_poss - params.obj_ids * self.max_grid_size**2
        obj_collected = jnp.logical_and(state.obj_existss, jnp.equal(old_obj_poss, pos))

        # Spawn new objects
        padded_p_respawn = jnp.take(params.obj_p_respawn, params.obj_ids)
        respawn = jax.random.bernoulli(respawn_key, padded_p_respawn)
        obj_existss = jnp.logical_or(state.obj_existss, respawn)

        if self.tabular:
            # Random object respawn not defined for tabular environments
            obj_poss = old_obj_poss
        else:
            # Generate new object positions for respawned objects
            max_grid_idxs = jnp.arange(self.max_grid_size**2)
            p_vacant = jnp.logical_and(max_grid_idxs < params.grid_size**2,
                                    jnp.logical_not(jnp.isin(max_grid_idxs, pos)),
                                    jnp.logical_not(jnp.isin(max_grid_idxs, params.walls)))
            p_vacant = p_vacant.at[old_obj_poss].set(False)
            p_vacant = jnp.divide(p_vacant, jnp.sum(p_vacant))
            random_obj_poss = jax.random.choice(obj_key, max_grid_idxs, (self.max_n_objs,), p=p_vacant, replace=False)
            use_new_poss = jnp.logical_and(jnp.logical_not(state.obj_existss), respawn)
            new_obj_poss = jnp.where(use_new_poss, random_obj_poss, old_obj_poss)
            # Update object positions (if random respawn enabled)
            obj_poss = jnp.where(params.random_respawn, new_obj_poss, old_obj_poss)
        obj_poss += params.obj_ids * self.max_grid_size**2

        # Remove collected objects
        obj_existss = jnp.logical_and(obj_existss, jnp.logical_not(obj_collected))

        # Remove unused objects
        used_mask = jnp.arange(self.max_n_objs) < params.n_objs
        obj_existss = jnp.logical_and(obj_existss, used_mask)

        # Update early termination
        padded_p_terminate = jnp.take(params.obj_p_terminate, params.obj_ids)
        term = jnp.logical_or(jax.random.bernoulli(term_key, jnp.dot(padded_p_terminate, obj_collected)), state.early_term)

        # Update time
        time = state.time + 1

        # Compute reward
        padded_obj_rewards = jnp.take(params.obj_rewards, params.obj_ids)
        reward = jnp.dot(padded_obj_rewards, obj_collected)

        # Compute done
        state = EnvState(time, pos, obj_poss, obj_existss, term)
        done = self.is_terminal(state, params)
        info = {"discount": self.discount(state, params)}

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            info
        )

    def _get_next_pos(self, pos: int, action: int, params: EnvParams) -> int:
        top_row, bottom_row = (pos < params.grid_size), (pos >= params.grid_size * (params.grid_size - 1))
        left_column, right_column = (pos % params.grid_size == 0), (pos % params.grid_size == params.grid_size - 1)
        step = (action == 0) * (1 - top_row) * -params.grid_size \
             + (action == 1) * (1 - bottom_row) * params.grid_size \
             + (action == 2) * (1 - left_column) * -1 \
             + (action == 3) * (1 - right_column) * 1
        next_pos = pos + step
        return jnp.where(params.walls[next_pos], pos, next_pos)

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        obj_key, pos_key = jax.random.split(key)
        # pos=jax.random.randint(pos_key, (), 0, params.grid_size**2)
        pos = params.start_pos
        if self.tabular:
            obj_poss = params.static_obj_poss
        else:
            # Select random or static object position
            max_grid_idxs = jnp.arange(self.max_grid_size**2)
            valid_idxs = jnp.logical_and(max_grid_idxs < params.grid_size**2,
                                        jnp.logical_not(jnp.isin(max_grid_idxs, pos)),
                                        jnp.logical_not(jnp.isin(max_grid_idxs, params.walls)))
            p = jnp.divide(valid_idxs, jnp.sum(valid_idxs))
            random_obj_poss = jax.random.choice(obj_key, max_grid_idxs, (self.max_n_objs,), p=p, replace=False)
            obj_poss = jnp.where(params.random_respawn, random_obj_poss, params.static_obj_poss)
        # Object positions are incremented by a factor of their object id, see get_obs for explaination
        obj_poss += params.obj_ids * self.max_grid_size**2
        state = EnvState(
            time=0,
            pos=pos,
            obj_poss=obj_poss,
            obj_existss=jnp.arange(self.max_n_objs) < params.n_objs,
            early_term=False,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        if self.tabular:
            # Tabular observation (pos + max_pos * sum(obj_existss ** 2))
            obs = jnp.zeros((self.max_grid_size**2)*(2**self.max_n_objs), dtype=bool)
            obs = obs.at[self._get_tabular_pos(state.pos, state.obj_existss)].set(True)
        else:
            # Non-tabular observation
            obs = jnp.zeros(self.max_grid_size**2, dtype=bool)
            obs = obs.at[state.pos].set(True)
            # state.obj_poss stores object position plus obj_id * [max number of grid cells], since we don't have
            # access to the object ids in the environment params, we store the object positions in this way.
            obj_obs = jnp.zeros(self.max_n_obj_types * self.max_grid_size**2, dtype=bool)
            obj_obs = obj_obs.at[state.obj_poss].set(state.obj_existss)
            obs = jnp.concatenate((obs, obj_obs))
        return jnp.append(obs.astype(jnp.float32), state.time * 0.001)

    def _get_tabular_pos(self, pos: int, obj_existss: chex.Array) -> int:
        """Get position of state in tabular observation."""
        all_obj_idx = jnp.power(2, jnp.arange(self.max_n_objs))
        exist_obj_factor = jnp.sum(jnp.where(obj_existss, all_obj_idx, 0))
        return pos + (self.max_grid_size**2) * exist_obj_factor

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        done_steps = state.time >= params.max_steps_in_episode
        done = jnp.logical_or(done_steps, state.early_term)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "GridWorld-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        # TODO: Handle explicit collection actions
        return 5

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(5)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        if self.tabular:
            shape = (self.max_grid_size**2) * (2**self.max_n_objs) + 1
        else:
            shape = (self.max_grid_size**2) * (self.max_n_obj_types + 1) + 1
        return spaces.Box(
            low=0.0,
            high=params.max_steps_in_episode-1,
            shape=shape,
            dtype=jnp.float32,
        )

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict({
            "time": spaces.Discrete(params.max_steps_in_episode),
            "pos": spaces.Discrete(self.max_grid_size**2),
            "obj_poss": spaces.Box(low=0, high=(self.max_n_obj_types * self.max_grid_size**2)-1, shape=(self.max_n_objs,), dtype=jnp.int32),
            "obj_existss": spaces.Box(low=0, high=1, shape=(self.max_n_objs,), dtype=jnp.int32),
            "early_term": spaces.Discrete(2),
        })

    def optimal_return(self, params: EnvParams, max_rollout_len: int, return_all: bool = False) -> float:
        """Optimal return of the environment with the given parameters."""
        if not self.tabular:
            raise NotImplementedError("Optimal return not implemented for non-tabular environments.")
        def _step_value(v_t1: jnp.array, time: int) -> tuple:
            """Compute value function at a given time step in the environment."""
            def _state_value(pos_t: int, obj_existss_t: jnp.array) -> float:
                """Optimal return of a given state in the environment."""
                def _q_value(action: int) -> float:
                    """Optimal return of a given state and action in the environment."""
                    # Compute next position
                    pos_t1 = self._get_next_pos(pos_t, action, params)
                    # Check if any existing objects collected
                    obj_collected = jnp.logical_and(obj_existss_t, jnp.equal(params.static_obj_poss, pos_t1))
                    # Compute reward
                    padded_obj_rewards = jnp.take(params.obj_rewards, params.obj_ids)
                    r = jnp.dot(padded_obj_rewards, obj_collected)
                    # Get object respawn probability
                    padded_p_respawn = jnp.take(params.obj_p_respawn, params.obj_ids)
                    # Compute ordered permutations of object existence states
                    obj_exist_states = jnp.array(list(itertools.product([0, 1], repeat=self.max_n_objs)))
                    obj_exist_states = jnp.flip(obj_exist_states, axis=1)
                    # Compute probability of each next object existence state
                    p_next_obj_existss = jnp.ones(obj_exist_states.shape[0])
                    for i in range(self.max_n_objs):
                        # Compute marginal probability of next object i state
                        obj_i_exist_states = obj_exist_states[:, i]
                        unused_p = 1 - obj_i_exist_states
                        collected_p = 1 - obj_i_exist_states
                        exists_p = obj_i_exist_states
                        p_next_obj_existss *= jnp.where(
                            i >= params.n_objs, unused_p, jnp.where(        # Object unused in level
                                obj_collected[i], collected_p, jnp.where(   # Object exists and collected
                                    obj_existss_t[i], exists_p, jnp.where(  # Object exists and not collected
                                        obj_i_exist_states,
                                        padded_p_respawn[i],                # Object respawns
                                        (1 - padded_p_respawn[i])))))       # Object doesn't respawn
                    # Compute tabular index of next positions
                    tab_pos_t1 = jax.vmap(self._get_tabular_pos, in_axes=(None, 0))(pos_t1, obj_exist_states)
                    # Compute value
                    v = jnp.dot(p_next_obj_existss, jnp.where(p_next_obj_existss > 0., v_t1[tab_pos_t1], 0.))
                    padded_p_terminate = jnp.take(params.obj_p_terminate, params.obj_ids)
                    p_term = jnp.dot(padded_p_terminate, obj_collected)
                    return r + v * (1 - p_term)

                q_max = jax.vmap(_q_value)(jnp.arange(self.num_actions)).max()
                # Return minimum reward if state is invalid at time t (position out-of-bounds or any unused objects exist)
                invalid_pos = jnp.logical_or((pos_t >= params.grid_size**2), params.walls[pos_t])
                invalid_obj = jnp.logical_and(obj_existss_t, jnp.arange(self.max_n_objs) >= params.n_objs).any()
                return jnp.where(jnp.logical_or(invalid_pos, invalid_obj), -jnp.inf, q_max)

            # Construct all permutations of position and object existence
            pos = jnp.arange(self.max_grid_size**2)
            obj_exist = tuple(jnp.arange(2) for _ in range(self.max_n_objs))
            all_states = jnp.array(jnp.meshgrid(pos, *obj_exist, indexing='ij')).T.reshape(-1, self.max_n_objs+1)
            # Return value of each state
            v = jax.vmap(_state_value)(all_states[:, 0], all_states[:, 1:])
            v = jnp.where(time < params.max_steps_in_episode, v, 0.)
            return v, v

        v_0, v = jax.lax.scan(
            _step_value,
            jnp.zeros((self.max_grid_size**2)*(2**self.max_n_objs)),
            jnp.flip(jnp.arange(max_rollout_len)),
            max_rollout_len,
        )
        if return_all:
            return jnp.flip(v, axis=0)
        else:
            start_pos = self._get_tabular_pos(params.start_pos, jnp.arange(self.max_n_objs) < params.n_objs)
            return v_0[start_pos]

    def __eq__(self, other):
        if not isinstance(other, GridWorld):
            return NotImplemented

        return self.max_grid_size == other.max_grid_size and \
               self.max_n_objs == other.max_n_objs and \
               self.max_n_obj_types == other.max_n_obj_types and \
               self.tabular == other.tabular

    def __hash__(self):
        return hash((self.max_grid_size, self.max_n_objs, self.max_n_obj_types, self.tabular))


registered_envs = ["GridWorld-v0"]
