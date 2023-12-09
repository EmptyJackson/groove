import chex
import jax
import jax.numpy as jnp

from jax import random
from jax.tree_util import tree_map
from flax import struct
from functools import partial

from util import *
from environments.environments import get_env, reset_env_params, get_env_spec
from environments.rollout import RolloutWrapper
from agents.agents import (
    create_agent,
    create_value_critic,
    eval_agent,
    AgentHyperparams,
)
from agents.a2c import train_a2c_agent, A2CHyperparams


# TODO: Reimplement positive_value_loss and l1_value_loss
SCORE_FUNCTIONS = ["random", "frozen", "alg_regret"]
SCORE_TRANSFORMS = ["proportional", "rank"]


@struct.dataclass
class LevelBuffer:
    level: Level  # Level parameters and lifetime
    score: float  # Most recent score on the level
    active: bool  # Whether the level is currently being evaluated
    new: bool  # Whether the level has been evaluated

    @staticmethod
    def create_buffer(params, lifetimes):
        def _create_level(params, lifetime, buffer_idx):
            return LevelBuffer(
                level=Level(params, lifetime, buffer_idx),
                score=0.0,
                active=False,
                new=True,
            )

        return jax.vmap(_create_level)(
            params, lifetimes, jnp.arange(lifetimes.shape[0])
        )

    def __len__(self):
        return self.score.shape[0]


class LevelSampler:
    """Level sampler, containing methods for domain randomisation and prioritised level replay."""

    def __init__(self, args):
        # --- Get environment and agent specifications ---
        self.env_name = args.env_name
        self.env_mode = args.env_mode
        self.env_workers = args.env_workers
        self.env_kwargs, self.max_rollout_len, self.max_lifetime = get_env_spec(
            self.env_name, self.env_mode
        )
        self.env = get_env(self.env_name, self.env_kwargs)
        self.rollout_manager = RolloutWrapper(
            self.env_name, args.train_rollout_len, self.max_rollout_len, self.env_kwargs
        )
        self.agent_hypers = AgentHyperparams.from_args(args)

        # --- Save sampler parameters ---
        if not args.score_function in SCORE_FUNCTIONS:
            raise ValueError(
                f"Level score function {args.score_function} not in known functions: {SCORE_FUNCTIONS}"
            )
        if not args.score_transform in SCORE_TRANSFORMS:
            raise ValueError(
                f"Level score transform {args.score_transform} not in known transforms: {SCORE_TRANSFORMS}"
            )
        self.score_function = args.score_function
        self.score_transform = args.score_transform
        self.score_temperature = args.score_temperature
        self.buffer_size = args.buffer_size
        self.p_replay = args.p_replay
        self.num_mini_batches = args.num_mini_batches
        self.a2c_hypers = A2CHyperparams(
            args.gamma, args.gae_lambda, args.entropy_coeff
        )

    def initialize_buffer(self, rng):
        """Creates a new level buffer, if used by the score function."""
        if self.score_function == "random":
            return None
        rng = jax.random.split(rng, self.buffer_size)
        random_params, random_lifetimes = self._sample_env_params(rng)
        return LevelBuffer.create_buffer(random_params, random_lifetimes)

    @partial(jax.vmap, in_axes=(None, 0))
    def _sample_env_params(self, rng):
        """Sample a batch of environment parameters and agent lifetimes."""
        return reset_env_params(rng, self.env_name, self.env_mode)

    def initial_sample(
        self,
        rng: chex.PRNGKey,
        level_buffer: LevelBuffer,
        batch_size: int,
        create_value_critics: bool,
    ):
        """Sample random initial levels and agents."""
        # --- Sample initial levels ---
        if self.score_function == "random":
            rng, _rng = jax.random.split(rng)
            levels = self._sample_random_levels(_rng, batch_size)
        else:
            levels = jax.tree_map(lambda x: x[:batch_size], level_buffer.level)
            level_buffer = level_buffer.replace(
                active=jnp.arange(self.buffer_size) < batch_size
            )

        # --- Initialise agents ---
        rng, _rng = random.split(rng)
        _rng = random.split(_rng, batch_size)
        agent_states = jax.vmap(self._create_agent)(_rng, levels)
        value_critics = None
        if create_value_critics:
            rng, _rng = jax.random.split(rng)
            _rng = jax.random.split(_rng, batch_size)
            value_critics = jax.vmap(create_value_critic, in_axes=(0, None, None))(
                _rng, self.agent_hypers, self.obs_shape
            )
        return level_buffer, agent_states, value_critics

    def sample(
        self,
        rng: chex.PRNGKey,
        level_buffer: LevelBuffer,
        old_agents: AgentState,
        old_value_critics: TrainState,
    ):
        """Update level buffer and sample new levels for terminated agents."""
        # --- Identify terminated agents ---
        terminated_mask = old_agents.actor_state.step >= old_agents.level.lifetime
        term_mask_fn = lambda term_val, active_val: jax.vmap(jnp.where)(
            terminated_mask, term_val, active_val
        )
        batch_size = terminated_mask.shape[0]

        if self.score_function == "random":
            # --- Sample random levels ---
            rng, _rng = jax.random.split(rng)
            new_levels = self._sample_random_levels(_rng, batch_size)
            new_levels = jax.tree_map(term_mask_fn, new_levels, old_agents.level)

        elif self.score_function == "frozen":
            # --- Randomly sample from frozen buffer ---
            p_uniform = jnp.ones((self.buffer_size,)) / self.buffer_size
            rng, _rng = jax.random.split(rng)
            level_ids = random.choice(
                _rng,
                jnp.arange(self.buffer_size),
                p=p_uniform,
                shape=(batch_size,),
                replace=True,
            )
            new_levels = jax.tree_map(lambda x: x[level_ids], level_buffer.level)
            new_levels = jax.tree_map(term_mask_fn, new_levels, old_agents.level)

        else:
            # --- Reset buffer levels ---
            rng, _rng = jax.random.split(rng)
            # Ensures there are at least batch_size new, inactive levels in the buffer
            level_buffer = self._reset_lowest_scoring(_rng, level_buffer, batch_size)

            # --- Evaluate agent and compute regret ---
            if self.score_function == "alg_regret":
                rng, _rng = jax.random.split(rng)
                _rng = jax.random.split(_rng, batch_size)
                score = mini_batch_vmap(
                    self._compute_algorithmic_regret, self.num_mini_batches
                )(_rng, old_agents)
            else:
                raise NotImplementedError(
                    f"Level score function {self.score_function} is not implemented."
                )

            # --- Update buffer with terminated levels ---
            old_ids = old_agents.level.buffer_id
            term_score = term_mask_fn(score, level_buffer.score[old_ids])
            term_active = term_mask_fn(
                jnp.full((batch_size,), False), level_buffer.active[old_ids]
            )
            term_new = term_mask_fn(
                jnp.full((batch_size,), False), level_buffer.new[old_ids]
            )
            level_buffer = level_buffer.replace(
                score=level_buffer.score.at[old_ids].set(term_score),
                active=level_buffer.active.at[old_ids].set(term_active),
                new=level_buffer.new.at[old_ids].set(term_new),
            )

            # --- Sample replay and random levels ---
            rng, replay_rng, random_rng = jax.random.split(rng, 3)
            replay_levels = self._replay_from_buffer(
                replay_rng, level_buffer, batch_size
            )
            random_levels = self._sample_random_from_buffer(
                random_rng, level_buffer, batch_size
            )

            # --- Select replay vs random levels ---
            rng, _rng = jax.random.split(rng)
            n_to_replay = jnp.sum(
                random.bernoulli(_rng, self.p_replay, shape=(batch_size,))
            )
            use_replay = jnp.arange(batch_size) < n_to_replay
            n_replayable = self.buffer_size - jnp.sum(
                jnp.logical_or(level_buffer.new, level_buffer.active)
            )
            # Replay only when there are enough inactive, evaluated levels in buffer
            use_replay = jnp.logical_and(use_replay, n_replayable >= batch_size)
            rng, _rng = jax.random.split(rng)
            # Shuffle to remove bias
            use_replay = random.permutation(_rng, use_replay)
            select_fn = lambda x, y: jax.vmap(jnp.where)(use_replay, x, y)
            # Select new levels from replay or random sets
            new_levels = jax.tree_map(select_fn, replay_levels, random_levels)
            # Use old levels for non-termianted agents
            new_levels = jax.tree_map(term_mask_fn, new_levels, old_agents.level)

            # --- Update active status of new levels in buffer ---
            level_buffer = level_buffer.replace(
                active=level_buffer.active.at[new_levels.buffer_id].set(True)
            )

        # --- Initialise new agents and environment workers ---
        rng, _rng = random.split(rng)
        _rng = random.split(_rng, batch_size)
        agent_states = jax.vmap(self._create_agent)(_rng, new_levels)

        # --- Initialise new value critics (if required) ---
        new_value_critics = None
        if old_value_critics is not None:
            rng, _rng = jax.random.split(rng)
            _rng = jax.random.split(_rng, batch_size)
            new_value_critics = jax.vmap(create_value_critic, in_axes=(0, None, None))(
                _rng, self.agent_hypers, self.obs_shape
            )

        # --- Return updated level buffer and agents ---
        # TODO: Had a function mismatch error without the below hack, need to investigate further.
        agent_states = agent_states.replace(
            critic_state=agent_states.critic_state.replace(
                tx=old_agents.critic_state.tx, apply_fn=old_agents.critic_state.apply_fn
            ),
            actor_state=agent_states.actor_state.replace(
                tx=old_agents.actor_state.tx, apply_fn=old_agents.actor_state.apply_fn
            ),
        )
        if new_value_critics is not None:
            new_value_critics = new_value_critics.replace(
                tx=old_value_critics.tx, apply_fn=old_value_critics.apply_fn
            )
        agent_states = jax.tree_map(term_mask_fn, agent_states, old_agents)
        value_critics = jax.tree_map(term_mask_fn, new_value_critics, old_value_critics)
        return level_buffer, agent_states, value_critics

    def _sample_random_levels(self, rng: chex.PRNGKey, batch_size: int):
        rng = jax.random.split(rng, batch_size)
        new_params, new_lifetimes = self._sample_env_params(rng)
        return Level(new_params, new_lifetimes, jnp.zeros(batch_size))

    def _create_agent(self, rng, level, value_critic=False):
        """Initialise an agent on the given level."""
        worker_rng, agent_rng = random.split(rng)
        env_obs, env_state = self.rollout_manager.batch_reset(
            worker_rng, level.env_params, self.env_workers
        )
        agent_hypers = self.agent_hypers
        if value_critic:
            agent_hypers = agent_hypers.replace(critic_dims=1)
        actor_state, critic_state = create_agent(
            agent_rng, agent_hypers, self.num_actions, self.obs_shape
        )
        return AgentState(
            actor_state=actor_state,
            critic_state=critic_state,
            level=level,
            env_obs=env_obs,
            env_state=env_state,
        )

    def _compute_algorithmic_regret(
        self, rng: chex.PRNGKey, lpg_agent_state: AgentState
    ):
        # --- Create antagonist (A2C) agent ---
        rng, _rng = jax.random.split(rng)
        a2c_agent_state = self._create_agent(
            _rng, lpg_agent_state.level, value_critic=True
        )

        # --- Train antagonist agent ---
        rng, _rng = jax.random.split(rng)
        a2c_agent_state, _ = train_a2c_agent(
            rng=_rng,
            agent_state=a2c_agent_state,
            rollout_manager=self.rollout_manager,
            num_train_steps=self.max_lifetime,
            hypers=self.a2c_hypers,
        )

        # --- Evaluate LPG and antagonist agents ---
        eval_fn = partial(
            eval_agent,
            rollout_manager=self.rollout_manager,
            num_workers=self.env_workers,
        )
        lpg_rng, a2c_rng = jax.random.split(rng)
        lpg_agent_return = eval_fn(
            rng=lpg_rng,
            env_params=lpg_agent_state.level.env_params,
            actor_train_state=lpg_agent_state.actor_state,
        )
        a2c_agent_return = eval_fn(
            rng=a2c_rng,
            env_params=lpg_agent_state.level.env_params,
            actor_train_state=a2c_agent_state.actor_state,
        )
        return a2c_agent_return - lpg_agent_return

    def _reset_lowest_scoring(
        self, rng: chex.PRNGKey, level_buffer: LevelBuffer, minimum_new: int
    ):
        """
        Reset the lowest scoring levels in the buffer.
        Ensures there are at least minimum_new new, inactive levels.
        """
        # --- Identify lowest scoring levels ---
        level_scores = jnp.where(level_buffer.new, -jnp.inf, level_buffer.score)
        level_scores = jnp.where(level_buffer.active, jnp.inf, level_scores)
        reset_ids = jnp.argsort(level_scores)[:minimum_new]
        rng = jax.random.split(rng, minimum_new)
        new_params, new_lifetimes = self._sample_env_params(rng)
        new_levels = Level(new_params, new_lifetimes, reset_ids)

        # --- Reset lowest scoring levels ---
        reset_fn = lambda x, y: x.at[reset_ids].set(y)
        return level_buffer.replace(
            level=tree_map(reset_fn, level_buffer.level, new_levels),
            score=level_buffer.score.at[reset_ids].set(0.0),
            active=level_buffer.active.at[reset_ids].set(False),
            new=level_buffer.active.at[reset_ids].set(True),
        )

    def _replay_from_buffer(
        self, rng: chex.PRNGKey, level_buffer: LevelBuffer, batch_size: int
    ):
        """
        Samples previously evaluated environment levels from the buffer.
        Levels are returned in sample order, which is significant for e.g. rank-based sampling.
        If there are not enough inactive, evaluated levels in the buffer, returns random levels.
        """
        invalid_levels = jnp.logical_or(level_buffer.new, level_buffer.active)
        # Softmax-normalize scores
        scores = jnp.exp(level_buffer.score / self.score_temperature)
        scores = jnp.where(invalid_levels, 0.0, scores)
        scores /= scores.sum()
        # Return uniform (invalid) score when there aren't enough inactive, seen levels in buffer
        p_replay = jnp.where(
            self.buffer_size - jnp.sum(invalid_levels) < batch_size,
            jnp.ones_like(scores),
            scores,
        )
        if self.score_transform == "rank":
            level_ids = jnp.flip(jnp.argsort(p_replay))[:batch_size]
        elif self.score_transform == "proportional":
            # Sample levels stochastically
            rng, _rng = jax.random.split(rng)
            level_ids = random.choice(
                _rng,
                jnp.arange(self.buffer_size),
                p=p_replay,
                shape=(batch_size,),
                replace=False,
            )
        else:
            raise NotImplementedError(
                f"Level score transform {self.score_transform} is not implemented."
            )
        return jax.tree_map(lambda x: x[level_ids], level_buffer.level)

    def _sample_random_from_buffer(
        self, rng: chex.PRNGKey, level_buffer: LevelBuffer, batch_size: int
    ):
        """Samples new (unevaluated), inactive levels from the buffer."""
        random_level_mask = jnp.logical_and(
            level_buffer.new, jnp.logical_not(level_buffer.active)
        )
        p_sample = jnp.where(random_level_mask, 1.0, 0.0)
        p_sample = p_sample / jnp.sum(p_sample)
        level_ids = random.choice(
            rng,
            jnp.arange(self.buffer_size),
            p=p_sample,
            shape=(batch_size,),
            replace=False,
        )
        return jax.tree_map(lambda x: x[level_ids], level_buffer.level)

    @property
    def num_actions(self):
        return self.env.num_actions

    @property
    def obs_shape(self):
        return self.env.observation_space(self.env.default_params).shape
