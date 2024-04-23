import jax
import sys

from jax import random
from rich.traceback import install

from util import *
from environments.level_sampler import LevelSampler
from experiments.parse_args import parse_args
from experiments.logging import init_logger, log_results
from meta.meta import create_lpg_train_state, make_lpg_train_step


def make_train(args):
    def _train_fn(rng):
        # --- Initialize LPG and level sampler ---
        rng, lpg_rng, buffer_rng = jax.random.split(rng, 3)
        train_state = create_lpg_train_state(lpg_rng, args)
        level_sampler = LevelSampler(args)
        level_buffer = level_sampler.initialize_buffer(buffer_rng)

        # --- Initialze agents and value critics ---
        require_value_critic = not args.use_es
        rng, _rng = jax.random.split(rng)
        level_buffer, agent_states, value_critic_states = level_sampler.initial_sample(
            _rng, level_buffer, args.num_agents, require_value_critic
        )

        # --- TRAIN LOOP ---
        lpg_train_step_fn = make_lpg_train_step(args, level_sampler)

        def _meta_train_loop(carry, _):
            rng, train_state, agent_states, value_critic_states, level_buffer = carry

            # --- Update LPG ---
            rng, _rng = jax.random.split(rng)
            train_state, agent_states, value_critic_states, metrics = lpg_train_step_fn(
                rng=_rng,
                lpg_train_state=train_state,
                agent_states=agent_states,
                value_critic_states=value_critic_states,
            )

            # --- Sample new levels and agents as required ---
            rng, _rng = jax.random.split(rng)
            level_buffer, agent_states, value_critic_states = level_sampler.sample(
                _rng, level_buffer, agent_states, value_critic_states
            )
            carry = (rng, train_state, agent_states, value_critic_states, level_buffer)
            return carry, metrics

        # --- Stack and return metrics ---
        carry = (rng, train_state, agent_states, value_critic_states, level_buffer)
        carry, metrics = jax.lax.scan(
            _meta_train_loop, carry, None, length=args.train_steps
        )
        return metrics, train_state, level_buffer

    return _train_fn


def run_training_experiment(args):
    if args.log:
        init_logger(args)
    train_fn = make_train(args)
    rng = random.PRNGKey(args.seed)
    metrics, train_state, level_buffer = jax.jit(train_fn)(rng)
    if args.log:
        log_results(args, metrics, train_state, level_buffer)
    else:
        print(metrics)


def main(cmd_args=sys.argv[1:]):
    args = parse_args(cmd_args)
    experiment_fn = jax_debug_wrapper(args, run_training_experiment)
    return experiment_fn(args)


if __name__ == "__main__":
    install()
    main()
