from functools import partial

from util import *
from flax.training.train_state import TrainState
from models.lpg import LPG
from models.optim import create_optimizer, create_es_strategy
from meta.train import lpg_meta_grad_train_step, lpg_es_train_step


def create_lpg_train_state(rng, args):
    """
    Initialises an LPG instance.
    Returns TrainState if using meta-gradients, return ESTrainState if using ES.
    """
    lpg_model = LPG(
        embedding_net_width=args.lpg_embedding_net_width,
        gru_width=args.lpg_gru_width,
        target_width=args.lpg_target_width,
        lifetime_conditioning=args.lifetime_conditioning,
    )
    r, d, pi, yt, yt1, step, lifetime = lpg_model.get_init_vector()
    params = lpg_model.init(rng, r, d, pi, yt, yt1, step, lifetime)["params"]
    tx = create_optimizer(args.lpg_opt, args.lpg_learning_rate, args.lpg_max_grad_norm)
    train_state = TrainState.create(apply_fn=lpg_model.apply, params=params, tx=tx)
    if not args.use_es:
        return train_state
    es_strategy = create_es_strategy(args, train_state.params)
    es_params = es_strategy.default_params
    es_state = es_strategy.initialize(rng, es_params)
    return ESTrainState(train_state, es_strategy, es_params, es_state)


def make_lpg_train_step(args, rollout_manager):
    lpg_hypers = LpgHyperparams.from_run_args(args)
    if args.use_es:
        return partial(
            lpg_es_train_step,
            rollout_manager=rollout_manager,
            num_mini_batches=args.num_mini_batches,
            lpg_hypers=lpg_hypers,
        )
    return partial(
        lpg_meta_grad_train_step,
        rollout_manager=rollout_manager,
        num_mini_batches=args.num_mini_batches,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        lpg_hypers=lpg_hypers,
    )
