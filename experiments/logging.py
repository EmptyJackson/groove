import os
import jax
import wandb

from flax.training import checkpoints


CKPT_DIR = "checkpoints"


def init_logger(args):
    assert (
        args.wandb_project and args.wandb_entity
    ), "Must provide --wandb_project and --wandb_entity arguments to log results."
    wandb.init(
        config=args,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        job_type="train",
    )
    os.mkdir(os.path.join(wandb.run.dir, CKPT_DIR))


def log_results(args, metrics, train_state, level_buffer):
    # Log metrics
    for step in range(args.train_steps):
        wandb.log(jax.tree_map(lambda x: x[step], metrics))

    # Log checkpoints
    ckpt_path = checkpoints.save_checkpoint(
        ckpt_dir=os.path.join(wandb.run.dir, CKPT_DIR),
        target=train_state,
        step=args.train_steps,
        keep=1,
    )
    wandb.save(ckpt_path, base_path=wandb.run.dir, policy="now")
    if level_buffer is not None:
        buffer_ckpt_path = checkpoints.save_checkpoint(
            ckpt_dir=os.path.join(wandb.run.dir, CKPT_DIR),
            target=level_buffer,
            step=args.train_steps,
            keep=1,
            prefix="buffer_",
        )
        wandb.save(buffer_ckpt_path, base_path=wandb.run.dir, policy="now")
    wandb.finish()
