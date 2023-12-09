#!/bin/bash
WANDB_API_KEY=$(cat ./setup/wandb_key)
git pull

script_and_args="${@:1}"
echo "Launching container groove_cpu"
docker run \
    --env CUDA_VISIBLE_DEVICES='-1' \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -v $(pwd):/home/duser/groove \
    --name groove_cpu \
    --user $(id -u) \
    --rm \
    -d \
    -t groove_cpu \
    /bin/bash -c "$script_and_args"
