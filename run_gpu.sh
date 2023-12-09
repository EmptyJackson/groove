#!/bin/bash
WANDB_API_KEY=$(cat ./setup/wandb_key)
git pull

script_and_args="${@:2}"
gpu=$1
echo "Launching container groove_gpu_$gpu on GPU $gpu"
docker run \
    --env CUDA_VISIBLE_DEVICES=$gpu \
    --gpus all \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -v $(pwd):/home/duser/groove \
    --name groove_gpu_$gpu \
    --user $(id -u) \
    --rm \
    -d \
    -t groove_gpu \
    /bin/bash -c "$script_and_args"
