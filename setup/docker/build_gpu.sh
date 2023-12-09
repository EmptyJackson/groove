#!/bin/bash

echo 'Building GPU image with name groove_gpu'
docker build \
    --build-arg UID=$(id -u ${USER}) \
    --build-arg GID=1234 \
    --build-arg REQS="$(cat ../requirements-base.txt ../requirements-gpu.txt | tr '\n' ' ')" \
    -t groove_gpu \
    .
