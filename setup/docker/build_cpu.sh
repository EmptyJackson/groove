#!/bin/bash

echo 'Building CPU image with name groove_cpu'
docker build \
    --build-arg UID=$(id -u ${USER}) \
    --build-arg GID=1234 \
    --build-arg REQS="$(cat ../requirements-base.txt ../requirements-cpu.txt | tr '\n' ' ')" \
    -t groove_cpu \
    .
