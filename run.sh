#!/usr/bin/env bash

podman run \
--volume=$(pwd):/home/babel:rw,U,Z \
-it \
--user $(id -u):$(id -g) \
--userns=keep-id \
babel:latest

# Flags for AMD ROCm GPU support
# --cap-add=SYS_PTRACE \
# --security-opt seccomp=unconfined \
# --device=/dev/kfd \
# --device=/dev/dri \
# --group-add video \
# --ipc=host \
