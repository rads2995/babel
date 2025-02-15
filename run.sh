#!/usr/bin/env bash

podman run \
--volume=$(pwd):/home/builder/babel:rw,U,Z \
-it \
--rm \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
--device=/dev/kfd \
--device=/dev/dri \
--group-add video \
--ipc=host \
babel:latest
