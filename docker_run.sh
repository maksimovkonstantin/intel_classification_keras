#!/bin/bash
ARG1=${1:-0}
ARG2=${2:-9999}
NV_GPU=$ARG1 nvidia-docker run \
--user $(id -u):$(id -g) \
-p $ARG2:$ARG2 \
--mount type=bind,source="$PWD",target=/project -it --rm \
$(whoami)/intel:env bash
