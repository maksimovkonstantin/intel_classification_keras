#!/usr/bin/env bash
docker build -t $(whoami)/intel:env \
--build-arg user_name=$(whoami) --build-arg user_id=$(id -u) .