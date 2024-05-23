#!/bin/bash

script_self=$(readlink -f "$0")
slef_path=$(dirname "${script_self}")
if [ $# == 1 ]; then
  CKPT=$1
else
  echo "Usage: bash start_eval.sh [CKPT_PATH]."
  echo "Example: bash start_eval.sh ./ckpt/checkpoint_500.ckpt"
fi
export OMP_NUM_THREADS=10
docker run -i --rm -v "${PWD}":/app -w /app --name fighting_game_d3dqn_eval fighting_game:v1 /bin/sh -c 'Xvfb :10 -screen 0 1024x768x16 & export DISPLAY=:10; python eval.py --ckpt_path='${CKPT}' > eval_log.txt 2> eval_error_log.txt'
