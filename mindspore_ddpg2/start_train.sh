#!/bin/bash

docker run -i --rm -v "${PWD}":/app -w /app --name fighting_game_ddpg2 fighting_game:v1 /bin/sh -c 'Xvfb :10 -screen 0 1024x768x16 & export DISPLAY=:10; python train.py > train_log.txt 2> train_error_log.txt'

# 这个命令是测试Java环境是否运行正常使用
# docker run -i --rm -v "${PWD}":/app -w /app --name fighting_game fighting_game:v1 /bin/sh -c 'Xvfb :10 -screen 0 1024x768x16 & export DISPLAY=:10; bash ftg.sh'