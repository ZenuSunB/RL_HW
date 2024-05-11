#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)

cd $SCRIPT_DIR

# for linux
#java -cp FightingICE.jar:./lib/lwjgl/*:./lib/natives/linux/*:./lib/*  Main --py4j --mute --port 4242
java  -cp FightingICE.jar:./lib/lwjgl/*:./lib/natives/linux/*:./lib/* Main --a1 BCP --a2 BlackMamba --c1 ZEN --c2 ZEN -n 1 --mute --fastmode --disable-window
# for macos
# java -XstartOnFirstThread -cp FightingICE.jar:./lib/lwjgl/*:./lib/natives/macos/*:./lib/* Main --limithp 400 400 --mute

# rm -rf ckpt/ akg_build_tmp.key  kernel_meta/ rank_0/ dsa_transfrom.log