#!/bin/sh

g++ -std=c++14 -fexceptions -pthread -g -O2 -I../../tiny-dnn/ -o train train.cpp
chmod +x ./train