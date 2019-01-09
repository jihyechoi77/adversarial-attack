#!/bin/bash

export PYTHONPATH=/data/jihyec/code/facenet-master/src
export PYTHONPATH="/data/jihyec/adversarial-attack/src":$PYTHONPATH
export PYTHONPATH="/data/jihyec/code/cleverhans/cleverhans":$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/lib

# to fix ValueError: unknown locale: UTF-8 error when importing matplotlib
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

