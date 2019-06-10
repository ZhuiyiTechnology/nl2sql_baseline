#!/usr/bin/bash
CUDA_VISIBLE_DEVICES=$1 python test.py --ca --gpu --output_dir $2
