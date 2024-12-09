#!/usr/bin/env bash

python run.py --data aifb --epochs 50 --bases 0 --hidden 16 --lr 0.01 --l2 0.01 --lambda_p 0.2 --seed 0
python run.py --data mutag --epochs 50 --bases 35 --hidden 16 --lr 0.01 --l2 5e-4  --drop 0.3 --lambda_p 0.2 --no_cuda --seed 0
python run.py --data am --epochs 50 --bases 40 --hidden 10 --lr 0.01 --l2 5e-4 --lambda_p 0.2 --no_cuda --seed 0
python run.py --data bgs --epochs 50 --bases 40 --hidden 16 --lr 0.01 --l2 5e-3 --drop 0.2 --no_cuda --lambda_p 0.2 --no_cuda --seed 0
