#!/bin/bash
#BSUB -J SFAN
#BSUB -q gpu_v100
#BSUB -o %SFAN.out
#BSUB -e %SFAN.err
#BSUB -gpu "num=1:mode=exclusive_process"


python dehazing_SFAN.py --results_dir ../results/RSDehazing/SFAN/SateHaze1k_thick/ --img_h 256 --img_w 256 --train_batch_size 4 --dataset SateHaze1k_thick --total_epoches 100 --save_start_epoch 95 --device cuda:0 --step_gamma 0.9 --step_size 10 --lr 0.0002
python dehazing_SFAN.py --results_dir ../results/RSDehazing/SFAN/SateHaze1k_thin/ --img_h 256 --img_w 256 --train_batch_size 4 --dataset SateHaze1k_thin --total_epoches 100 --save_start_epoch 95 --device cuda:0 --step_gamma 0.9 --step_size 10 --lr 0.0002
python dehazing_SFAN.py --results_dir ../results/RSDehazing/SFAN/SateHaze1k_moderate/ --img_h 256 --img_w 256 --train_batch_size 4 --dataset SateHaze1k_moderate --total_epoches 100 --save_start_epoch 95 --device cuda:0 --step_gamma 0.9 --step_size 10 --lr 0.0002

python dehazing_SFAN.py --results_dir ../results/RSDehazing/SFAN/RICE1/ --img_h 256 --img_w 256 --train_batch_size 4 --dataset RICE1 --total_epoches 100 --save_start_epoch 95 --device cuda:0 --step_gamma 0.9 --step_size 10 --lr 0.0002
python dehazing_SFAN.py --results_dir ../results/RSDehazing/SFAN/RICE2/ --img_h 256 --img_w 256 --train_batch_size 4 --dataset RICE2 --total_epoches 100 --save_start_epoch 95 --device cuda:0 --step_gamma 0.9 --step_size 10 --lr 0.0002

python dehazing_SFAN.py --results_dir ../results/RSDehazing/SFAN/LHID/ --img_h 256 --img_w 256 --train_batch_size 4 --dataset LHID --total_epoches 50 --save_start_epoch 45 --device cuda:0 --step_gamma 0.9 --step_size 10 --lr 0.0002
python dehazing_SFAN.py --results_dir ../results/RSDehazing/SFAN/DHID/ --img_h 256 --img_w 256 --train_batch_size 4 --dataset DHID --total_epoches 50 --save_start_epoch 45 --device cuda:0 --step_gamma 0.9 --step_size 10 --lr 0.0002

python dehazing_SFAN.py --results_dir ../results/RSDehazing/SFAN/RSID/ --img_h 256 --img_w 256 --train_batch_size 4 --dataset RSID --total_epoches 100 --save_start_epoch 95 --device cuda:0 --step_gamma 0.9 --step_size 10 --lr 0.0002


