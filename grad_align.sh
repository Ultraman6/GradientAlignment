#!/usr/bin/env bash
# 112800 training datapoints
port = 1

for l in 0 1 2; do
python interface.py --batch_size=10 --gpu_ids=[0] --learning_rate=0.3 --nprocesses=4 --nrounds=801 --nsubsets=772 \
 --nlogging_steps=20 --port=19 --nsteps=5 --task=sent140 --scheduler=[100000] --percentage_iid=0.0 --gradient_clipping \
 --clipping_norm=10 --model=rnn_sent140 --wandb_exp_name=Sent140 --weight_decay=0.0001 --plot_grad_alignment \
 --sequence_len=25 --wandb
done

for lr in 0.3; do
    for beta in 0.4 0.2 0.1 0.05; do
python interface.py --batch_size=10 --gpu_ids=[0] --learning_rate=$lr --nprocesses=4 --nrounds=801 --nsubsets=772 \
 --nlogging_steps=20 --port=$port --nsteps=5 --task=sent140 --scheduler=[100000] --percentage_iid=0.0 --gradient_clipping \
 --clipping_norm=10 --model=rnn_sent140 --wandb_exp_name=Sent140 --weight_decay=0.0001 --plot_grad_alignment \
 --sequence_len=25 --wandb --grad_align --beta=$beta

        port=$((port+1))
    done
done
