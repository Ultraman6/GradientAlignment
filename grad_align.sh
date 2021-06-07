# 112800 training datapoints
port = 0
for lr in 0.2; do
    for nsubsets in 94 188; do
        for sm in 0.0; do
            for g_lr in 0.0 0.2 0.4 0.6 0.8; do
                python interface.py --batch_size=$((112800 / nsubsets)) --gpu_ids=[0,1,2,3,4,5,6,7] --learning_rate=$lr --model=cnn_mnist  \
    --nprocesses=$((10 * (nsubsets / 47))) \
    --nrounds=801 --nsubsets=$nsubsets --nlogging_steps=5 --port=$((51+port)) --server_momentum=$sm --nsteps=1 --task=EMNIST \
    --scheduler=[100000] --wandb_exp_name=fed_test_emnist --wandb --weight_decay=0.001 --percentage_iid=1.0 \
    --gradient_clipping --clipping_norm=10 --model=cnn_mnist --wandb --grad_align --grad_lr=$g_lr
                port=$((port+1))
            done
        done
    done
done
