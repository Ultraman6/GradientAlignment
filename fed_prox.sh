# 112800 training datapoints
port_ = 0
for lr in 0.1 0.05; do
    for prox in 0.01 0.1 1.0; do
            for sm in 0.0; do
                python interface.py --batch_size=240 --gpu_ids=[0] --learning_rate=$lr --model=cnn_mnist --nprocesses=10 \
    --nrounds=5001 --nsubsets=47 --nlogging_steps=20 --port=$((80+port_)) --server_momentum=$sm --nsteps=10 --task=EMNIST \
    --scheduler=[100000] --wandb_exp_name=fed_test_emnist --wandb --weight_decay=0.001 --percentage_iid=0.0 \
    --gradient_clipping --clipping_norm=10 --model=cnn_mnist --fedprox --prox_gamma=$prox
                port_=$((port_+1))
        done
    done
done