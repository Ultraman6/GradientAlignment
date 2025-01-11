# 假设你的训练函数已经在 interface.py 中定义，且接口类似于以下
from interface import construct_and_train  # 假设 train_model 是你的训练函数

# 这里用字典来代替命令行参数的传递
def run_training_with_params(params):
    try:
        # 调用你原本在 interface.py 中定义的训练函数，传递参数字典
        construct_and_train(**params)
        print(f"Training successfully completed with params: {params}")
    except Exception as e:
        print(f"Error during training with params: {params}\n{e}")

# 第一个循环 (l = 0, 1, 2)
for l in range(3):  # l takes values 0, 1, 2
    params = {
        "batch_size": 10,
        "gpu_ids": [0],
        "learning_rate": 0.3,
        "nprocesses": 4,
        "nrounds": 801,
        "nsubsets": 772,
        "nlogging_steps": 20,
        "port": 19,
        "nsteps": 5,
        "task": "sent140",
        "scheduler": [100000],
        "percentage_iid": 0.0,
        "gradient_clipping": True,
        "clipping_norm": 10,
        "model": "rnn_sent140",
        "wandb_exp_name": "Sent140",
        "weight_decay": 0.0001,
        "plot_grad_alignment": True,
        "sequence_len": 25,
        "wandb": True
    }
    run_training_with_params(params)

# 第二个循环 (lr 和 beta 的值变化)
port = 1  # 初始端口号

for lr in [0.3]:  # lr takes values 0.3
    for beta in [0.4, 0.2, 0.1, 0.05]:  # beta takes different values
        params = {
            "batch_size": 10,
            "gpu_ids": [0],
            "learning_rate": lr,
            "nprocesses": 4,
            "nrounds": 801,
            "nsubsets": 772,
            "nlogging_steps": 20,
            "port": port,
            "nsteps": 5,
            "task": "sent140",
            "scheduler": [100000],
            "percentage_iid": 0.0,
            "gradient_clipping": True,
            "clipping_norm": 10,
            "model": "rnn_sent140",
            "wandb_exp_name": "Sent140",
            "weight_decay": 0.0001,
            "plot_grad_alignment": True,
            "sequence_len": 25,
            "wandb": True,
            "grad_align": True,
            "beta": beta
        }
        run_training_with_params(params)
        port += 1  # 增加端口号
