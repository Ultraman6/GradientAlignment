# Implicit Gradient Alignment in Distributed and Federated Learning


**Requirements:**
It requires Python 3 with the installation of torch, torchvision and wandb is recommended.
We recommend to use pip to install them

pip install torch

pip install torchvision 

**Weights and Biases:**
To log the experiments using wandb, a free account of Weights and biases is needed. Please go to https://www.wandb.com to create a free account.
Once your account is created you need to instal wandb.
With pip3 it requires to run:

pip install wandb

Then login in your account by running in the terminal the line:

wandb login WANDB_API_KEY

where WANDB_API_KEY can be found in https://app.wandb.ai/settings after loging in. 

You are done and ready to run experiments.

**Code samples:**

For FedAvg run it like:

python interface.py --batch_size=100 --gpu_ids=[0] --learning_rate=0.2 --model=cnn_cifar --nsubsets=10 --nprocesses=10 --nrounds=2001 --nlogging_steps=10 --port=28 --nsteps=20 --task=CIFAR10 --wandb_exp_name=test_cifar --wandb --weight_decay=0.001 --percentage_iid=1.0

For MiniBatch SGD (largeBatchSGD) run it like:

python interface.py --batch_size=2400 --gpu_ids=[0,1,3,4] --learning_rate=0.2 --model=cnn_mnist --nsubsets=47 --nprocesses=10 --nrounds=2001 --nlogging_steps=5 --port=83--task=EMNIST --wandb_exp_name=test_emnist --wandb --weight_decay=0.001 --percentage_iid=0.0 --largeBatchSGD

For SCAFFOLD run like:

python interface.py --batch_size=2400 --gpu_ids=[0] --learning_rate=0.05 --model=cnn_mnist --nsubsets=47 --nprocesses=10 --nrounds=2001 --nlogging_steps=5 --port=49 --server_momentum=0.0 --nsteps=10 --task=EMNIST --wandb_exp_name=test_emnist --wandb --weight_decay=0.001 --percentage_iid=0.0 --model=cnn_mnist --scaffold

For FedGA run like:

python interface.py --batch_size=2400 --gpu_ids=[0] --learning_rate=0.05 --model=cnn_mnist --nprocesses=10 --nrounds=2001 --nsubsets=47 --nlogging_steps=5 --port=59 --server_momentum=0.0 --nsteps=10 --task=EMNIST --wandb_exp_name=test_emnist --wandb --weight_decay=0.001 --percentage_iid=0.0 --grad_align --beta=0.025

**Possible variables:**

--**nsubsets**, default=100, type=int, choices=range(1, 1001), help="Total number of clients to be used"

--**nprocesses**, default=1, type=int, choices=range(1, 48), help="number of processes, i.e., number of clients sampled in each round"

--**batch_size**, default=50, type=int, choices=range(1, 5001), help="batch size of every process"

--**nsteps**, default=5, type=int, choices=range(0, 2001), help="number of local steps per round"

--**nrounds**, default=1001, type=int, choices=range(1, 400002), help="number of rounds"

--**nlogging_steps**, default=100, type=int, choices=range(1, 1001), help="number of rounds for logging, i.e., there is logging whenever round % nlogging_steps == 0"

--**learning_rate**, default=0.01, type=float, help="learning rate for each of the local steps"

--**task**, default="MNIST, type=str, choices={"MNIST", "EMNIST", "CIFAR10", "CIFAR100"}, help="Dataset to be used"

--**model**, default="cnn_mnist", choices={"cnn_mnist", "cnn_cifar", "resnet18"}, type=str, help="model to be used"

--**gradient_clipping**, boolean, help="Clip gradients of workers"

--**clipping_norm**, default=1.0, type=float,
                    help="max magnitud of L_2 norm when using gradient clipping with full gradient"
--**target_acc**, default=100.0, type=float, help="Target Accuracy of this run, it stops when reaching this accuracy if the parameter is set"

--**wandb**, boolean, help="Use weights and biases logging system. You need an account and to be logged in for this to work."

--**wandb_exp_name**, default='random_experiments', type=str, help="Experiment name in wandb"

--**load_seed, type=int**, default=-1, help="Used to set the random seed of the algorithm."

--**scheduler**, default="[1000000000], type=str, help="scheduler decreases the learning rates when the the round is in the given list"

--**lr_decay**, default=0.1, type=float, help="Learning rate decay"

--**port**, default=21, type=int, choices=range(0, 100), help="final digit of port number for the distributed process. No two processes with the same port can run simultaneously. "

--**gpu_ids**, default="[0,1], type=str, help="a list with the indices of the gpus to run the experiment. Processes will be distributed evenly among them."

--**percentage_iid**, default=1.0, type=float, help="percentage of iid data in workers' partition. 1.0 for fully IID data and 0.0 for completely heterogeneous data."

--**weight_decay**, default=0.001, type=float, help="Optimizer's weight decay"

--**server_momentum**, default=0.0, type=float, help="server momentum"

--**momentum**, default=0.0, type=float, help="local momentum"

--**server_learning_rate**, default=1.0, type=float, help="server step learning rate**, default is 1.0"

--**server_nesterov**, boolean, help="Use Nesterov's server momentum"

--**batchnorm**, boolean, help="Use batch normalization (only available in Resnet Models)"


--**beta**, default=0.1, type=float, help="Corresponds to the beta constant in the FedGA algorithm"

--**fg_batch_size**, default=-1, type=int, help="Full gradient Batch size"

--**plot_grad_alignment**, boolean, help="Set to true to plot the gradient variance after each round."

--**fedavg**, boolean, help="Use FedAvg algorithm, it is also used by default if no other is set"

--**grad_align**, boolean, help="Use the FedGA algorithm"

--**scaffold**, boolean, help="Use SCAFFOLD algorithm."

--**largeBatchSGD**, boolean, help="Use largeBatch SGD"