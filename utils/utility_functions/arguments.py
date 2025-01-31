import argparse
import ast
from argparse import ArgumentParser
import random
import sys
import os

def set_variable_from_dict(dictionary):
    frame = sys._getframe(1)
    locals = frame.f_locals
    for name in dictionary:
        print(name, " => ", dictionary[name])
        locals[name] = dictionary[name]

def parse_arguments():
    parser = ArgumentParser(description="PyTorch experiments")

    parser.add_argument('--algorithm', default='FedInit',
                        choices=["lbsgd", "GradAlign", "OralAlign", "Scaffold", "FedProx", "SGA", "SGD",
                                 "FedAvg", "SAM", "FedSAM", "SAMAlign", "FedAware", "DSGD", "FedInit" "DualAlign"])
    parser.add_argument("--learning_rate", default=0.01, type=float, help="learning rate for each of the local steps")
    parser.add_argument("--beta", default=0.1, type=float,
                        help="Corresponds to the beta constant in the FedGA algorithm")
    parser.add_argument("--rho", default=0.05, type=float,
                        help="Corresponds to the beta constant in the FedGA algorithm")
    parser.add_argument("--weight_decay", default=0.001, type=float, help="Optimizer's weight decay")

    parser.add_argument("--server_momentum", default=0.0, type=float, help="server momentum")
    parser.add_argument("--momentum", default=0.0, type=float, help="local momentum")
    parser.add_argument("--server_learning_rate", default=1.0, type=float,
                        help="server step learning rate, default is 1.0")

    parser.add_argument("--nprocesses", default=10, type=int,
                        help="客户数量")
    parser.add_argument("--nsubsets", default=100, type=int,
                        help="客户样本量")
    parser.add_argument("--batch_size", default=50, type=int,
                        help="客户样本批量大小")
    parser.add_argument("--nsteps", default=5, type=int, help="本地更新轮次（nprocesses）级别")
    parser.add_argument("--nrounds", default=101, type=int, help="number of rounds")
    parser.add_argument("--nlogging_steps", default=100, type=int,
                        help="number of rounds for logging, i.e., there is logging whenever round % nlogging_steps == 0")
    parser.add_argument("--scaling", action='store_true', help="Use scaling of the gradients per layer")
    parser.add_argument("--snapshot", action='store_true', help="Use scaling of the gradients per layer")

    parser.add_argument("--task", default="MNIST", type=str, choices={"MNIST", "EMNIST", "CIFAR10", "CIFAR100", "shakespeare", "sent140"}, help="Dataset to be used")
    parser.add_argument("--model", default="cnn_mnist", choices={"cnn_mnist", "cnn_cifar", "resnet18","rnn_shakespeare", "rnn_sent140"}, type=str, help="model to be used")

    parser.add_argument('--gradient_clipping', action='store_true', help="Clip gradients of workers")
    parser.add_argument("--clipping_norm", default=1.0, type=float,
                        help="max magnitud of L_2 norm when using gradient clipping with full gradient")
    parser.add_argument("--target_acc", default=100.0, type=float, help="Target Accuracy of this run, it stops when reaching this accuracy if the parameter is set")

    parser.add_argument('--wandb', default=True, help="Use weights and biases logging system. You need an account and to be logged in for this to work.")

    parser.add_argument("--wandb_exp_name", default='test for server look ahead', type=str, help="Experiment name in wandb")
    # 'variance via orthogonal' 'element-wise for oral_align niid'

    parser.add_argument("--load_seed", type=int, default=-1, help="Used to set the random seed of the algorithm.")

    parser.add_argument("--scheduler", default="[1000000000]", type=str, help="scheduler decreases the learning rates when the the round is in the given list")

    parser.add_argument("--lr_decay", default=0.1, type=float, help="Learning rate decay")

    parser.add_argument("--port", default=21, type=int, help="final digit of port number for the distributed process. No two processes with the same port can run simultaneously. ")

    parser.add_argument("--gpu_ids", default="[0]", type=str, help="a list with the indices of the gpus to run the experiment. Processes will be distributed evenly among them.")
    parser.add_argument("--percentage_iid", default=1, type=float,
                        help="percentage of iid data in workers' partition. 1.0 for fully IID data and 0.0 for completely heterogeneous data.")
    parser.add_argument('--server_nesterov', action='store_true', help="Use Nesterov's server momentum")
    parser.add_argument('--batchnorm', action='store_true',
                        help="Use batch normalization (only available in Resnet Models)")

    parser.add_argument("--prox_gamma", default=0.1, type=float,
                        help="FedProx constant")

    parser.add_argument("--fg_batch_size", default=-1, type=int, help="Full gradient Batch size")

    parser.add_argument("--plot_grad_alignment", default=True, help="Set to true to plot the gradient variance after each round.")

    parser.add_argument("--sequence_len", type=int, default=80, help="Used to set the random seed of the algorithm.")
    parser.add_argument("--single_loss", default=False, help="Use just the loss of the last char in shakespeare.")
    parser.add_argument('--resplit_data', default=False, help="reSplit data in each round")

    parser.add_argument('--batch_size_iid', default=None, help="reSplit data in each round")
    parser.add_argument('--master_addr', default='localhost', help="reSplit data in each round")
    parser.add_argument('--master_port', default=12345, help="reSplit data in each round")
    parser.add_argument('--ngpus', default=None, help="reSplit data in each round")
    parser.add_argument('--seed', default=None, help="reSplit data in each round")
    parser.add_argument('--data_home', default='E:\data', help="reSplit data in each round")

    parser.add_argument('--validation_loader', default=None, help="reSplit data in each round")

    parser.add_argument('--best_accuracy', default=1.0, help="reSplit data in each round")
    parser.add_argument('--best_loss', default=0.0, help="reSplit data in each round")

    args = parser.parse_args()
    if ((args.percentage_iid == 0) or (args.percentage_iid == 1)):
        args.batch_size_iid = 0
    args.ngpus = 1 + args.gpu_ids.count(",")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids.replace(" ", "").replace("[", "").replace("]", "")

    args.seed = random.randint(1, 100000) if args.load_seed == -1 else args.load_seed

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids.replace(" ", "").replace("[", "").replace("]", "")

    dictor = vars(args)
    temp = args.scheduler.replace(" ", "").replace("[", "").replace("]", "").split(",")
    dictor["scheduler"] = [int(x) for x in temp]

    return dictor



class Arguments():
    dictionary = None
    dictionary = parse_arguments()
    set_variable_from_dict(dictionary)

    @classmethod
    def to_string(cls):
        return str(cls.dictionary)
