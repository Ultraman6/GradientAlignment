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

    parser.add_argument("--nsubsets", default=100, type=int, choices=range(1, 1001),
                        help="Total number of clients to be used")
    parser.add_argument("--nprocesses", default=1, type=int, choices=range(1, 48),
                        help="number of processes, i.e., number of clients sampled in each round")
    parser.add_argument("--batch_size", default=50, type=int, choices=range(1, 5001),
                        help="batch size of every process")
    parser.add_argument("--nsteps", default=5, type=int, choices=range(0, 2001), help="number of local steps per round")
    parser.add_argument("--nrounds", default=1001, type=int, choices=range(1, 400002), help="number of rounds")
    parser.add_argument("--nlogging_steps", default=100, type=int, choices=range(1, 1001),
                        help="number of rounds for logging, i.e., there is logging whenever round % nlogging_steps == 0")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="learning rate for each of the local steps")
    parser.add_argument("--scaling", action='store_true', help="Use scaling of the gradients per layer")
    parser.add_argument("--snapshot", action='store_true', help="Use scaling of the gradients per layer")


    parser.add_argument("--task", default="MNIST", type=str, choices={"MNIST", "EMNIST", "CIFAR10", "CIFAR100", "shakespeare", "sent140"}, help="Dataset to be used")
    parser.add_argument("--model", default="cnn_mnist", choices={"cnn_mnist", "cnn_cifar", "resnet18","rnn_shakespeare", "rnn_sent140"}, type=str, help="model to be used")

    parser.add_argument('--gradient_clipping', action='store_true', help="Clip gradients of workers")
    parser.add_argument("--clipping_norm", default=1.0, type=float,
                        help="max magnitud of L_2 norm when using gradient clipping with full gradient")
    parser.add_argument("--target_acc", default=100.0, type=float, help="Target Accuracy of this run, it stops when reaching this accuracy if the parameter is set")


    parser.add_argument('--wandb', action='store_true', help="Use weights and biases logging system. You need an account and to be logged in for this to work.")
    parser.add_argument("--wandb_exp_name", default='random_experiments', type=str, help="Experiment name in wandb")

    parser.add_argument("--load_seed", type=int, default=-1, help="Used to set the random seed of the algorithm.")

    parser.add_argument("--scheduler", default="[1000000000]", type=str, help="scheduler decreases the learning rates when the the round is in the given list")
    parser.add_argument("--lr_decay", default=0.1, type=float, help="Learning rate decay")

    parser.add_argument("--port", default=21, type=int, choices=range(0, 100), help="final digit of port number for the distributed process. No two processes with the same port can run simultaneously. ")

    parser.add_argument("--gpu_ids", default="[0,1]", type=str, help="a list with the indices of the gpus to run the experiment. Processes will be distributed evenly among them.")
    parser.add_argument("--percentage_iid", default=1.0, type=float,
                        help="percentage of iid data in workers' partition. 1.0 for fully IID data and 0.0 for completely heterogeneous data.")

    parser.add_argument("--weight_decay", default=0.001, type=float, help="Optimizer's weight decay")
    parser.add_argument("--server_momentum", default=0.0, type=float, help="server momentum")
    parser.add_argument("--momentum", default=0.0, type=float, help="local momentum")
    parser.add_argument("--server_learning_rate", default=1.0, type=float, help="server step learning rate, default is 1.0")
    parser.add_argument('--server_nesterov', action='store_true', help="Use Nesterov's server momentum")
    parser.add_argument('--batchnorm', action='store_true',
                        help="Use batch normalization (only available in Resnet Models)")


    parser.add_argument("--beta", default=0.1, type=float, help="Corresponds to the beta constant in the FedGA algorithm")
    parser.add_argument("--prox_gamma", default=0.1, type=float,
                        help="FedProx constant")

    parser.add_argument("--fg_batch_size", default=-1, type=int, help="Full gradient Batch size")

    parser.add_argument("--plot_grad_alignment", action='store_true', help="Set to true to plot the gradient variance after each round.")

    parser.add_argument('--sgd', action='store_true',
                        help="Use SGD algorithm, it is also used by default if no other is set")
    parser.add_argument('--fedavg', action='store_true', help="Use FedAvg algorithm, it is also used by default if no other is set")
    parser.add_argument("--grad_align", action='store_true', help="Use the FedGA algorithm")
    parser.add_argument("--fedprox", action='store_true', help="Use the FedProx algorithm")
    parser.add_argument("--scaffold", action='store_true', help="Use SCAFFOLD algorithm.")
    parser.add_argument('--largeBatchSGD', action='store_true', help="Use largeBatch SGD")

    parser.add_argument("--sequence_len", type=int, default=80, help="Used to set the random seed of the algorithm.")
    parser.add_argument("--single_loss", action='store_true', help="Use just the loss of the last char in shakespeare.")
    parser.add_argument('--resplit_data', action='store_true', help="reSplit data in each round")

    args = vars(parser.parse_args())
    if ((args['percentage_iid'] == 0) or (args['percentage_iid'] == 1)):
        args['batch_size_iid'] = 0
    args['master_addr'] = '127.0.0.' + str(args['port'])
    args['master_port'] = str(295 + args['port'])
    args['ngpus'] = 1 + args['gpu_ids'].count(",")
    os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu_ids'].replace(" ", "").replace("[", "").replace("]", "")

    temp = args["scheduler"].replace(" ", "").replace("[", "").replace("]", "").split(",")
    args["scheduler"] = [int(x) for x in temp]

    args["seed"] = random.randint(1, 100000) if args["load_seed"] == -1 else args["load_seed"]

    args['data_home'] = '../data'

    return args

class Arguments():
    dictionary = None
    dictionary = parse_arguments()
    set_variable_from_dict(dictionary)

    @classmethod
    def to_string(cls):
        return str(cls.dictionary)
