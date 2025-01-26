import os
import random
import warnings
from torch.multiprocessing import Process

from algo.dsgd import DSGD
from algo.dualalign import DualAlign
from algo.fedavg import FedAvg
from algo.fedaware import FedAware
from algo.fedinit import FedInit
from algo.fedprox import FedProx
from algo.fedsam import FedSAM
from algo.gradalign import GradAlign
from algo.lbsgd import lbsgd
from algo.oralalign import OralAlign
from algo.sam import SAM
from algo.samalign import SAMAlign
from algo.scaffold import Scaffold
from algo.sga import SGA
from algo.sgd import SGD
from utils.data_processing.data_splitter import split_data, get_validation_data
from utils.utility_functions.functions import *
from utils.utility_functions.arguments import Arguments
from utils.utility_functions.accumulators import *



def init_processes(rank: int, size: int, fn, dataloaders: list, validation_loader, indices: list, seeds, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    print(rank, size, backend)
    try:
        dist.init_process_group(backend, rank=rank, world_size=size, init_method="env://?use_libuv=False")
    except Exception as e:
        print("Error in init process group ", Arguments.master_addr,"rank=",rank,Arguments.master_port)
        print(str(e))
        return
    fn(rank, size, dataloaders, validation_loader, indices, seeds)


def construct_and_train():
    # We partition the data according to the chosen distribution. The parameter Arguments.percentage_iid controls
    # the percent of i.i.d. data in each worker.
    dataloaders = [torch.utils.data.DataLoader(subset, batch_size=Arguments.batch_size, shuffle=True)
                   for subset in split_data(Arguments.task, Arguments.nsubsets, Arguments.percentage_iid,
                                            use_two_splits=False)]

    seeds = [random.randint(0, 100000) for _ in range(Arguments.nrounds)]

    # We get a list such that for each process i, the i-th entry of this list contains the least of indices of the
    # clients that will be sampled by this process in each of the rounds, i.e., if the list for the i-th process is
    #[1,4,6,7,9], it means that in the first round it will sample the 1st client, in the second the 4-th, and so on.
    # The distribution is sampled uniformly at random in each round and there is no overlap in each round, i.e., no two
    # procesess sample the same client in the same round.
    workers_to_processes = np.array(
        [np.random.permutation(Arguments.nsubsets)[:Arguments.nprocesses] for _ in range(Arguments.nrounds)]
    )

    validation_dataset = get_validation_data(Arguments.task)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=100, shuffle=False)

    processes = []
    if Arguments.algorithm in algorithm_map:
        fn = algorithm_map[Arguments.algorithm]
    else:
        raise

    for rank in range(Arguments.nprocesses):
        p = Process(
            target=init_processes,
            args=(
                rank,
                Arguments.nprocesses,
                fn,
                dataloaders,
                validation_loader,
                workers_to_processes[:, rank],
                seeds
            )
        )
        p.start()
        processes.append(p)
    print("starting...")
    for p in processes:
        p.join()
    print("ending...")

algorithm_map = {
    "lbsgd": lbsgd,
    "GradAlign": GradAlign,
    "OralAlign": OralAlign,
    "Scaffold": Scaffold,
    "FedProx": FedProx,
    "SGA": SGA,
    "SGD": SGD,
    "D-SGD": DSGD,
    "FedAvg": FedAvg,
    "SAM": SAM,
    "FedSAM": FedSAM,
    "SAMAlign": SAMAlign,
    "FedAware": FedAware,
    "DSGD": DSGD,
    "FedInit": FedInit,
    "DualAlign": DualAlign
}

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, message=".*lr_scheduler.step() before optimizer.step()")
    construct_and_train()
