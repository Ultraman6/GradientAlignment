import copy
import os
import random
import warnings

import torch

import wandb
import torch.optim as optim
from torch.multiprocessing import Process
from utils.data_processing.data_splitter import split_data, get_validation_data
from utils.optimizer.sam import sam
from utils.utility_functions.functions import *
from utils.utility_functions.arguments import Arguments
from utils.utility_functions.accumulators import *
from torchvision import datasets, transforms
from utils.data_processing.data_splitter import CIFAR10Transform

def lbsgd(rank: int, size: int, dataloaders: list, validation_loader, indices, seeds):
    """ Implementation of Large Batch SGD. On each round, the algorithm computes the full gradient of each worker,
    averages it and use this average for the update. """
    print("Training lbsgd")
    with torch.cuda.device(get_device(rank, as_str=True)):
        model = define_model()

        if rank == 0 and Arguments.wandb:
            wandb.init(project=Arguments.wandb_exp_name, name=Arguments.to_string(), config=Arguments.dictionary)
            wandb.watch(model)

        group = dist.new_group(range(size))

        optimizer = optim.SGD(
            model.parameters(),
            lr=Arguments.learning_rate,
            weight_decay=Arguments.weight_decay,
            momentum=Arguments.server_momentum,
            nesterov=Arguments.server_nesterov
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=Arguments.scheduler,
            gamma=Arguments.lr_decay,
        )

        num_updates = 0
        num_rounds = 0

        criterion = torch.nn.functional.cross_entropy

        for counter, worker_index in enumerate(indices):
            # Computation of the full gradient by averaging the full local gradient among all active workers
            dataloader = dataloaders[worker_index]
            local_gradient, full_gradient = compute_full_gradient(model, worker_index, criterion, dataloader, group)
            optimizer.zero_grad()

            for i, param in enumerate(model.parameters()):
                param.grad = full_gradient[i].clone()

            if Arguments.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Arguments.clipping_norm, norm_type=2)

            optimizer.step()
            num_updates += 1

            scheduler.step()
            num_rounds += 1

            if counter % Arguments.nlogging_steps == 0:
                if check_validation_and_stopping(counter, model, num_updates, num_rounds, rank, group, validation_loader):
                    return

        if rank == 0 and Arguments.wandb:
            wandb.finish()