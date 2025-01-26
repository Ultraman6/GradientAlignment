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

def SGA(rank: int, size: int, dataloaders: list, validation_loader, indices, seeds):
    """Implementation of SGA. In each round the algorithm samples the active workers
    and performs Arguments.nsteps local updates before averaging the models to complete the round."""
    print("Training SGA ", rank)

    with torch.cuda.device(get_device(rank, as_str=True)):
        model = define_model()

        group = dist.new_group(range(size))

        if rank == 0 and Arguments.wandb:
            wandb.init(project=Arguments.wandb_exp_name, name=Arguments.to_string(), config=Arguments.dictionary)
            wandb.watch(model)

        optimizer = optim.SGD(
            model.parameters(),
            lr=Arguments.learning_rate,
            momentum=Arguments.momentum,
            weight_decay=Arguments.weight_decay
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
            mean_train_loss = Mean()
            mean_train_acc = Mean()
            dataloader = dataloaders[worker_index]

            # if counter in Arguments.scheduler:
            #     Arguments.beta *= Arguments.lr_decay

            for data, target in dataloader:
                data, target = data.cuda(), target.reshape(-1).cuda()

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                local_gradient = get_gradient(model.parameters())

                _, full_gradient = compute_full_gradient(model, worker_index, criterion, dataloader, group)

                # Before the round starts, we take the displacement step scaled by the beta constant
                for i, param in enumerate(model.parameters()):
                    param.data -= Arguments.beta * (full_gradient[i] - local_gradient[i])

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                acc = accuracy(output, target)
                loss.backward()

                mean_train_loss.add(loss.item(), weight=len(data))
                mean_train_acc.add(acc.item(), weight=len(data))

                for i, param in enumerate(model.parameters()):
                    param.data += Arguments.beta * (full_gradient[i] - local_gradient[i])

                if Arguments.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), Arguments.clipping_norm, norm_type=2)

                optimizer.step()
                num_updates += 1

            scheduler.step()
            num_rounds += 1

            train_stats = [torch.tensor(100 * mean_train_acc.value()), torch.tensor(mean_train_loss.value())]
            if rank == 0:
                log_metric(
                    ["Train Accuracy", "Train Loss", "num_updates", "rounds"],
                    [train_stats[0].item(), train_stats[1].item(), num_updates, num_rounds],
                    num_updates
                )

            if counter % Arguments.nlogging_steps == 0:
                if check_validation_and_stopping(counter, model, num_updates, num_rounds, rank, group, validation_loader):
                    return
        if rank == 0 and Arguments.wandb:
            wandb.finish()