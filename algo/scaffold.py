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

def Scaffold(rank: int, size: int, dataloaders: list, validation_loader, indices: list, seeds):
    """Implementation of Scaffold. On each round, from the sampled workers the drift correction is computed and is
    applied on each of the local updates"""
    print("Training Scaffold")
    with torch.cuda.device(get_device(rank, as_str=True)):
        model = define_model()
        model_at_start = define_model()

        if rank == 0 and Arguments.wandb:
            wandb.init(project=Arguments.wandb_exp_name, name=Arguments.to_string(), config=Arguments.dictionary)
            wandb.watch(model)

        group = dist.new_group(range(size))

        optimizer = optim.SGD(
            model.parameters(),
            lr=Arguments.learning_rate,
            momentum=Arguments.momentum,
            weight_decay=Arguments.weight_decay
        )
        optimizer_at_start = optim.SGD(
            model_at_start.parameters(),
            lr=Arguments.server_learning_rate,
            momentum=Arguments.server_momentum,
            nesterov=Arguments.server_nesterov,
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
            # Computation of the elements needed for the drift correction
            dataloader = dataloaders[worker_index]
            local_gradient, full_gradient = compute_full_gradient(model, worker_index, criterion, dataloader, group)
            # To compute this gradient, there is an additional round of communication
            num_rounds += 1

            # Optional argument to log the gardient Variance
            if Arguments.plot_grad_alignment and rank == 0:
                difference = create_zero_list(model)
                for i, param in enumerate(model.parameters()):
                    difference[i] = full_gradient[i] - local_gradient[i]
                log_metric(["Variance", "Normalized_variance"],
                           [compute_norm(difference), compute_norm(difference) / compute_norm(full_gradient)],
                           num_updates, False)

            # The algorithm performs Arguments.nsteps local updates
            for step in range(Arguments.nsteps):
                try:
                    data, target = train_iterable.next()
                except:
                    train_iterable = iter(dataloader)
                    data, target = train_iterable.next()
                data, target = data.cuda(), target.reshape(-1).cuda()
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                acc = accuracy(output, target)
                loss.backward()

                mean_train_loss.add(loss.item(), weight=len(data))
                mean_train_acc.add(acc.item(), weight=len(data))

                # The drift correction is added to the gradient in each of the local updates.
                for i, param in enumerate(model.parameters()):
                    param.grad += (full_gradient[i] - local_gradient[i])

                if Arguments.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), Arguments.clipping_norm, norm_type=2)

                optimizer.step()
                num_updates += 1

            average_models(model, group)

            model_parameters = list(model.parameters())
            for i, param in enumerate(model_at_start.parameters()):
                param.grad = -1 * (model_parameters[i].data - param.data)
            optimizer_at_start.step()

            replace_model_data(model, model_at_start)

            scheduler.step()
            scheduler.step()
            num_rounds += 1

            train_stats = [torch.tensor(100 * mean_train_acc.value()), torch.tensor(mean_train_loss.value())]
            average_lists(train_stats, group)
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