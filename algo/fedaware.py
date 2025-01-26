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

def FedAware(rank: int, size: int, dataloaders: list, validation_loader, indices, seeds):
    """Implementation of FedAvg. In each round the algorithm samples the active workers
    and performs Arguments.nsteps local updates before averaging the models to complete the round."""
    print("Training FedAware with resplit_data->", Arguments.resplit_data, rank)

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
            mean_train_loss = Mean()
            mean_train_acc = Mean()

            dataloader = dataloaders[worker_index]
            # Optional argument to log the gardient Variance
            if Arguments.plot_grad_alignment:
                local_gradient, full_gradient = compute_full_gradient(model, worker_index, criterion, dataloader,
                                                                      group)
                difference = create_zero_list(model)
                for i, param in enumerate(model.parameters()):
                    difference[i] = full_gradient[i] - local_gradient[i]
                variance = compute_norm(difference)
                norm_variance = compute_norm(difference) / compute_norm(full_gradient)

            # The algorithm performs Arguments.nsteps local epochs
            for step in range(Arguments.nsteps):  # 先每个样本迭代n步（本地epoch）
                try:
                    data, target = next(train_iterable)
                except:
                    train_iterable = iter(dataloader)
                    data, target = next(train_iterable)

                data, target = data.cuda(), target.reshape(-1).cuda()
                optimizer.zero_grad()

                output = model(data)
                loss = criterion(output, target)
                acc = accuracy(output, target)

                loss.backward()

                mean_train_loss.add(loss.item(), weight=len(data))
                mean_train_acc.add(acc.item(), weight=len(data))

                if Arguments.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), Arguments.clipping_norm, norm_type=2)

                optimizer.step()
                num_updates += 1

            # 先获取本地更新梯度
            # 创建一个与模型结构相同的模型，用于存储梯度差异
            local_gradient = list(model.parameters())

            for idx, (param_now, param_before) in enumerate(zip(model.parameters(), model_at_start.parameters())):
                local_gradient[idx] = param_now.data - param_before.data

            # 本地梯度标准化(可选)
            local_norm = compute_norm(local_gradient)
            for i, (param_now, param_before) in enumerate(zip(model.parameters(), model_at_start.parameters())):
                local_gradient[i] /= local_norm  # 梯度标准化
                param_now.data = param_before.data + local_gradient[i]

            avg_local_norm = fedaware_average(local_gradient, group)

            for i, param in enumerate(model_at_start.parameters()):
                param.grad = -1 * local_gradient[i]

            optimizer_at_start.step()
            optimizer_at_start.zero_grad()

            # 梯度多样性测试
            if Arguments.plot_grad_alignment:
                local_ahead_gradient = compute_ahead_gradient(model, worker_index, criterion, dataloader, group)
                full_ahead_gradient = compute_ahead_gradient(model_at_start, worker_index, criterion, dataloader, group)
                ahead_difference = create_zero_list(model)
                for i, param in enumerate(model.parameters()):
                    ahead_difference[i] = local_ahead_gradient[i] - full_ahead_gradient[i]
                diversity = compute_norm(ahead_difference)
                norm_diversity = diversity / compute_norm(full_ahead_gradient)
                div_diversity = avg_local_norm / compute_norm(local_gradient)

            replace_model_data(model, model_at_start)

            scheduler.step()
            num_rounds += 1

            train_stats = [torch.tensor(100 * mean_train_acc.value()), torch.tensor(mean_train_loss.value())]
            metric_stats = [torch.tensor(variance), torch.tensor(norm_variance),
                            torch.tensor(diversity), torch.tensor(norm_diversity), torch.tensor(div_diversity)]
            average_lists(train_stats, group)
            average_lists(metric_stats, group)
            if rank == 0:
                log_metric(
                    ["Train Accuracy", "Train Loss", "num_updates", "rounds",
                     "Variance", "Normalized_variance", "Diversity", "Normalized_Diversity", "Div_Diversity"],
                    [train_stats[0].item(), train_stats[1].item(), num_updates, num_rounds,
                     metric_stats[0].item(), metric_stats[1].item(), metric_stats[2].item(), metric_stats[3].item(), metric_stats[4].item()],
                    num_updates
                )

            if counter % Arguments.nlogging_steps == 0:
                if check_validation_and_stopping(counter, model, num_updates, num_rounds, rank, group, validation_loader):
                    return
        if rank == 0 and Arguments.wandb:
            wandb.finish()