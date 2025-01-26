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

def DualAlign(rank: int, size: int, dataloaders: list, validation_loader, indices, seeds: list):
    """Implementation of the FedInit algorithm. GradAlign is the particular case when Arguments.nsteps=1.
    The algorithm computes the drift correction and scales it to apply a correction before starting.
    Then it performs Arguments.nsteps local steps before averaging the models to conlcude one round."""
    print("Training DualAlign with resplit_data->", Arguments.resplit_data, rank)
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

        if Arguments.resplit_data:
            dataset = datasets.CIFAR10(root=Arguments.data_home, train=True, download=True,
                                       transform=CIFAR10Transform.train_transform())

        for counter, worker_index in enumerate(indices):
            if Arguments.resplit_data:
                perm = [i for i in range(len(dataset))]
                random.seed(seeds[counter])
                random.shuffle(perm)
                slice_size = len(dataset) // Arguments.nsubsets
                subset = torch.utils.data.Subset(dataset,
                                                 perm[worker_index * slice_size: (worker_index + 1) * slice_size])
                dataloader = torch.utils.data.DataLoader(subset, batch_size=Arguments.batch_size, shuffle=True)
            else:
                dataloader = dataloaders[worker_index]

            mean_train_loss = Mean()
            mean_train_acc = Mean()
            # Computation of full gradient and local gradient needed to compute the drift
            local_gradient, full_gradient = compute_full_gradient(model, worker_index, criterion, dataloader, group)
            # To compute this gradient, there is an additional round of communication
            num_rounds += 1

            # Optional argument to log the gardient Variance
            if Arguments.plot_grad_alignment:
                local_gradient, full_gradient = compute_full_gradient(model, worker_index, criterion, dataloader,
                                                                      group)
                difference = create_zero_list(model)
                for i, param in enumerate(model.parameters()):
                    difference[i] = full_gradient[i] - local_gradient[i]
                variance = compute_norm(difference)
                norm_variance = compute_norm(difference) / compute_norm(full_gradient)
                # cossim = cosine_similarity(full_gradient, local_gradient, False)

            # Before the round starts, we take the displayment step scaled by the beta constant
            # 将列表转换成一维 tensor (flatten)
            oral_grad_split, parallel_part_split = orthogonal_decomposition(full_gradient, local_gradient)
            # 使用 oral_grad 更新模型的梯度
            for i, param in enumerate(model.parameters()):
                # local_grad = torch.flatten(local_gradient[i])
                # full_grad = torch.flatten(full_gradient[i])
                # align_grad = torch.flatten(full_gradient[i] - local_gradient[i])
                para_grad = parallel_part_split[i].view_as(param.data)
                oral_grad = oral_grad_split[i].view_as(param.data)
                # sign_para = torch.sign(local_grad * parallel_part_split[i])
                # sign_oral = torch.sign(local_grad * oral_grad_split[i])
                # sign_align = torch.sign(local_grad * align_grad)
                # para_values, para_counts = torch.unique(sign_para, return_counts=True)
                # oral_values, oral_counts = torch.unique(sign_oral, return_counts=True)
                # print(f"para: {para_values[para_counts.argmax()]} "
                #       f"oral: {oral_values[oral_counts.argmax()]}")
                # param.data -= Arguments.beta * torch.mul(para_grad, sign_para.view_as(param.data))
                # param.data -= Arguments.beta * torch.mul(oral_grad, sign_oral.view_as(param.data))
                # param.data -= Arguments.beta * torch.mul(align_grad.view_as(param.data), sign_align)
                param.data += Arguments.beta * para_grad
                param.data -= Arguments.beta * oral_grad

            # Before the round starts, we take the displayment step scaled by the beta constant

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

            # After the local updates, the models are averaged among the active workers.
            local_model = copy.deepcopy(model)
            average_models(model, group)
            local_grad = create_zero_list(model)
            full_grad = create_zero_list(model)

            for param_now, param_local, param_before in zip(model.parameters(), local_model.parameters(), model_at_start.parameters()):  # 最后在对齐梯度上迭代
                local_grad[i] = param_local.data - param_before.data
                full_grad[i] = param_now.data - param_before.data
                param_before.grad = -1 * (param_now.data - param_before.data)
            optimizer_at_start.step()

            # nesterov momentum federated
            global_grad = create_zero_list(model)
            for i, param in enumerate(model.parameters()): # 全局更新梯度
                global_grad[i] = full_grad[i] - local_grad[i]

            exp_local_grad = compute_local_gradient(model, worker_index, criterion, dataloader, group)
            oral_grad_split, parallel_part_split = orthogonal_decomposition_sim(exp_local_grad, global_grad)
            # 使用 oral_grad 更新模型的梯度
            for i, param in enumerate(model_at_start.parameters()):
                param.data += Arguments.beta * parallel_part_split[i].view_as(param.data)
                param.data -= Arguments.beta * oral_grad_split[i].view_as(param.data)

            # for i, param in enumerate(model_at_start.parameters()):
            #     param.data += Arguments.beta * (full_grad[i] - local_grad[i])

            # if num_rounds in Arguments.scheduler:
            #     Arguments.beta *= Arguments.lr_decay

            replace_model_data(model, model_at_start)

            scheduler.step()
            num_rounds += 1

            train_stats = [torch.tensor(100 * mean_train_acc.value()), torch.tensor(mean_train_loss.value())]
            metric_stats = [torch.tensor(variance), torch.tensor(norm_variance)]
            average_lists(train_stats, group)
            average_lists(metric_stats, group)
            if rank == 0:
                log_metric(
                    ["Train Accuracy", "Train Loss", "num_updates", "rounds",
                     "Variance", "Normalized_variance"],
                    [train_stats[0].item(), train_stats[1].item(), num_updates, num_rounds,
                     metric_stats[0].item(), metric_stats[1].item()],
                    num_updates
                )

            if counter % Arguments.nlogging_steps == 0:
                if check_validation_and_stopping(counter, model, num_updates, num_rounds, rank, group, validation_loader):
                    return

        if rank == 0 and Arguments.wandb:
            wandb.finish()