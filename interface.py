import os
import random

import torch

import wandb
import torch.optim as optim
from torch.multiprocessing import Process
from utils.data_processing.data_splitter import split_data, get_validation_data
from utils.utility_functions.functions import *
from utils.utility_functions.arguments import Arguments
from utils.utility_functions.accumulators import *
from torchvision import datasets, transforms
from utils.data_processing.data_splitter import CIFAR10Transform


def FedProx(rank: int, size: int, dataloaders: list, validation_loader, indices, seeds):
    """Implementation of FedProx. In each round the algorithm samples the active workers
    and performs Arguments.nsteps local updates before averaging the models to complete the round."""

    with torch.cuda.device(get_device(rank, as_str=True)):
        model = define_model()
        model_at_start = define_model()

        if rank == 0 and Arguments.wandb:
            wandb.init(project=Arguments.wandb_exp_name, name=Arguments.algo, config=Arguments.dictionary)
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
                if rank == 0:
                    difference = create_zero_list(model)
                    for i, param in enumerate(model.parameters()):
                        difference[i] = full_gradient[i] - local_gradient[i]
                    log_metric(["Variance", "Normalized_variance"],
                               [compute_norm(difference), compute_norm(difference) / compute_norm(full_gradient)],
                               num_updates, False)

            # The algorithm performs Arguments.nsteps local epochs

            global_model = create_model_param_list(model)
            for step in range(Arguments.nsteps):  #开始本地训练
                try:
                    data, target = train_iterable.next()
                except:
                    train_iterable = iter(dataloader)
                    data, target = train_iterable.next()
                data, target = data.cuda(), target.reshape(-1).cuda()
                optimizer.zero_grad()
                output = model(data)
                proximal_term = create_zero_list(model)
                for i, param in enumerate(model_at_start.parameters()):
                    proximal_term[i] = param.data[i] - global_model[i]

                loss = criterion(output, target) + Arguments.prox_gamma * compute_norm(proximal_term, squared=True)
                acc = accuracy(output, target)
                mean_train_loss.add(loss.item(), weight=len(data))
                mean_train_acc.add(acc.item(), weight=len(data))

                loss.backward()

                if Arguments.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), Arguments.clipping_norm, norm_type=2)

                optimizer.step()
                num_updates += 1

            # After the local updates, the models are averaged among the active workers.
            average_models(model, group)

            model_parameters = list(model.parameters())
            for i, param in enumerate(model_at_start.parameters()):
                param.grad = -1 * (model_parameters[i].data - param.data)
            optimizer_at_start.step()

            replace_model_data(model, model_at_start)

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


def SGD(rank: int, size: int, dataloaders: list, validation_loader, indices, seeds):
    """Implementation of SGD. In each round the algorithm samples the active workers
    and performs Arguments.nsteps local updates before averaging the models to complete the round."""
    print("Training SGD ", rank)

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

            for data, target in dataloader:
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


def FedAvg(rank: int, size: int, dataloaders: list, validation_loader, indices, seeds):
    """Implementation of FedAvg. In each round the algorithm samples the active workers
    and performs Arguments.nsteps local updates before averaging the models to complete the round."""
    print(rank)

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
                if rank == 0:
                    difference = create_zero_list(model)
                    for i, param in enumerate(model.parameters()):
                        difference[i] = full_gradient[i] - local_gradient[i]
                    log_metric(["Variance", "Normalized_variance"],
                               [compute_norm(difference), compute_norm(difference) / compute_norm(full_gradient)],
                               num_updates, False)

            # The algorithm performs Arguments.nsteps local epochs
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

                if Arguments.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), Arguments.clipping_norm, norm_type=2)

                optimizer.step()
                num_updates += 1

            # After the local updates, the models are averaged among the active workers.
            average_models(model, group)

            model_parameters = list(model.parameters())
            for i, param in enumerate(model_at_start.parameters()):
                param.grad = -1 * (model_parameters[i].data - param.data)
            optimizer_at_start.step()

            replace_model_data(model, model_at_start)

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

def SAM(rank: int, size: int, dataloaders: list, validation_loader, indices, seeds: list):
    print("Training SAM with resplit_data->", Arguments.resplit_data, rank)
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
            if Arguments.plot_grad_alignment and rank == 0:
                difference = create_zero_list(model)
                for i, param in enumerate(model.parameters()):
                    difference[i] = full_gradient[i] - local_gradient[i]
                    log_metric(["Variance", "Normalized_variance"],
                               [compute_norm(difference), compute_norm(difference) / compute_norm(full_gradient)],
                               num_updates, True)

            # if num_rounds in Arguments.scheduler:
            #     Arguments.beta *= Arguments.lr_decay

            # Before the round starts, we take the displayment step scaled by the beta constant
            for i, param in enumerate(model.parameters()):
                # param.data += Arguments.beta * (local_gradient[i]) #SAM
                param.data += Arguments.beta * (local_gradient[i])  # SAM

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
            average_models(model, group)

            model_parameters = list(model.parameters())
            for i, param in enumerate(model_at_start.parameters()):  # 最后在对齐梯度上迭代
                param.grad = -1 * (model_parameters[i].data - param.data)
            optimizer_at_start.step()

            replace_model_data(model, model_at_start)

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

def GradAlign(rank: int, size: int, dataloaders: list, validation_loader, indices, seeds: list):
    """Implementation of the FedGA algorithm. GradAlign is the particular case when Arguments.nsteps=1.
    The algorithm computes the drift correction and scales it to apply a correction before starting.
    Then it performs Arguments.nsteps local steps before averaging the models to conlcude one round."""
    print("Training GradAlign with resplit_data->", Arguments.resplit_data, rank)
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
            if Arguments.plot_grad_alignment and rank == 0:
                difference = create_zero_list(model)
                for i, param in enumerate(model.parameters()):
                    difference[i] = full_gradient[i] - local_gradient[i]
                    log_metric(["Variance", "Normalized_variance"],
                               [compute_norm(difference), compute_norm(difference) / compute_norm(full_gradient)],
                               num_updates, True)

            # if num_rounds in Arguments.scheduler:
            #     Arguments.beta *= Arguments.lr_decay

            # Before the round starts, we take the displayment step scaled by the beta constant
            for i, param in enumerate(model.parameters()):
                # param.data += Arguments.beta * (local_gradient[i]) #SAM
                param.data += Arguments.beta * (full_gradient[i] - local_gradient[i])

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
            average_models(model, group)

            model_parameters = list(model.parameters())
            for i, param in enumerate(model_at_start.parameters()):  # 最后在对齐梯度上迭代
                param.grad = -1 * (model_parameters[i].data - param.data)
            optimizer_at_start.step()

            replace_model_data(model, model_at_start)

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

def OralAlign(rank: int, size: int, dataloaders: list, validation_loader, indices, seeds: list):
    """Implementation of the FedGA algorithm. GradAlign is the particular case when Arguments.nsteps=1.
    The algorithm computes the drift correction and scales it to apply a correction before starting.
    Then it performs Arguments.nsteps local steps before averaging the models to conlcude one round."""
    print("Training OralAlign with resplit_data->", Arguments.resplit_data, rank)
    with torch.cuda.device(get_device(rank, as_str=True)):
        model = define_model()
        model_at_start = define_model()
        beta = Arguments.beta
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
            if Arguments.plot_grad_alignment and rank == 0:
                difference = create_zero_list(model)
                for i, param in enumerate(model.parameters()):
                    difference[i] = full_gradient[i] - local_gradient[i]
                    log_metric(["Variance", "Normalized_variance"],
                               [compute_norm(difference), compute_norm(difference) / compute_norm(full_gradient)],
                               num_updates, False)

            # Before the round starts, we take the displayment step scaled by the beta constant
            # 将列表转换成一维 tensor (flatten)
            full_grad = torch.cat([torch.flatten(grad) for grad in full_gradient])  # 将 full_gradient 转为一个 flat vector
            local_grad = torch.cat([torch.flatten(grad) for grad in local_gradient])  # 将 local_gradient 转为一个 flat vector
            align_grad = full_grad - local_grad  # 计算对齐梯度
            if Arguments.plot_grad_alignment and rank == 0:
                log_metric(["Inner Product"],
                           [torch.dot(align_grad, local_grad)],
                           num_updates, False)
            local_grad_norm = torch.norm(local_grad)  # 计算 local_gradient[i] 的范数
            parallel_part = torch.dot(align_grad, local_grad) / (local_grad_norm ** 2) * local_grad
            oral_grad = align_grad - parallel_part

            # 将 oral_grad 分解回原模型参数的形状，并更新梯度
            oral_grad_split = torch.split(oral_grad, [grad.numel() for grad in full_gradient])
            parallel_part_split = torch.split(parallel_part, [grad.numel() for grad in full_gradient])
            inner = torch.dot(align_grad, local_grad)
            ou = -1.0 if inner >= 0 else 1.0
            # 使用 oral_grad 更新模型的梯度
            for i, param in enumerate(model.parameters()):
                param.data += beta * parallel_part_split[i].view_as(param.data)
                param.data -= beta * oral_grad_split[i].view_as(param.data)

            # if num_rounds in Arguments.scheduler:
            Arguments.beta *= Arguments.lr_decay

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
            average_models(model, group)

            model_parameters = list(model.parameters())
            for i, param in enumerate(model_at_start.parameters()):  # 最后在对齐梯度上迭代
                param.grad = -1 * (model_parameters[i].data - param.data)
            optimizer_at_start.step()

            replace_model_data(model, model_at_start)

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
    "FedAvg": FedAvg,
    "SAM": SAM
}

if __name__ == "__main__":
    construct_and_train()
