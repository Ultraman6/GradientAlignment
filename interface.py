import os

import numpy as np
import wandb
import torch.optim as optim
from torch.multiprocessing import Process
from utils.data_processing.data_splitter import split_data, get_validation_data
from utils.utility_functions.functions import *
from utils.utility_functions.arguments import Arguments


def FedAvg(rank:int, size:int, dataloaders:list, indices):
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
            # Optional argument to log the gardient Variance
            if Arguments.plot_grad_alignment:
                local_gradient, full_gradient = compute_full_gradient(model, worker_index, criterion, dataloaders,
                                                                      group)
                if rank == 0:
                    difference = create_zero_list(model)
                    for i, param in enumerate(model.parameters()):
                        difference[i] = full_gradient[i] - local_gradient[i]
                    log_metric(["Variance"], [compute_norm(difference)], num_updates, False)

            train_iterable = iter(dataloaders[worker_index])
            # The algorithm performs Arguments.nsteps local updates
            for step in range(Arguments.nsteps):
                try:
                    data, target = train_iterable.next()
                except:
                    train_iterable = iter(dataloaders[worker_index])

                    data, target = train_iterable.next()
                data, target = data.cuda(),target.cuda()
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
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

            if counter % Arguments.nlogging_steps == 0:
                if check_validation_and_stopping(counter, model, num_updates, num_rounds, rank, group):
                    return
        if rank == 0 and Arguments.wandb:
            wandb.finish()



def GradAlign(rank:int, size:int, dataloaders:list, indices):
    """Implementation of the FedGA algorithm. GradAlign is the particular case when Arguments.nsteps=1.
    The algorithm computes the drift correction and scales it to apply a correction before starting.
    Then it performs Arguments.nsteps local steps before averaging the models to conlcude one round."""
    print("Training GradAlign")
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
            # Computation of full gradient and local gradient needed to compute the drift
            local_gradient, full_gradient = compute_full_gradient(model, worker_index, criterion, dataloaders, group)
            # To compute this gradient, there is an additional round of communication
            num_rounds += 1

            # Optional argument to log the gardient Variance
            if Arguments.plot_grad_alignment and rank == 0:
                difference = create_zero_list(model)
                for i, param in enumerate(model.parameters()):
                    difference[i] = full_gradient[i] - local_gradient[i]
                log_metric(["Variance"], [compute_norm(difference)], num_updates, False)

            if num_rounds in Arguments.scheduler:
                Arguments.beta *= Arguments.lr_decay

            train_iterable = iter(dataloaders[worker_index])

            # Before the round starts, we take the displayment spet scaled by the drift and the beta constant
            for i, param in enumerate(model.parameters()):
                param.data -= Arguments.beta * (full_gradient[i] - local_gradient[i])

            # The algorithm performs Arguments.nsteps local updates
            for step in range(Arguments.nsteps):

                try:
                    data, target = train_iterable.next()
                except:
                    train_iterable = iter(dataloaders[worker_index])
                    data, target = train_iterable.next()

                data, target = data.cuda(),target.cuda()
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
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

            if counter % Arguments.nlogging_steps == 0:
                if check_validation_and_stopping(counter, model, num_updates, num_rounds, rank, group):
                    return
        if rank == 0 and Arguments.wandb:
            wandb.finish()


def Scaffold(rank:int, size:int, dataloaders:list, indices:list):
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
            # Computation of the elements needed for the drift correction
            local_gradient, full_gradient = compute_full_gradient(model, worker_index, criterion, dataloaders, group)
            # To compute this gradient, there is an additional round of communication
            num_rounds += 1

            # Optional argument to log the gardient Variance
            if Arguments.plot_grad_alignment and rank == 0:
                difference = create_zero_list(model)
                for i, param in enumerate(model.parameters()):
                    difference[i] = full_gradient[i] - local_gradient[i]
                log_metric(["Variance"], [compute_norm(difference)], num_updates, False)


            train_iterable = iter(dataloaders[worker_index])
            # The algorithm performs Arguments.nsteps local updates
            for step in range(Arguments.nsteps):
                try:
                    data, target = train_iterable.next()
                except:
                    train_iterable = iter(dataloaders[worker_index])
                    data, target = train_iterable.next()

                data, target = data.cuda(),target.cuda()
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()

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

            if counter % Arguments.nlogging_steps == 0:
                if check_validation_and_stopping(counter, model, num_updates, num_rounds, rank, group):
                    return
        if rank == 0 and Arguments.wandb:
            wandb.finish()


def lbsgd(rank:int, size:int, dataloaders:list, indices):
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
            local_gradient, full_gradient = compute_full_gradient(model, worker_index, criterion, dataloaders, group)
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
                if check_validation_and_stopping(counter, model, num_updates, num_rounds, rank, group):
                    return

        if rank == 0 and Arguments.wandb:
            wandb.finish()



def init_processes(rank:int, size:int, fn, dataloaders:list, indices:list, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = Arguments.master_addr
    os.environ['MASTER_PORT'] = Arguments.master_port
    try:
        dist.init_process_group(backend, rank=rank, world_size=size)
    except Exception as e:
        print("Error in init process group ", Arguments.master_addr,"rank=",rank,Arguments.master_port)
        print(str(e))
        return
    fn(rank, size, dataloaders, indices)

def construct_and_train():
    # We partition the data according to the chosen distribution. The parameter Arguments.percentage_iid controls
    # the percent of i.i.d. data in each worker.
    dataloaders =[ torch.utils.data.DataLoader(subset,batch_size=Arguments.batch_size, shuffle= True)
                     for subset in split_data(Arguments.task, Arguments.nsubsets, Arguments.percentage_iid,
                                              use_two_splits=False )]

    # We get a list such that for each process i, the i-th entry of this list contains the least of indices of the
    # clients that will be sampled by this process in each of the rounds, i.e., if the list for the i-th process is
    #[1,4,6,7,9], it means that in the first round it will sample the 1st client, in the second the 4-th, and so on.
    # The distribution is sampled uniformly at random in each round and there is no overlap in each round, i.e., no two
    # procesess sample the same client in the same round.
    workers_to_processes=np.array(
        [ np.random.permutation(Arguments.nsubsets)[:Arguments.nprocesses] for _ in range(Arguments.nrounds)]
    )

    validation_dataset = get_validation_data(Arguments.task)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=100, shuffle=False)
    Arguments.validation_loader = validation_loader

    processes = []
    if Arguments.largeBatchSGD:
        Arguments.dictionary["algorithm"] = Arguments.algorithm = "lbsgd"
        fn = lbsgd
    elif Arguments.grad_align:
        Arguments.dictionary["algorithm"] = Arguments.algorithm = "GradAlign"
        fn = GradAlign
    elif Arguments.scaffold:
        Arguments.dictionary["algorithm"] = Arguments.algorithm = "Scaffold"
        fn = Scaffold
    else:
        Arguments.dictionary["algorithm"] = Arguments.algorithm = 'FedAvg'
        fn = FedAvg

    Arguments.best_accuracy = 0
    Arguments.best_loss = 100000

    for rank in range(Arguments.nprocesses):
        p = Process(
            target=init_processes,
            args=(
                rank,
                Arguments.nprocesses,
                fn,
                dataloaders,
                workers_to_processes[:,rank]
            )
        )
        p.start()
        processes.append(p)
    print("starting...")
    for p in processes:
        p.join()
    print("ending...")


if __name__ == "__main__":
    construct_and_train()
