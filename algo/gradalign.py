import random
import torch.optim as optim
from utils.utility_functions.functions import *
from utils.utility_functions.arguments import Arguments
from utils.utility_functions.accumulators import *
from torchvision import datasets, transforms
from utils.data_processing.data_splitter import CIFAR10Transform

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

        last_full_gradient = None

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
                difference = create_zero_list(model)
                for i, param in enumerate(model.parameters()):
                    difference[i] = full_gradient[i] - local_gradient[i]
                variance = compute_norm(difference)
                norm_variance = compute_norm(difference) / compute_norm(full_gradient)

            # if num_rounds in Arguments.scheduler:
            #     Arguments.beta *= Arguments.lr_decay

            # Before the round starts, we take the displayment step scaled by the beta constant
            for i, param in enumerate(model.parameters()):
                if last_full_gradient is not None:
                    param.data += Arguments.beta * (last_full_gradient[i] - local_gradient[i])
                else:
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

            if last_full_gradient is None:
                last_full_gradient = create_zero_list(model)
            for i, (param_now, param_before) in enumerate(zip(model_at_start.parameters(), model.parameters())):  # 最后在对齐梯度上迭代
                last_full_gradient[i] = param_now.data - param_before.data

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