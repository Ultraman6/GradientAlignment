
import torch.optim as optim
from utils.optimizer.sam import sam
from utils.utility_functions.functions import *
from utils.utility_functions.arguments import Arguments
from utils.utility_functions.accumulators import *

def SAMAlign(rank: int, size: int, dataloaders: list, validation_loader, indices, seeds):
    """Implementation of FedAvg. In each round the algorithm samples the active workers
    and performs Arguments.nsteps local updates before averaging the models to complete the round."""
    print("Training SAMAlign with resplit_data->", Arguments.resplit_data, rank)

    with ((torch.cuda.device(get_device(rank, as_str=True)))):
        model = define_model()
        model_at_start = define_model()

        if rank == 0 and Arguments.wandb:
            wandb.init(project=Arguments.wandb_exp_name, name=Arguments.to_string(), config=Arguments.dictionary)
            wandb.watch(model)

        group = dist.new_group(range(size))
        sam_optimizer = sam(
            model.parameters(),
            optim.SGD,
            rho=Arguments.rho,
            lr=Arguments.learning_rate,
            momentum=Arguments.momentum,
            weight_decay=Arguments.weight_decay
        )
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
                local_gradient1, full_gradient1, local_gradient2, full_gradient2 = compute_full_sam_gradient(model, worker_index, sam_optimizer, criterion, dataloader, group)

                difference11 = create_zero_list(model)
                difference22 = copy.deepcopy(difference11)
                for i, param in enumerate(model.parameters()):
                    difference11[i] = full_gradient1[i] - local_gradient1[i]
                    difference22[i] = full_gradient1[i] - local_gradient1[i]
                    # log_metric(["Variance11", "Normalized_variance11"],
                    #            [compute_norm(difference11), compute_norm(difference11) / compute_norm(full_gradient1)],
                    #            num_updates, False)
                    # log_metric(["Variance22", "Normalized_variance22"],
                    #            [compute_norm(difference22), compute_norm(difference22) / compute_norm(full_gradient2)],
                    #            num_updates, False)

                # 本地-全局方差对齐
                oral_grad_split12, parallel_part_split12 = 222(difference11, difference22)
                oral_grad_split21, parallel_part_split21 = orthogonal_decomposition_sim(difference22, difference11)
                oral_grad_split1, parallel_part_split1 = orthogonal_decomposition_sim(oral_grad_split12, local_gradient1)
                oral_grad_split2, parallel_part_split2 = orthogonal_decomposition_sim(oral_grad_split21, local_gradient2)
                for i, param in enumerate(model.parameters()):
                    param.data += Arguments.beta * parallel_part_split1[i].view_as(param.data)
                    param.data -= Arguments.beta * oral_grad_split1[i].view_as(param.data)
                    # param.data += Arguments.beta * parallel_part_split2[i].view_as(param.data)
                    # param.data -= Arguments.beta * oral_grad_split2[i].view_as(param.data)

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