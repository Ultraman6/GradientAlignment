import math
from typing import List

import torch
import copy
import wandb
from utils.utility_functions.accumulators import *
from utils.model.NN_models import *
from utils.utility_functions.arguments import Arguments
import torch.distributed as dist
import json
import numpy as np

def validation_accuracy(model: torch.nn.Module, cuda_dev, validation_loader):
    model.eval()
    mean_test_accuracy = Mean()
    mean_test_loss = Mean()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_x, batch_y in validation_loader:
            batch_x, batch_y = batch_x.to(cuda_dev), batch_y.reshape(-1).to(cuda_dev)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            acc = accuracy(prediction, batch_y)

            mean_test_loss.add(loss.item(), weight=len(batch_x))
            mean_test_accuracy.add(acc.item(), weight=len(batch_x))
    model.train()
    return mean_test_loss.value(), 100.0 * mean_test_accuracy.value()

def get_device(rank, as_str:bool = True):
    gpu_id = (Arguments.port + rank) % Arguments.ngpus
    device = "cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu"
    if as_str:
        return device
    else:
        return torch.device(device)

def define_model():
    # set the same seed for the creation of all models
    torch.manual_seed(Arguments.seed)
    model = None
    num_classes = 100 if Arguments.task == "CIFAR100" else 10
    num_classes = 47 if Arguments.task == "EMNIST" else num_classes

    if Arguments.model == "resnet18":
        model = ResNet18(num_classes=num_classes)
    # elif Arguments.model == "resnet152":
    #     model = ResNet152(num_classes=num_classes)
    elif Arguments.model == "cnn_mnist":
        model = CNN_MNIST(num_classes=num_classes)
    elif Arguments.model == "cnn_cifar":
        model = CNN_CIFAR10()
    elif Arguments.model == "rnn_shakespeare":
        model = CharRNN(101, hidden_size=100, model="lstm", n_layers=2)
    elif Arguments.model == "rnn_sent140":
        # model = LSTMModel(Arguments.emb_array, hidden_dim=100)
        model = CharRNN(3, hidden_size=100, model="lstm", n_layers=2, word_emb_array=Arguments.emb_array)

    if Arguments.load_seed != -1:
        try:
            model_path = "snapshots/" + Arguments.algorithm + "_" + str(Arguments.seed) + "_model" + ".pt"
            # load existing model
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        except:
            print("Model not found, initializing with given seed")

    model.train()
    model.cuda()
    return model

def orthogonal_decomposition(full_gradient, local_gradient):
    full_grad = torch.cat([torch.flatten(grad) for grad in full_gradient])  # 将 full_gradient 转为一个 flat vector
    local_grad = torch.cat([torch.flatten(grad) for grad in local_gradient])  # 将 local_gradient 转为一个 flat vector
    align_grad = full_grad - local_grad  # 计算对齐梯度
    # if Arguments.plot_grad_alignment and rank == 0:
    #     log_metric(["Inner Product"],
    #                [torch.dot(align_grad, local_grad)],
    #                num_updates, False)
    local_grad_norm = torch.norm(local_grad, 2)  # 计算 local_gradient[i] 的范数
    parallel_part = torch.dot(align_grad, local_grad) / (local_grad_norm ** 2) * local_grad
    oral_grad = align_grad - parallel_part

    # 将 oral_grad 分解回原模型参数的形状，并更新梯度
    oral_grad_split = torch.split(oral_grad, [grad.numel() for grad in full_gradient])
    parallel_part_split = torch.split(parallel_part, [grad.numel() for grad in full_gradient])
    return oral_grad_split, parallel_part_split

def orthogonal_decomposition_sim(full_gradient, local_gradient):
    full_grad = torch.cat([torch.flatten(grad) for grad in full_gradient])  # 将 full_gradient 转为一个 flat vector
    local_grad = torch.cat([torch.flatten(grad) for grad in local_gradient])  # 将 local_gradient 转为一个 flat vector
    local_grad_norm = torch.norm(local_grad, 2)  # 计算 local_gradient[i] 的范数
    parallel_part = torch.dot(full_grad, local_grad) / (local_grad_norm ** 2) * local_grad
    oral_grad = full_grad - parallel_part

    # 将 oral_grad 分解回原模型参数的形状，并更新梯度
    oral_grad_split = torch.split(oral_grad, [grad.numel() for grad in full_gradient])
    parallel_part_split = torch.split(parallel_part, [grad.numel() for grad in full_gradient])
    return oral_grad_split, parallel_part_split

def average_models(model: nn.Module, group):
    """ Model averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.data, op=torch.distributed.ReduceOp.SUM, group=group)
        param.data /= size

def average_models_weighted(model: nn.Module, group, weight: torch.Tensor):
    """
    加权模型平均聚合。

    Args:
        model (nn.Module): 要聚合的模型。
        group: 分布式训练的进程组。
        weight (torch.Tensor): 当前设备的权重，必须是标量张量。
    """
    # 确保weight是标量
    assert weight.dim() == 0, "Weight must be a scalar tensor."
    # 复制当前设备的权重
    local_weight = weight.clone()
    # 计算所有设备的总权重
    dist.all_reduce(local_weight, op=dist.ReduceOp.SUM, group=group)
    total_weight = local_weight.item()
    # 遍历模型的所有参数，进行加权聚合
    for param in model.parameters():
        # 将参数乘以当前设备的权重
        param_weighted = param.data * weight
        # 创建一个临时张量来存储加权后的参数
        param_weighted_sum = torch.zeros_like(param_weighted)
        # 聚合所有设备上的加权参数
        dist.all_reduce(param_weighted_sum, tensor=param_weighted, op=dist.ReduceOp.SUM, group=group)
        # 计算加权平均参数
        param.data.copy_(param_weighted_sum / total_weight)

def fedaware_average(gradient: List[torch.Tensor], group):
    """
    联邦感知的梯度加权平均聚合。
    Args:
        gradient (List[torch.Tensor]): 要更新的模型。
        group: 分布式训练的进程组。
    """

    # Step 1: 计算本地梯度的L2范数
    grad_tensor = torch.cat([torch.flatten(grad) for grad in gradient]).cuda()
    grad_norm = torch.norm(grad_tensor, p=2)
    grad_unit_tensor = grad_tensor / grad_norm

    # Step 2: 收集所有设备的梯度范数
    world_size = dist.get_world_size(group=group)
    all_grad_tensors = [torch.zeros_like(grad_unit_tensor) for _ in range(world_size)]
    dist.all_gather(all_grad_tensors, grad_unit_tensor, group=group)

    sol, val = MinNormSolver.find_min_norm_element_FW(all_grad_tensors)

    # Step 3: 获取当前进程的 rank
    assert sol.sum() - 1 < 1e-5

    weighted_grad = torch.zeros_like(grad_tensor)
    for i in range(world_size):
        weighted_grad += sol[i] * all_grad_tensors[i].cuda()

    # Step 7: 将聚合后的梯度向量转换回列表结构
    pointer = 0
    for grad in gradient:
        num_elements = grad.numel()
        grad_slice = weighted_grad[pointer:pointer + num_elements]
        grad_reshaped = grad_slice.view_as(grad)
        grad.copy_(grad_reshaped)  # 直接在原地更新梯度
        pointer += num_elements

    avg_local_norm = sum([torch.norm(grad, p=2, dim=0).item()**2 * w**2 for grad, w in
                          zip(all_grad_tensors, sol)]) / len(all_grad_tensors)

    return math.sqrt(avg_local_norm)

def compute_full_gradient(model:nn.Module, worker_index:int, criterion, dataloader, group):
    full_gradient = create_zero_list(model)
    worker_full_gradient = create_zero_list(model)
    data_points = 0
    for data, target in dataloader:
        data, target = data.cuda(), target.reshape(-1).cuda()
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        data_points += len(data)
        for i, param in enumerate(model.parameters()):
            if param.grad is not None:
                full_gradient[i] += param.grad.data * len(data)

        if Arguments.fg_batch_size > 0 and data_points >= Arguments.fg_batch_size:
            break

    data_points_wrapper = [torch.tensor([Arguments.nprocesses * data_points]).float()]
    average_lists(data_points_wrapper, group)

    for i, param in enumerate(model.parameters()):
        # param.grad = full_gradient[i].clone() / data_points
        worker_full_gradient[i] = (full_gradient[i].clone() / data_points).data.clone()
        full_gradient[i] *= Arguments.nprocesses / data_points_wrapper[0].item()

    # for i, param in enumerate(model.parameters()):
    #     worker_full_gradient[i] = param.grad.data.clone()

    if float(dist.get_world_size()) > 1:
        average_lists(full_gradient, group)

    model.zero_grad()

    return worker_full_gradient, full_gradient

def compute_local_gradient(model:nn.Module, worker_index:int, criterion, dataloader, group):
    worker_full_gradient = create_zero_list(model)
    data_points = 0
    for data, target in dataloader:
        data, target = data.cuda(), target.reshape(-1).cuda()
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        data_points += len(data)
        for i, param in enumerate(model.parameters()):
            if param.grad is not None:
                worker_full_gradient[i] += param.grad.data * len(data)

        if Arguments.fg_batch_size > 0 and data_points >= Arguments.fg_batch_size:
            break

    for i, param in enumerate(model.parameters()):
        worker_full_gradient[i] = (worker_full_gradient[i].clone() / data_points).data.clone()

    model.zero_grad()

    return worker_full_gradient

def compute_full_sam_gradient(model:nn.Module, worker_index:int, optimizer, criterion, dataloader, group):
    full_gradient_first = create_zero_list(model)
    worker_full_gradient_first = create_zero_list(model)
    full_gradient_second = create_zero_list(model)
    worker_full_gradient_second = create_zero_list(model)
    origin_param_list = create_model_param_list(model)

    data_points = 0
    for data, target in dataloader:
        data, target = data.cuda(), target.reshape(-1).cuda()
        optimizer.zero_grad()
        # 扰动
        output = model(data)
        first_loss = criterion(output, target)
        first_loss.backward(retain_graph=True)
        for i, param in enumerate(model.parameters()):
            if param.grad is not None:
                full_gradient_first[i] += param.grad.data.clone() * len(data)
        optimizer.first_step(zero_grad=True)

        # 更新
        outputs = model(data)
        second_loss = criterion(outputs, target)
        second_loss.backward()
        for i, param in enumerate(model.parameters()):
            if param.grad is not None:
                full_gradient_second[i] += param.grad.data.clone() * len(data)
        optimizer.second_step(zero_grad=True)

        data_points += len(data) # 还原模型参数
        for i, param in enumerate(model.parameters()):
            param.data = origin_param_list[i].clone()

        if Arguments.fg_batch_size > 0 and data_points >= Arguments.fg_batch_size:
            break

    for i, param in enumerate(model.parameters()):
        if param.grad is not None:
            full_gradient_second[i] -= full_gradient_first[i]

    data_points_wrapper = [torch.tensor([Arguments.nprocesses * data_points]).float()]
    average_lists(data_points_wrapper, group)

    for i, param in enumerate(model.parameters()):
        param.grad = full_gradient_first[i].clone() / data_points
        full_gradient_first[i] *= Arguments.nprocesses / data_points_wrapper[0].item()

    for i, param in enumerate(model.parameters()):
        worker_full_gradient_first[i] = param.grad.data.clone()

    if float(dist.get_world_size()) > 1:
        average_lists(worker_full_gradient_first, group)

    for i, param in enumerate(model.parameters()):
        param.grad = full_gradient_second[i].clone() / data_points
        full_gradient_second[i] *= Arguments.nprocesses / data_points_wrapper[0].item()

    for i, param in enumerate(model.parameters()):
        worker_full_gradient_second[i] = param.grad.data.clone()

    if float(dist.get_world_size()) > 1:
        average_lists(worker_full_gradient_second, group)

    model.zero_grad()

    return worker_full_gradient_first, full_gradient_first, worker_full_gradient_second, full_gradient_second

def compute_ahead_gradient(model:nn.Module, worker_index:int, criterion, dataloader, group):
    full_gradient = create_zero_list(model)

    data_points = 0
    for data, target in dataloader:
        data, target = data.cuda(), target.reshape(-1).cuda()
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        data_points += len(data)
        for i, param in enumerate(model.parameters()):
            if param.grad is not None:
                full_gradient[i] += param.grad.data * len(data)

        if Arguments.fg_batch_size > 0 and data_points >= Arguments.fg_batch_size:
            break

    data_points_wrapper = [torch.tensor([Arguments.nprocesses * data_points]).float()]
    average_lists(data_points_wrapper, group)

    for i, param in enumerate(model.parameters()):
        full_gradient[i] *= Arguments.nprocesses / data_points_wrapper[0].item()

    if float(dist.get_world_size()) > 1:
        average_lists(full_gradient, group)

    model.zero_grad()

    return full_gradient

# def compute_ahead_gradient(model:nn.Module, worker_index:int, criterion, dataloader, group):
#     gradient = create_zero_list(model)
#
#     data_points = 0
#     for data, target in dataloader:
#         data, target = data.cuda(), target.reshape(-1).cuda()
#         model.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         data_points += len(data)
#         for i, param in enumerate(model.parameters()):
#             if param.grad is not None:
#                 gradient[i] += param.grad.data * len(data)
#
#         if Arguments.fg_batch_size > 0 and data_points >= Arguments.fg_batch_size:
#             break
#
#     for i, param in enumerate(model.parameters()):
#         gradient[i] /= data_points
#
#     model.zero_grad()
#     return gradient


def average_lists(gradients, group):
    """ List averaging. """
    size = float(dist.get_world_size())
    for key in range(len(gradients)):
        dist.all_reduce(gradients[key], op=torch.distributed.ReduceOp.SUM, group=group)
        gradients[key] /= size


def should_stop(step, accuracy):
    if accuracy > Arguments.target_acc:
        return 1
    return 0


def log_norm_gradient(norm, counter, label:str = "Gradient norm"):
    if not Arguments.log_norms:
        return
    log_metric([label, "num_updates"], [norm, counter], counter, False)


def log_rounds_to_target(round:int, achieved_acc:float):
    print("Achieved ", achieved_acc, " acc after ", round, " rounds")
    if Arguments.wandb:
        wandb.log({'round to target acc':round})

def updateBestResults(model:nn.Module, avg_acc, avg_loss):
    if Arguments.wandb:
        if avg_acc > Arguments.best_accuracy:
            Arguments.best_accuracy = avg_acc
            wandb.run.summary["best_accuracy"] = avg_acc
        if avg_loss < Arguments.best_loss:
            Arguments.best_loss = avg_loss
            wandb.run.summary["best_loss"] = avg_loss
            # save sanpshot of model
            if Arguments.snapshot:
                model_path = "snapshots/" + Arguments.algorithm + "_" + str(Arguments.seed) + "_model" + ".pt"
                torch.save(model.cpu().state_dict(), model_path)
                state = copy.deepcopy(model.state_dict()).cpu()
                torch.save(state, model_path)

def check_validation_and_stopping(counter, model, num_updates, num_rounds, rank, group, validation_loader):
    stop = torch.tensor(0)
    if rank == 0:
        cuda_dev = get_device(rank)
        loss, acc = validation_accuracy(model, cuda_dev, validation_loader)
        log_metric(
            ["Test Accuracy", "Test Loss", "num_updates", "rounds"],
            [acc, loss, num_updates, num_rounds],
            num_updates
        )
        updateBestResults(model, acc, loss)
        if acc > Arguments.target_acc:
            log_rounds_to_target(counter, acc)
        stop = torch.tensor(should_stop(counter, acc))
    dist.broadcast(stop, 0, group)
    return stop


def log_metric(names:list, values, round:int, printing:bool = True):
    labels = [name + ' : ' +str(value) for name, value in zip(names, values)]
    if printing:
        print(', '.join(labels), ", round: ",round)
    if Arguments.wandb:
        wandb.log({name:value for name,value in zip(names, values)}, step=round)

def zero_parameters(pars: dict):
    res = copy.copy(pars)
    for key, value in res.items():
        res[key] = torch.zeros_like(value)
    return res

def add_network_direction_of_update(par1: dict,par2: dict, times: float = 1.0):
    for key in par1.keys():
        par1[key].data += times*par2[key]

def replace_model_data(model1, model2):
    model2_parameters = list(model2.parameters())
    for i, param in enumerate(model1.parameters()):
        param.data = model2_parameters[i].data.clone()

def get_gradient(pars:dict):#carefull: in place =
    grads = []
    for param in pars:
        grads.append(param.grad.clone())
    return grads

def add_parameters(par1: dict, par2: dict, times: float = 1.0, times_0: float=1.0):
    if times_0==1 and times == 0:
        return copy.deepcopy(par1)
    elif times_0==0 and times ==1:
        return copy.deepcopy(par2)
    res = copy.copy(par1)
    for key in res.keys():
        res[key] = times_0*par1[key]+ times*par2[key]
    return res

def in_place_add_parameters(par1: dict,par2: dict):
    for key in par1.keys():
        par1[key] += par2[key]

def compute_norm(l:list, squared:bool=False):
    norm = 0
    for param in l:
        norm += torch.norm(param).item() ** 2
    if squared:
        return norm
    else:
        norm = norm ** 0.5
        return norm

def compute_norm_model_grad(model):
    norm = 0
    for param in model.parameters():
        norm += torch.norm(param.grad).item() ** 2

    norm = norm ** 0.5
    return norm

def create_zero_list(model, cpu = False):
    l = []
    param_list = list(model.parameters())
    for i in range(0, len(param_list)):
        if cpu:
            l.append(torch.zeros_like(param_list[i]).to('cpu'))
        else:
            l.append(torch.zeros_like(param_list[i]))
    return l

def clip_norm_list(l, model, args):
    for i, param in enumerate(model.parameters()):
        param.grad = l[i].clone()

    if args["gradient_clipping"]:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args["clipping_norm"], norm_type=2)
    for i, param in enumerate(model.parameters()):
        l[i] = param.grad.data.clone()
    return l

def create_model_param_list(model):
    l = []
    for param in model.parameters():
        l.append(param.data.clone())
    return l

def cuda_to_cpu(par: dict):
    res= copy.copy(par)
    for key, value in res.items():
        res[key] = value.cpu()
    return res

def cpu_to_cuda(par: dict):
    res= copy.copy(par)
    for key, value in res.items():
        res[key] = value.cuda()
    return res

def cosine_similarity(weights_from_grad: list, weights_from_data:list,is_in_grad:bool=True):
    weights1=copy.copy(weights_from_grad)
    if not is_in_grad:
        for i,par in enumerate(weights1):
            weights1[i]=par.view(-1)
    else:
        for i,par in enumerate(weights1):
            weights1[i]=par.grad.data.view(-1)
    weights2=copy.copy(weights_from_data)
    for i,par in enumerate(weights2):
        weights2[i]=par.data.view(-1)
    weights1=torch.cat(weights1)
    weights2=torch.cat(weights2)
    return torch.nn.functional.cosine_similarity(weights1,weights2,dim=0)

