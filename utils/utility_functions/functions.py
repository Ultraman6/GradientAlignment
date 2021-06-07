import torch
import copy
import wandb
from utils.utility_functions.accumulators import *
from utils.model.NN_models import *
from utils.utility_functions.arguments import Arguments
import torch.distributed as dist

def validation_accuracy(model: torch.nn.Module, cuda_dev):
    model.eval()
    mean_test_accuracy = Mean()
    mean_test_loss = Mean()
    validation_loader = Arguments.validation_loader
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_x, batch_y in validation_loader:
            batch_x, batch_y = batch_x.to(cuda_dev), batch_y.to(cuda_dev)
            prediction = model(batch_x)
            if type(prediction) == list:
                prediction = prediction[-1]
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
    elif Arguments.model == "resnet152":
        model = ResNet152(num_classes=num_classes)
    elif Arguments.model == "cnn_mnist":
        model = CNN_MNIST(num_classes=num_classes)
    elif Arguments.model == "cnn_cifar":
        model = CNN_CIFAR10()

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

def average_models(model: nn.Module, group):
    """ Model averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.data, op=torch.distributed.ReduceOp.SUM, group=group)
        param.data /= size

def compute_full_gradient(model:nn.Module, worker_index:int, criterion, dataloaders, group):
    full_gradient = create_zero_list(model)
    worker_full_gradient = create_zero_list(model)

    data_points = 0
    for data, target in dataloaders[worker_index]:
        data, target = data.cuda(), target.cuda()
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        data_points += len(data)
        for i, param in enumerate(model.parameters()):
            full_gradient[i] += param.grad.data * len(data)
        if Arguments.fg_batch_size > 0 and data_points >= Arguments.fg_batch_size:
            break

    for i, param in enumerate(model.parameters()):
        full_gradient[i] /= data_points
        param.grad = full_gradient[i].clone()

    for i, param in enumerate(model.parameters()):
        worker_full_gradient[i] = param.grad.data.clone()

    # average the full gradient overall workers
    average_lists(full_gradient, group)
    model.zero_grad()

    return worker_full_gradient, full_gradient


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

def check_validation_and_stopping(counter, model, num_updates, num_rounds, rank, group):
    stop = torch.tensor(0)
    if rank == 0:
        cuda_dev = get_device(rank)
        loss, acc = validation_accuracy(model, cuda_dev=cuda_dev)
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


def log_metric(names:list, values:list, round:int, printing:bool = True):
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
    temp=copy.copy(pars)
    for key in temp.keys():
        temp[key]=pars[key].grad.clone()
    return temp

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
        par1[key]+=par2[key]

def compute_norm(l:list):
    norm = 0
    for param in l:
        norm += torch.norm(param).item() ** 2

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

def cosine_similarity(weights_from_grad: list, weights_from_data:list,is_in_grad:bool =True):
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
