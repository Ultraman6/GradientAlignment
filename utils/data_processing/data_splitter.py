import torch
from torchvision import datasets, transforms
import math
import numpy as np
from utils.utility_functions.arguments import Arguments
import json
import os
from collections import defaultdict
import string
import torch.nn.functional as F

class MNISTTransform:
    @classmethod
    def train_transform(cls):
        return transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307, ), (0.3081, ))
           ])

class EMNISTTransform:
    @classmethod
    def train_transform(cls):
        return transforms.Compose([transforms.Resize((28,28)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))])
class CIFAR10Transform:
    data_mean = (0.4914, 0.4822, 0.4465)
    data_stddev = (0.2023, 0.1994, 0.2010)

    @classmethod
    def train_transform(cls):

        return transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(CIFAR10Transform.data_mean, CIFAR10Transform.data_stddev),
                    ]
                )

    @classmethod
    def test_transform(cls):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10Transform.data_mean, CIFAR10Transform.data_stddev),
        ])


class CIFAR100Transform:
    data_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    data_stddev = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    @classmethod
    def train_transform(cls):

        return transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(CIFAR100Transform.data_mean, CIFAR100Transform.data_stddev),
                    ]
                )

    @classmethod
    def test_transform(cls):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100Transform.data_mean, CIFAR100Transform.data_stddev),
        ])


def char_to_tokens(str_list):
    all_characters = string.printable
    new_str_list = []
    for str in str_list:
        char_list = []
        for c in str:
            char_list.append(all_characters.index(c))
        new_str_list.append(char_list)
    return new_str_list



def get_shakespeare_ds(train:bool):
    datasets = []
    extra = "train" if train else "test"
    clients, groups, data = read_dir(Arguments.data_home + "/shakespeare/"+extra)
    for c in clients:
        data_labels = [
            (torch.tensor(x), torch.tensor(y)) for x, y in
            zip(
                char_to_tokens(list(data[c].values())[0]),
                [t[1:] + r for t, r in
                 zip(char_to_tokens(list(data[c].values())[0]), char_to_tokens(list(data[c].values())[1]))]
            )
        ]
        datasets.append(data_labels)
    if train:
        return datasets
    else:
        return list(set().union(*datasets))



def get_sent140_ds(train:bool):

	def process_x(raw_x_batch):
        x_batch = [e[4] for e in raw_x_batch]
        x_batch = [line_to_indices(e, word_indices, seq_len) for e in x_batch]
        temp = np.asarray(x_batch)
        x_batch = torch.from_numpy(np.asarray(x_batch))
        return x_batch

	def process_y(raw_y_batch):
        #return torch.from_numpy(np.asarray(raw_y_batch, dtype=np.float32))
        return torch.from_numpy(np.asarray(raw_y_batch, dtype=np.float64))

    def get_word_emb_arr(path):
	    with open(path, 'r') as inf:
	        embs = json.load(inf)
	    vocab = embs['vocab']
	    word_emb_arr = np.array(embs['emba'])
	    indd = {}
	    for i in range(len(vocab)):
	        indd[vocab[i]] = i
	    vocab = {w: i for i, w in enumerate(embs['vocab'])}
	    return word_emb_arr, indd, vocab

	def sent140_preprocess_x(X):
	    x_batch = [e[4] for e in X]  # list of lines/phrases
	    x = np.zeros((len(x_batch), embed_dim))
	    for i in range(len(x_batch)):
	        line = x_batch[i]
	        words = split_line(line)
	        idxs = [vocab[word] if word in vocab.keys() else emb_array.shape[0] - 1
	                for word in words]
	        word_embeddings = np.mean([emb_array[idx] for idx in idxs], axis=0)
	        x[i, :] = word_embeddings
	    return x

	def sent140_preprocess_y(raw_y_batch):
	    res = []
	    for i in range(len(raw_y_batch)):
	        res.append(float(raw_y_batch[i]))
	    return res

	global VOCAB_DIR
	global emb_array
    global vocab
    global embed_dim
    VOCAB_DIR = 'sent140/embs.json'
    emb_array, _, vocab = get_word_emb_arr(VOCAB_DIR)
    # print('shape obtained : ' + str(emb_array.shape))
    embed_dim = emb_array.shape[1]

    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    # START Old version :
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        print('reading train file ' + str(file_path))
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        print('reading test file ' + str(file_path))
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])
    # END Old version

    #counter = 0
    #for f in train_files:
    #    file_path = os.path.join(train_data_dir, f)
    #    with open(file_path, 'r') as inf:
    #        cdata = json.load(inf)
    #    clients.extend(cdata['users'])
    #    if 'hierarchies' in cdata:
    #        groups.extend(cdata['hierarchies'])
    #    train_data.update(cdata['user_data'])
    #    counter += 1
    #    if counter == 50:
    #        break

    #clients = [list(train_data.keys()). list(test_data.keys())]
    if split_by_user:
        clients = {
            'train_users': list(train_data.keys()),
            'test_users': list(test_data.keys())
        }
    else:
        clients = {
            'train_users': list(train_data.keys())
        }

	datasets = []
    for u in clients['train_users']:
        data_labels = 
            (process_x(sent140_preprocess_x(train_data[u]['x'])), process_y(sent140_preprocess_y(train_data[u]['y'])))
        datasets.append(data_labels)
    if train:
        return datasets
    else:
        return list(set().union(*datasets))


def partition_noniid(dataset: torch.utils.data.Dataset,workers: int, percentage_iid: float,use_two_splits):
    perm = torch.randperm(len(dataset))
    iid_indices = perm[:int(len(dataset)*percentage_iid)]
    noniid_indices = perm[int(len(dataset)*percentage_iid):]
    targets = torch.tensor(dataset.targets).clone().detach()

    labels=torch.unique(targets)
    structured_noniiid_indices=[]
    for label in labels:
        indices = np.intersect1d( torch.squeeze((targets == label).nonzero()) , noniid_indices)
        structured_noniiid_indices.append(torch.from_numpy(indices))

    #if nshards == 2:
    #    all_indices=torch.cat(all_indices).view(2*workers,-1)
    #    all_indices=all_indices[torch.randperm(all_indices.size()[0])]
    #    all_indices=all_indices.view(workers,-1)
    structured_noniiid_indices=torch.cat(structured_noniiid_indices).view(workers,-1)
    if percentage_iid == 0:
        return [torch.utils.data.Subset(dataset, structured_noniiid_indices[i] ) for i in range(workers)]
    iid_indices=iid_indices.view(workers,-1)
    if use_two_splits == False:
        return [torch.utils.data.Subset(dataset, torch.cat((iid_indices[i],structured_noniiid_indices[i])) ) for i in range(workers)]
    else:
        return [[torch.utils.data.Subset(dataset, iid_indices[i] ), torch.utils.data.Subset(dataset, structured_noniiid_indices[i] ) ] for i in range(workers)]


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data

def split_data(task: str, workers: int, percentage_iid: float, use_two_splits: bool):
    if task == "MNIST":
        dataset = datasets.MNIST(root=Arguments.data_home, train=True, download=True, transform=MNISTTransform.train_transform())
    elif task == "EMNIST":
        dataset = datasets.EMNIST(root=Arguments.data_home,split="balanced" ,train=True, download=True,transform=EMNISTTransform.train_transform())
    elif task == "CIFAR100":
        dataset = datasets.CIFAR100(root=Arguments.data_home, train=True, download=True,
                                   transform=CIFAR100Transform.train_transform())
    elif Arguments.task == "shakespeare":
        return get_shakespeare_ds(train=True)
    elif Arguments.task == "sent140":
    	return get_sent140_ds(train=True)
    else :# task == "CIFAR10"
        dataset = datasets.CIFAR10(root=Arguments.data_home, train=True, download=True,
                                   transform=CIFAR10Transform.train_transform())
    if percentage_iid == 1:
        return torch.utils.data.random_split(dataset, [math.floor(len(dataset)/workers) for _ in range(workers)])
    elif percentage_iid == -1:
        return [dataset for _ in range(workers)]
    else:
        return partition_noniid(dataset,workers,percentage_iid,use_two_splits)

def get_validation_data(task: str):
    if task == "MNIST":
        dataset =  datasets.MNIST(root=Arguments.data_home, train=False, download=True, transform=MNISTTransform.train_transform())
    elif task == "EMNIST":
        dataset =  datasets.EMNIST(root=Arguments.data_home,split="balanced" ,train=False, download=True,transform=EMNISTTransform.train_transform())
    elif task == "CIFAR100":
        dataset = datasets.CIFAR100(root=Arguments.data_home, train=False, download=True, transform=CIFAR100Transform.test_transform())
    elif Arguments.task == "shakespeare":
        return get_shakespeare_ds(train=False)
    elif Arguments.task == "sent140":
    	return get_sent140_ds(train=False)
    else:# task == "CIFAR10"
        dataset = datasets.CIFAR10(root=Arguments.data_home, train=False, download=True, transform=CIFAR10Transform.test_transform())
    return dataset
