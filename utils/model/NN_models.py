import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utility_functions.arguments import Arguments
from torch.autograd import Variable


class LSTMModel(torch.nn.Module):
    def __init__(self, word_emb_array, hidden_dim=-1,
                 n_lstm_layers=2):
        super(LSTMModel, self).__init__()

        torch.set_default_dtype(torch.float64)

        # Word embedding
        embedding_dim = word_emb_array.shape[1]
        self.output_dim = word_emb_array.shape[0]
        self.embedding = torch.nn.Embedding.from_pretrained(
            torch.DoubleTensor(word_emb_array))

        # Hidden dimensions
        self.hidden_dim = hidden_dim if hidden_dim > 0 else embedding_dim

        # Number of stacked lstm layers
        self.n_lstm_layers = n_lstm_layers

        # shape of input/output tensors: (batch_dim, seq_dim, feature_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, self.hidden_dim, n_lstm_layers, batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x):
        # word embedding
        x = self.embedding(x)

        self.h0 = torch.zeros(self.n_lstm_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()
        self.c0 = torch.zeros(self.n_lstm_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()

        # lstm
        out, _ = self.lstm(x, (self.h0.detach(), self.c0.detach()))

        # Index hidden state of last time step; out.size = `batch, seq_len, hidden`
        out = self.fc(out[:, -1, :])
        return out.reshape(-1, self.output_dim)  # hard-coded for binary classification

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size=100, model="lstm", n_layers=2, word_emb_array=None):
        super(CharRNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = input_size
        self.n_layers = n_layers


        embedding_dimension = 8 if word_emb_array is None else word_emb_array.shape[1]
        self.encoder = nn.Embedding(input_size, embedding_dimension)
        if word_emb_array is not None:
            self.encoder = torch.nn.Embedding.from_pretrained(
                torch.FloatTensor(word_emb_array))
        if self.model == "gru":
            self.rnn = nn.GRU(embedding_dimension, hidden_size, n_layers)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(embedding_dimension, hidden_size, n_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, inp):
        batch_size = inp.size(0)
        self.hidden = self.init_hidden(batch_size)
        self.zero_grad()

        encoded = self.encoder(inp)
        output, _ = self.rnn(encoded, self.hidden)
        output = output[:, -1, :]
        output = self.decoder(output)

        return output.reshape(-1, self.input_size)

    def init_hidden(self, batch_size):
        if self.model == "lstm":
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda()),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda())
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda())


class CNN_MNIST(nn.Module):
    def __init__(self, input_size = (1, 28, 28), num_classes = 10):
        """
        init convolution and activation layers
        Args:
            input_size: (1,28,28)
            num_classes: 10
        """
        super(CNN_MNIST, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(input_size[0], 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.fc1 = nn.Linear(4 * 4 * 64, num_classes)

    def forward(self, x):
        """
        forward function describes how input tensor is transformed to output tensor
        Args:
            x: (Nx1x28x28) tensor
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return x



class CNN_CIFAR10(nn.Module):
    def __init__(self, w_conv_bias=True, w_fc_bias=True):
        super(CNN_CIFAR10, self).__init__()
        # decide the num of classes.
        self.num_classes = 10 if Arguments.task == "CIFAR10" else 100

        # define layers.
        self.conv1 = nn.Conv2d(3, 6, 5, bias=w_conv_bias)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=w_conv_bias)
        self.fc1 = nn.Linear(16 * 5 * 5, 120, bias=w_fc_bias)
        self.fc2 = nn.Linear(120, 84, bias=w_fc_bias)
        self.fc3 = nn.Linear(84, self.num_classes, bias=w_fc_bias)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleNet(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
        #return F.log_softmax(x,dim=0)

class CIFAR10ConvNet(nn.Module):
    def __init__(self):
        super(CIFAR10ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x), 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EMNISTNet(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(EMNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5,padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5,padding=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc2 = nn.Linear(1024, 47)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
        #return F.log_softmax(x,dim=0)

class CNN(nn.Module):
    def __init__(self, dataset):
        super(CNN, self).__init__()
        self.dataset = dataset
        if dataset == "cifar10" or dataset == "mnist" or dataset.startswith("emnist"):
            if dataset == "cifar10" or dataset == "mnist":
                self.num_classes = 10
            elif dataset == 'emnist-byclass':
                self.num_classes = 62
            elif dataset == 'emnist-balanced' or dataset == 'emnist-bymerge':
                self.num_classes = 47
            if dataset == "cifar10":
                side_size = 32
                n_filters = 3
            else:
                n_filters = 1
                side_size = 28
            flatten_side_size = side_size - 2 * 3

            class Flatten(nn.Module):
                def forward(self, inp):
                    return inp.view(-1,flatten_side_size*flatten_side_size*64)

            self.net = nn.Sequential(
                nn.Conv2d(n_filters, 32, (3, 3)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (3, 3)),
                nn.ReLU(),
                nn.Conv2d(64, 64, (3, 3)),
                nn.ReLU(),
                Flatten(),
                nn.Linear(64 * flatten_side_size * flatten_side_size, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_classes)
            )
        elif dataset.startswith("digits"):
            if dataset == "digits":
                self.num_classes = 10
            else:
                self.num_classes = int(dataset[6:])
            self.net = nn.Sequential(
                nn.Conv2d(1, 8, (3, 3)),
                nn.ReLU(),
                nn.Conv2d(8, 16, (3, 3)),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(16 * 4 * 4, 32),
                nn.ReLU(),
                nn.Linear(32, self.num_classes)
            )

    def forward(self, x):
        return self.net(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_batchnorm=True):
        super(BasicBlock, self).__init__()
        self.active = True
        self.all_blocks = None
        self.index = None
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)

        if not use_batchnorm:
            self.bn1 = self.bn2 = nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes) if use_batchnorm else nn.Sequential()
            )

    def next_active_block(self):
        block = self
        while block.next_block() is not None:
            block = block.next_block()
            if block.active:
                return block
        return None

    def previous_active_block(self):
        block = self
        while block.previous_block() is not None:
            block = block.previous_block()
            if block.active:
                return block
        return None

    def next_block(self):
        return self.all_blocks[self.index + 1] if self.index + 1 < len(self.all_blocks) else None

    def previous_block(self):
        return self.all_blocks[self.index - 1] if self.index - 1 > -1 else None

    def forward(self, x, scaling):
        if not self.active:
            return self.shortcut(x)

        out = float(scaling) * F.relu(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        out += self.shortcut(x)

        out = F.relu(out)
        return out


class Layer(nn.Module):
    def __init__(self, block, in_planes, planes, num_blocks, stride, use_batchnorm: bool = True):
        super(Layer, self).__init__()
        self.level = 0
        self.use_batchnorm = use_batchnorm
        self.in_planes = in_planes

        strides = [stride] + [1] * (num_blocks - 1)
        self.blocks = []
        for stride in strides:
            newblock = block(self.in_planes, planes, stride, self.use_batchnorm)
            self.blocks.append(newblock)
            self.layersBundle = nn.Sequential(*self.blocks)

        for i, block in enumerate(self.blocks):
            block.all_blocks = self.blocks
            block.index = i

    def set_level(self, level, with_reset):
        self.level = level
        if with_reset:
            for i, block in enumerate(self.blocks):
                block.active = (i % 2 ** self.level == 0)

    def forward(self, x, scaling):
        for block in self.blocks:
            x = block.forward(x, scaling)
        return x


class Transition(nn.Module):
    def __init__(self, in_planes, planes, stride, use_batchnorm: bool = True):
        super(Transition, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) if use_batchnorm else nn.Sequential()

    def forward(self, x):
        return self.bn1(self.conv1(x))


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_batchnorm=True):
        super(ResNet, self).__init__()

        # self.param_names = ["conv1.weight", "bn1.weight", "bn1.bias", "conv2.weight", "bn2.weight", "bn2.bias"]
        self.param_names = ["conv1.weight", "bn1.weight", "bn1.bias"]

        self.level = 0
        self.in_planes = 64
        self.use_batchnorm = use_batchnorm
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if use_batchnorm else nn.Sequential()
        self.layer1 = Layer(block, 64, 64, num_blocks[0], stride=1, use_batchnorm=use_batchnorm)
        self.transition1 = Transition(64, 128, stride=2, use_batchnorm=use_batchnorm)
        self.layer2 = Layer(block, 128, 128, num_blocks[1], stride=1, use_batchnorm=use_batchnorm)
        self.transition2 = Transition(128, 256, stride=2, use_batchnorm=use_batchnorm)
        self.layer3 = Layer(block, 256, 256, num_blocks[2], stride=1, use_batchnorm=use_batchnorm)
        self.transition3 = Transition(256, 512, stride=2, use_batchnorm=use_batchnorm)
        self.layer4 = Layer(block, 512, 512, num_blocks[3], stride=1, use_batchnorm=use_batchnorm)

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]

        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def set_grads_to_none(self):
        for param in self.parameters():
            param.grad = None

    def num_active_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.grad is not None)

    def set_level(self, level, with_reset=False):
        self.level = level
        for layer in self.layers:
            layer.set_level(level, with_reset)

    def get_active_blocks(self):
        active_blocks = []
        for layer in self.layers:
            active_blocks.append([i for i, block in enumerate(layer.blocks) if block.active])
        return active_blocks

    def get_all_blocks(self):
        all_blocks = []
        for layer in self.layers:
            all_blocks.append([i for i, block in enumerate(layer.blocks)])
        return all_blocks

    def smooth_layers(self, domain, restriction_domain=None, block_list=None):
        """For each block in the restriction domain, it takes
        the next a previous items in the domain and averages their data"""
        if restriction_domain is None:
            restriction_domain = domain
        with torch.no_grad():
            for j, layer in enumerate(self.layers):
                if block_list is not None:
                    layer = block_list[j]
                layer_domain = domain[j]
                layer_restriction_domain = restriction_domain[j]
                for i, block in enumerate(layer.blocks):
                    if block_list is not None:
                        block = block_list[j][i]
                    if i in layer_restriction_domain:
                        index = layer_domain.index(i)
                        previous_index = layer_domain[index - 1] if index - 1 > -1 else None
                        next_index = layer_domain[index + 1] if index + 1 < len(layer_domain) else None

                        prev_block = dict(
                            layer.blocks[previous_index].named_parameters()
                        ) if previous_index is not None else None
                        next_block = dict(
                            layer.blocks[next_index].named_parameters()
                        ) if next_index is not None else None
                        block_params = dict(block.named_parameters())
                        # not exactly what we want, there is a cascading effect
                        for name in self.param_names:
                            div = 1
                            if prev_block is not None:
                                block_params[name].data += prev_block[name].data
                                div += 1
                            if next_block is not None:
                                block_params[name].data += next_block[name].data
                                div += 1
                            block_params[name].data /= div
        if block_list is not None:
            return block_list

    def activate_blocks_per_layer(self, blocks_per_layer):
        for i, layer in enumerate(self.layers):
            for j, block in enumerate(layer.blocks):
                block.active = j in blocks_per_layer[i]

    def activate_all_blocks(self):
        for i, layer in enumerate(self.layers):
            for block in layer.blocks:
                block.active = True

    def forward(self, x):
        scaling = 2 ** self.level if Arguments.scaling else 1
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out, scaling)
        out = self.transition1(out)
        out = self.layer2(out, scaling)
        out = self.transition2(out)
        out = self.layer3(out, scaling)
        out = self.transition3(out)
        out = self.layer4(out, scaling)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [4, 4, 4, 4], num_classes=num_classes, use_batchnorm=Arguments.batchnorm)

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [8, 8, 8, 8], num_classes=num_classes, use_batchnorm=Arguments.batchnorm)


#
# def ResNet50(num_classes=10, use_batchnorm=True):
#     return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes, use_batchnorm=Arguments.batchnorm)
#
# def ResNet101(num_classes=10, use_batchnorm=True):
#     return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes, use_batchnorm=Arguments.batchnorm)
#
# def ResNet152(num_classes=10, use_batchnorm=True):
#     return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes, use_batchnorm=Arguments.batchnorm)



