import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utility_functions.arguments import Arguments


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
        self.num_classes = 10

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
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if not use_batchnorm:
            self.bn1 = self.bn2 = nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes) if use_batchnorm else nn.Sequential()
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, use_batchnorm=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        if not use_batchnorm:
            self.bn1 = self.bn2 = self.bn3 = nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes) if use_batchnorm else nn.Sequential()
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_batchnorm=True):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.use_batchnorm = use_batchnorm
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if use_batchnorm else nn.Sequential()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.use_batchnorm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10, use_batchnorm=True):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, use_batchnorm=Arguments.batchnorm)

def ResNet34(num_classes=10, use_batchnorm=True):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes, use_batchnorm=Arguments.batchnorm)

def ResNet50(num_classes=10, use_batchnorm=True):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes, use_batchnorm=Arguments.batchnorm)

def ResNet101(num_classes=10, use_batchnorm=True):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes, use_batchnorm=Arguments.batchnorm)

def ResNet152(num_classes=10, use_batchnorm=True):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes, use_batchnorm=Arguments.batchnorm)



