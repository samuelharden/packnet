"""Contains various network definitions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from sklearn.manifold import TSNE

from fastai.text import *
from fastai.lm_rnn import *



class View(nn.Module):
    """Changes view using a nn.Module."""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class TextModelMY(nn.Module):
    """Text Module from fastai used else where and different classifier"""

    def __init__(self, make_model=True, vocabulary_size=30002):
        super(TextModelMY, self).__init__()
        self.vocabulary_size = vocabulary_size
        if make_model:
            self.make_model()

    def make_model(self):
        """Creates the model."""
        # Get the pretrained model.
        bptt,em_sz,nh,nl = 70,400,1150,3
        dps = np.array([0.4,0.5,0.05,0.3,0.4])*1.0
        self.datasets, self.classifiers = [], nn.ModuleList()

        # Shared params are those which are to be pruned.
        self.shared = MultiBatchRNN(bptt, 20*70, self.vocabulary_size, em_sz, nh, nl, 1, False,
                      dropouth=dps[3], dropouti=dps[0], dropoute=dps[2], wdrop=dps[1], qrnn=False)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

        # Make sure conv transform is correct.
        # self.check_correctness(vgg16)

    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        print("Adding dataset", dataset, num_outputs)
        bptt,em_sz,nh,nl = 70,400,1150,3
        dps = np.array([0.4,0.5,0.05,0.3,0.4])*1.0
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(PoolingLinearClassifier([em_sz*3, 50, num_outputs], [dps[4], 0.1]))

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]
        self.model = nn.Sequential(self.shared, self.classifier)

    def forward(self, x):
        print("Batch", x.size())
        x = self.shared(x)
        #x = x.view(x[0].size(0), -1)
        #print("Shared", x[0])
        #X_embedded = TSNE(n_components=2).fit_transform(x[0])
        X_view = copy.deepcopy(x[0][0])
        print("Before", X_view.shape)
        X_view = X_view.view(X_view.size(1), -1)
        print("After SHape", X_view.shape)
        sys.stdout.flush()
        X_embedded = TSNE(n_components=2).fit_transform(to_np(X_view))
        print(X_embedded)
        x = self.classifier(x)
        #print("Classifier", x.size())
        return x

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(TextModelMY, self).train(mode)

    def check_correctness(self, vgg16):
        """Ensures that conversion of fc layers to conv is correct."""
        # Test to make sure outputs match.
    #    vgg16.eval()
     #   self.shared.eval()
     #   self.classifier.eval()

      #  rand_input = Variable(torch.rand(1, 3, 224, 224))
       # fc_output = vgg16(rand_input)
       # print(fc_output)

        #x = self.shared(rand_input)
        #x = x.view(x.size(0), -1)
        #conv_output = self.classifier[-1](x)
        #print(conv_output)

        #print(torch.sum(torch.abs(fc_output - conv_output)))
        #assert torch.sum(torch.abs(fc_output - conv_output)).data[0] < 1e-8
        print('Check passed')
        raw_input()


class ModifiedVGG16(nn.Module):
    """VGG16 with different classifiers."""

    def __init__(self, make_model=True):
        super(ModifiedVGG16, self).__init__()
        if make_model:
            self.make_model()

    def make_model(self):
        """Creates the model."""
        # Get the pretrained model.
        vgg16 = models.vgg16(pretrained=True)
        self.datasets, self.classifiers = [], nn.ModuleList()

        idx = 6
        for module in vgg16.classifier.children():
            if isinstance(module, nn.Linear):
                if idx == 6:
                    fc6 = module
                elif idx == 7:
                    fc7 = module
                elif idx == 8:
                    self.datasets.append('imagenet')
                    self.classifiers.append(module)
                idx += 1
        features = list(vgg16.features.children())
        features.extend([
            View(-1, 25088),
            fc6,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            fc7,
            nn.ReLU(inplace=True),
            nn.Dropout(),
        ])

        # Shared params are those which are to be pruned.
        self.shared = nn.Sequential(*features)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

        # Make sure conv transform is correct.
        # self.check_correctness(vgg16)

    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(4096, num_outputs))

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]

    def forward(self, x):
        x = self.shared(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedVGG16, self).train(mode)

    def check_correctness(self, vgg16):
        """Ensures that conversion of fc layers to conv is correct."""
        # Test to make sure outputs match.
        vgg16.eval()
        self.shared.eval()
        self.classifier.eval()

        rand_input = Variable(torch.rand(1, 3, 224, 224))
        fc_output = vgg16(rand_input)
        print(fc_output)

        x = self.shared(rand_input)
        x = x.view(x.size(0), -1)
        conv_output = self.classifier[-1](x)
        print(conv_output)

        print(torch.sum(torch.abs(fc_output - conv_output)))
        assert torch.sum(torch.abs(fc_output - conv_output)).data[0] < 1e-8
        print('Check passed')
        raw_input()


class ModifiedVGG16BN(ModifiedVGG16):
    """VGG16 with batch norm."""

    def __init__(self, make_model=True):
        super(ModifiedVGG16BN, self).__init__(make_model=False)
        if make_model:
            self.make_model()

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedVGG16BN, self).train(mode)

        # Set the BNs to eval mode so that the running means and averages
        # do not update.
        for module in self.shared.children():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def make_model(self):
        """Creates the model."""
        # Get the pretrained model.
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.datasets, self.classifiers = [], nn.ModuleList()

        # Get classifiers.
        idx = 6
        for module in vgg16_bn.classifier.children():
            if isinstance(module, nn.Linear):
                if idx == 6:
                    fc6 = module
                elif idx == 7:
                    fc7 = module
                elif idx == 8:
                    self.datasets.append('imagenet')
                    self.classifiers.append(module)
                idx += 1

        features = list(vgg16_bn.features.children())
        features.extend([
            View(-1, 25088),
            fc6,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            fc7,
            nn.ReLU(inplace=True),
            nn.Dropout(),
        ])

        # Shared params are those which are to be pruned.
        self.shared = nn.Sequential(*features)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

        # Make sure conv transform is correct.
        # self.check_correctness(vgg16)


class ModifiedResNet(ModifiedVGG16):
    """ResNet-50."""

    def __init__(self, make_model=True):
        super(ModifiedResNet, self).__init__(make_model=False)
        if make_model:
            self.make_model()

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedResNet, self).train(mode)

        # Set the BNs to eval mode so that the running means and averages
        # do not update.
        for module in self.shared.modules():
            if 'BatchNorm' in str(type(module)):
                module.eval()

    def make_model(self):
        """Creates the model."""
        # Get the pretrained model.
        resnet = models.resnet50(pretrained=True)
        self.datasets, self.classifiers = [], nn.ModuleList()

        # Create the shared feature generator.
        self.shared = nn.Sequential()
        for name, module in resnet.named_children():
            if name != 'fc':
                self.shared.add_module(name, module)

        # Add the default imagenet classifier.
        self.datasets.append('imagenet')
        self.classifiers.append(resnet.fc)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(2048, num_outputs))


class ModifiedDenseNet(ModifiedVGG16):
    """DenseNet-121."""

    def __init__(self, make_model=True):
        super(ModifiedDenseNet, self).__init__(make_model=False)
        if make_model:
            self.make_model()

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedDenseNet, self).train(mode)

        # Set the BNs to eval mode so that the running means and averages
        # do not update.
        for module in self.shared.modules():
            if 'BatchNorm' in str(type(module)):
                module.eval()

    def make_model(self):
        """Creates the model."""
        # Get the pretrained model.
        densenet = models.densenet121(pretrained=True)
        self.datasets, self.classifiers = [], nn.ModuleList()

        # Create the shared feature generator.
        self.shared = densenet.features

        # Add the default imagenet classifier.
        self.datasets.append('imagenet')
        self.classifiers.append(densenet.classifier)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

    def forward(self, x):
        features = self.shared(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(features.size(0), -1)
        out = self.classifier(out)
        return out

    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(1024, num_outputs))
