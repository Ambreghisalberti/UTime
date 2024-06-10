import torch
import torch.nn as nn
from torch.nn import (MaxPool2d, Conv2d, Upsample, BatchNorm2d)
from UTime.EvaluateAndPred import Model


class Architecture(nn.Module, Model):
    def __init__(self, n_classes,
                 n_time,
                 nb_moments,
                 nb_channels_spectro,
                 depth,
                 filters,
                 kernels,
                 poolings,
                 **kwargs):

        super(Architecture, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.depth = depth
        self.n_classes = n_classes
        self.n_time = n_time
        self.filters = filters if isinstance(filters, list) else [int(filters * 2 ** i) for i in range(self.depth)]
        self.poolings = poolings if isinstance(poolings, (list, tuple)) else [poolings] * (self.depth - 1)
        self.kernels = kernels if isinstance(kernels, (list, tuple)) else [kernels] * self.depth
        self.dropout = kwargs.get('dropout', 0)

        self.sizes = [n_time]
        self.nb_channels_spectro = [nb_channels_spectro]
        self.nb_moments = nb_moments
        self.check_inputs()
        self.batch_norm = kwargs.get('batch_norm', True)



    def check_inputs(self):
        if len(self.poolings) != self.depth - 1:
            raise Exception("The number of pooling kernel sizes needs to be one less than the network's depth.")
        if len(self.kernels) != self.depth:
            raise Exception("The number of convolution kernel sizes needs to be equal to the network's depth.")
        if len(self.filters) != self.depth:
            raise Exception("The number of filters needs to be equal to the network's depth, or a single integer.")
        return None

    def get_kernel_size(self,size_data, given_kernel_size):
        if size_data < given_kernel_size:
            kernel_size = (size_data - 1) // 2 * 2 + 1
            '''This gives the closest smaller odd number (if nb_channels_spectro is odd, the value is kept, 
            otherwise it gives nb_channels_spectro-1'''
        else:
            kernel_size = given_kernel_size
        return kernel_size

    def get_pooling_size(self, size_data, given_pooling_size):
        if given_pooling_size <= size_data:
            return given_pooling_size
        else:
            return size_data

    def apply_batchnorm(self, x, layer):
        a, b, c, d = x.shape
        if c * d > 1:
            x = layer(x)
        return x

    def _build_conv_block(self, i, kernel_size1, kernel_size2):
        layers = []
        layers.append(nn.Dropout(self.dropout))

        if i == 0:
            if kernel_size1 == 1:  # We are working on moments
                input_size = self.nb_moments
            else:
                input_size = 1
            layers.append(
                nn.Conv2d(input_size, self.filters[i], kernel_size=(kernel_size1, kernel_size2), padding='same'))
        else:
            layers.append(nn.Conv2d(self.filters[i - 1], self.filters[i], kernel_size=(
                kernel_size1, kernel_size2), padding='same'))

        if self.batch_norm:
            layers.append(BatchNorm2d(num_features=self.filters[i]))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def add_pooling(self, i, nb_features):
        layers = []
        pooling1 = self.get_pooling_size(nb_features, self.poolings[i])
        pooling2 = self.get_pooling_size(self.sizes[i], self.poolings[i])
        layers.append(nn.MaxPool2d(kernel_size=(pooling1, pooling2)))
        return nn.Sequential(*layers), pooling1, pooling2

    def _build_encoder1D(self):
        layers = []
        for i in range(self.depth - 1):
            layers.append(self._build_conv_block(i, 1, self.kernels[i]))
            new_layers, pooling1, pooling2 = self.add_pooling(i, 1)
            layers.append(new_layers)
            self.sizes.append(int(self.sizes[-1] // pooling2))

        layers.append(self._build_conv_block(-1, 1, self.kernels[-1]))

        return nn.Sequential(*layers)

    def _build_decoder(self):
        layers = []
        for i in range(1, self.depth):
            # layers.append(nn.Upsample(scale_factor=(1,2)))
            layers.append(nn.Upsample(size=(1, self.sizes[::-1][i])))
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Conv2d(self.filters[-i] + 2 * self.filters[-i - 1], self.filters[-i - 1],
                                    kernel_size=(1, self.kernels[-i]), padding='same'))
            if self.batch_norm:
                layers.append(BatchNorm2d(num_features=self.filters[-i - 1]))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _build_classifier(self):

        layers = [nn.Dropout(self.dropout), Conv2d(self.filters[0], self.n_classes, kernel_size=(1, 1))]
        if self.batch_norm:
            layers.append(BatchNorm2d(num_features=self.n_classes))

        if self.n_classes == 1:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Softmax(dim=1))

        return nn.Sequential(*layers)

    def forward_encoder(self, x, encoder):
        encoder_outputs = []
        for i, layer in enumerate(encoder):
            if isinstance(layer, nn.BatchNorm2d):
                x = self.apply_batchnorm(x, layer)
            else:
                if isinstance(layer, nn.MaxPool2d):
                    encoder_outputs.append(x)
                x = layer(x)
        return x, encoder_outputs