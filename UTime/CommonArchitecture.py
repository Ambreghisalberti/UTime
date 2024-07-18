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
        self.label_names=kwargs.get('label_names',['label_BL'])
        self.n_time = n_time
        self.nb_channels_spectro = nb_channels_spectro
        self.filters = filters if isinstance(filters, list) else [int(filters * 2 ** i) for i in range(self.depth)]
        self.poolings = poolings if isinstance(poolings, (list, tuple)) else [poolings] * (self.depth - 1)
        self.kernels = kernels if isinstance(kernels, (list, tuple)) else [kernels] * self.depth
        self.dropout = kwargs.get('dropout', 0)
        self.nb_blocks_per_layer = kwargs.get('nb_blocks_per_layer', 1)   # Make it used for all architectures

        self.sizes = [n_time]
        self.nb_channels = [nb_channels_spectro]
        self.nb_moments = nb_moments
        self.check_inputs()
        self.batch_norm = kwargs.get('batch_norm', True)


    def check_inputs(self):
        if len(self.poolings) != self.depth - 1:
            raise Exception("The number of pooling kernel sizes needs to be one less than the network's depth.")
        #if len(self.kernels) != self.depth:
        #    raise Exception("The number of convolution kernel sizes needs to be equal to the network's depth.")
        if len(self.filters) != self.depth:
            raise Exception("The number of filters needs to be equal to the network's depth, or a single integer.")
        return None

    def get_kernel_size(self, size_data, given_kernel_size):
        if size_data < given_kernel_size:
            kernel_size = (size_data - 1) // 2 * 2 + 1
            '''This gives the closest smaller odd number (if nb_channels_spectro is odd, the value is kept, 
            otherwise it gives nb_channels_spectro-1'''
        else:
            kernel_size = given_kernel_size

        assert kernel_size >= 1, "kernel size must be strictly greater than 0"
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

    def conv_block(self, i, kernel_size1, kernel_size2, **kwargs):
        layers = []
        layers.append(nn.Dropout(self.dropout))

        if i == 0:
            if kernel_size1 == 1:  # We are working on moments
                input_size = self.nb_moments
            else:
                input_size = kwargs.get('nb_channels_in',1)
            layers.append(
                nn.Conv2d(input_size, self.filters[i], kernel_size=(kernel_size1, kernel_size2), padding='same'))
        else:
            layers.append(nn.Conv2d(self.filters[i - 1], self.filters[i], kernel_size=(
                kernel_size1, kernel_size2), padding='same'))

        if self.batch_norm:
            layers.append(BatchNorm2d(num_features=self.filters[i]))
        layers.append(nn.ReLU(inplace=True))

        return layers

    def _build_conv_block(self, i, size_data1, given_pooling_size1, size_data2, given_pooling_size2, **kwargs):
        kernel_size1 = self.get_kernel_size(size_data1, given_pooling_size1)
        kernel_size2 = self.get_kernel_size(size_data2, given_pooling_size2)

        layers = self.conv_block(i, kernel_size1, kernel_size2, **kwargs)
        return layers

    def add_pooling(self, i, nb_features):
        layers = []
        pooling1 = self.get_pooling_size(nb_features, self.poolings[i])
        pooling2 = self.get_pooling_size(self.sizes[i], self.poolings[i])
        layers.append(nn.MaxPool2d(kernel_size=(pooling1, pooling2)))
        return layers, pooling1, pooling2

    def _build_encoder1D(self):
        self.sizes = [self.n_time]
        layers = []
        for i in range(self.depth - 1):
            layers += self._build_conv_block(i, 1, 1, self.sizes[-1], self.kernels[i])
            new_layers, pooling1, pooling2 = self.add_pooling(i, 1)
            layers += new_layers
            new_size = int(self.sizes[-1] // pooling2)
            assert new_size > 0, "There was too much maxpooling, there is no data left"
            self.sizes.append(int(self.sizes[-1] // pooling2))

        layers += self._build_conv_block(-1, 1, 1, self.sizes[-1], self.kernels[-1])

        return nn.Sequential(*layers)

    def _build_encoder2D(self, **kwargs):
        self.nb_channels = [self.nb_channels_spectro]
        kernel_sizes = kwargs.get('kernel_sizes', self.kernels)
        layers = []
        for i in range(self.depth - 1):
            for iter in range(self.nb_blocks_per_layer):
                layers += self._build_conv_block(i, self.nb_channels[-1], kernel_sizes[i],
                                                 self.sizes[i], kernel_sizes[i], **kwargs)

            new_layers, pooling1, pooling2 = self.add_pooling(i, self.nb_channels[-1])
            layers += new_layers
            self.nb_channels.append(int(self.nb_channels[-1] // pooling1))

        # Last block without maxpooling
        layers += self._build_conv_block(-1, self.nb_channels[-1], kernel_sizes[-1],
                                         self.sizes[-1], kernel_sizes[-1], **kwargs)

        if self.batch_norm:
            layers.append(BatchNorm2d(num_features=self.filters[-1]))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers).to(self.device)

    def _build_classifier(self, **kwargs):
        nb_classes = kwargs.get('nb_classes_classifier',self.n_classes)
        layers = [nn.Dropout(self.dropout), Conv2d(self.filters[0], nb_classes, kernel_size=(1, 1))]

        if self.batch_norm:
            layers.append(BatchNorm2d(num_features=nb_classes))

        if nb_classes == 1:
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