import torch
import torch.nn as nn
from torch.nn import (MaxPool2d, Conv2d, Upsample, BatchNorm2d)
from UTime.EvaluateAndPred import Model


class UTime(nn.Module, Model):
    def __init__(self, n_classes,
                 n_time,
                 nb_moments,
                 nb_channels_spectro,
                 depth,
                 filters,
                 kernels,
                 poolings,
                 **kwargs):

        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.depth = depth
        self.n_classes = n_classes
        self.n_time = n_time
        self.filters = filters if isinstance(filters, list) else [int(filters * 2 ** i) for i in range(self.depth)]
        self.poolings = poolings if isinstance(poolings, (list, tuple)) else [poolings] * (self.depth - 1)
        self.kernels = kernels if isinstance(kernels, (list, tuple)) else [kernels] * self.depth

        self.sizes = [n_time]
        self.nb_channels_spectro = [nb_channels_spectro]
        self.nb_moments = nb_moments
        self.check_inputs()
        self.batch_norm = kwargs.get('batch_norm', True)

        # Encoder layers
        self.encoder = self._build_encoder1D()
        self.encoder2D = self._build_encoder2D()
        # print(self.encoder)
        # print(self.encoder2D)

        # Decoder layers
        self.decoder = self._build_decoder()
        # print(self.decoder)

        self.classifier = self._build_classifier()

    def check_inputs(self):
        if len(self.poolings) != self.depth - 1:
            raise Exception("The number of pooling kernel sizes needs to be one less than the network's depth.")
        if len(self.kernels) != self.depth:
            raise Exception("The number of convolution kernel sizes needs to be equal to the network's depth.")
        if len(self.filters) != self.depth:
            raise Exception("The number of filters needs to be equal to the network's depth, or a single integer.")
        return None

    def _build_encoder1D(self):
        layers = []
        for i in range(self.depth - 1):
            if i == 0:
                layers.append(
                    nn.Conv2d(self.nb_moments, self.filters[i], kernel_size=(1, self.kernels[i]), padding='same'))
            else:
                layers.append(
                    nn.Conv2d(self.filters[i - 1], self.filters[i], kernel_size=(1, self.kernels[i]), padding='same'))
            if self.batch_norm:
                layers.append(BatchNorm2d(num_features=self.filters[i]))
            layers.append(nn.ReLU(inplace=True))

            layers.append(nn.MaxPool2d(kernel_size=(1, self.poolings[i])))

            self.sizes.append(int(self.sizes[-1] // self.poolings[i]))

        layers.append(nn.Conv2d(self.filters[-2], self.filters[-1], kernel_size=(self.kernels[-1], self.kernels[-1]),
                                padding='same'))
        if self.batch_norm:
            layers.append(BatchNorm2d(num_features=self.filters[-1]))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _build_encoder2D(self):
        layers = []
        for i in range(self.depth - 1):
            if i == 0:
                layers.append(
                    nn.Conv2d(1, self.filters[i], kernel_size=(min(self.kernels[i], self.nb_channels_spectro[-1]), self.kernels[i]), padding='same'))
            else:
                layers.append(nn.Conv2d(self.filters[i - 1], self.filters[i], kernel_size=(
                min(self.kernels[i], self.nb_channels_spectro[-1]), self.kernels[i]), padding='same'))

            if self.batch_norm:
                layers.append(BatchNorm2d(num_features=self.filters[i]))
            layers.append(nn.ReLU(inplace=True))

            layers.append(nn.MaxPool2d(kernel_size=(self.poolings[i], self.poolings[i])))

            self.nb_channels_spectro.append(int(self.nb_channels_spectro[-1] // self.poolings[i]))

        # Last block without maxpooling
        layers.append(nn.Conv2d(self.filters[-2], self.filters[-1], kernel_size=(min(self.kernels[-1], self.nb_channels_spectro[-1]), self.kernels[-1]),
                                padding='same'))
        if self.batch_norm:
            layers.append(BatchNorm2d(num_features=self.filters[-1]))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _build_decoder(self):
        layers = []
        for i in range(1, self.depth):
            # layers.append(nn.Upsample(scale_factor=(1,2)))
            layers.append(nn.Upsample(size=(1, self.sizes[::-1][i])))
            layers.append(nn.Conv2d(self.filters[-i] + 2 * self.filters[-i - 1], self.filters[-i - 1],
                                    kernel_size=(1, self.kernels[-i]), padding='same'))
            if self.batch_norm:
                layers.append(BatchNorm2d(num_features=self.filters[-i - 1]))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _build_classifier(self):
        layers = [Conv2d(self.filters[0], self.n_classes, kernel_size=(1, 1))]
        if self.batch_norm:
            layers.append(BatchNorm2d(num_features=self.n_classes))

        if self.n_classes == 1:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Softmax(dim=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        moments, spectro = x

        # Encoder
        encoder_moments_outputs = []
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.MaxPool2d):
                encoder_moments_outputs.append(moments)
            moments = layer(moments)

        # Encoder 2D
        encoder_spectro_outputs = []
        for i, layer in enumerate(self.encoder2D):
            if isinstance(layer, nn.MaxPool2d):
                encoder_spectro_outputs.append(spectro)
            spectro = layer(spectro)

        # Concatenate moments and spectro encoder info
        spectro = torch.mean(spectro, dim=2)
        ''' These following lines ensure that if there are still severl channels in the spectro, 
        it will be transformed into something of the same shape as the moments resultss, to be concatenated'''
        a, b, c = spectro.shape
        spectro = spectro.reshape((a, b, 1, c))
        x = torch.cat([spectro, moments], dim=1).double()
        conv = nn.Conv2d(self.filters[-1] * 2, self.filters[- 1], kernel_size=(1, self.kernels[-1]),
                         padding='same').double().to(self.device)
        x = conv(x.double())

        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.Upsample):
                x = layer(x)

                res_connection_spectro = encoder_spectro_outputs.pop()
                res_connection_spectro = torch.mean(res_connection_spectro, dim=2)
                a, b, c = res_connection_spectro.shape
                res_connection_spectro = res_connection_spectro.reshape((a, b, 1, c))
                res_connection_moments = encoder_moments_outputs.pop()

                x = torch.cat([x, res_connection_spectro, res_connection_moments], dim=1)

            else:
                x = layer(x)

        # Dense classification
        for layer in self.classifier:
            x = layer(x)

        return x
