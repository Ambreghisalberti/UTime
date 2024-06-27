import torch
import torch.nn as nn
from torch.nn import (Conv2d, BatchNorm2d)
from UTime.EvaluateAndPred import Model


class AutoEncoder(nn.Module, Model):
    def __init__(self,
                 n_time,
                 depth,
                 filters,
                 kernels):

        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.depth = depth
        self.n_time = n_time
        self.filters = filters if isinstance(filters, list) else [int(filters * 2 ** i) for i in range(self.depth)]
        self.kernels = kernels if isinstance(kernels, (list, tuple)) else [kernels] * self.depth

        self.check_inputs()

        # Encoder layers
        self.encoder = self._build_encoder()
        # print(self.encoder)
        # Decoder layers
        self.decoder = self._build_decoder()
        # print(self.decoder)

    def check_inputs(self):
        if len(self.kernels) != self.depth:
            raise Exception("The number of convolution kernel sizes needs to be equal to the network's depth.")
        if len(self.filters) != self.depth:
            raise Exception("The number of filters needs to be equal to the network's depth, or a single integer.")
        return None

    def _build_encoder(self):
        layers = []
        for i in range(self.depth):
            if i == 0:
                layers.append(
                    nn.Conv2d(1, self.filters[i], kernel_size=(self.kernels[i], self.kernels[i]), padding='same'))
            else:
                # print(f'Layer {i}: {self.filters[i-1]} -> {self.filters[i]}, kernels = {self.kernels[i]}')
                layers.append(nn.Conv2d(self.filters[i - 1], self.filters[i],
                                        kernel_size=(self.kernels[i], self.kernels[i]),
                                        padding='same'))
            layers.append(BatchNorm2d(num_features=self.filters[i]))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _build_decoder(self):
        layers = []
        for i in range(1, self.depth):
            layers.append(nn.Conv2d(self.filters[-i], self.filters[-i - 1],
                                    kernel_size=(self.kernels[-i], self.kernels[-i]), padding='same'))
            layers.append(BatchNorm2d(num_features=self.filters[-i - 1]))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(self.filters[-i-1], 1,
                                kernel_size=(self.kernels[-i], self.kernels[-i]), padding='same'))

        return nn.Sequential(*layers)


    def forward(self, x):
        # Encoder
        encoder_outputs = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)

        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            x = layer(x)

        return x