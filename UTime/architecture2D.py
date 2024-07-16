import torch
import torch.nn as nn
from torch.nn import (MaxPool2d, Conv2d, Upsample, BatchNorm2d)
from UTime.CommonArchitecture import Architecture
from UTime.architectureResNet import UTime as UTimeResNet


class UTime(UTimeResNet):
    def __init__(self, n_classes,
                 n_time,
                 nb_channels,
                 depth,
                 filters,
                 kernels,
                 poolings, **kwargs):

        super(UTime, self).__init__(n_classes, n_time, 0, nb_channels,
                 depth, filters, kernels, poolings, **kwargs)

        '''
        # Encoder layers
        nb_channels_in = kwargs.get('nb_channels_in', 1)
        # (This allows to give several 2D inputs (more than just the spectro))
        self.encoder = self._build_encoder(nb_channels_in)
        # print(self.encoder)
        # Decoder layers
        self.decoder = self._build_decoder()
        # print(self.decoder)

        self.classifier = self._build_classifier()

    def _build_encoder(self, nb_channels_in):
        layers = []
        for i in range(self.depth - 1):
            if i == 0:
                layers.append(
                    nn.Conv2d(nb_channels_in, self.filters[i], kernel_size=(min(self.kernels[i], self.nb_channels[-1]), self.kernels[i]), padding='same'))
            else:
                # print(f'Layer {i}: {self.filters[i-1]} -> {self.filters[i]}, kernels = {self.kernels[i]}')
                layers.append(nn.Conv2d(self.filters[i - 1], self.filters[i],
                                        kernel_size=(min(self.kernels[i], self.nb_channels[-1]), self.kernels[i]),
                                        padding='same'))
            layers.append(BatchNorm2d(num_features=self.filters[i]))
            layers.append(nn.ReLU(inplace=True))

            # layers.append(nn.Conv2d(self.filters[i], self.filters[i], kernel_size = (1,self.kernels[i]),
            # padding='same'))
            # layers.append(BatchNorm2d(num_features = self.filters[i]))
            # layers.append(nn.ReLU(inplace=True))

            layers.append(nn.MaxPool2d(kernel_size=(self.poolings[i], self.poolings[i])))

            self.sizes.append(int(self.sizes[-1] // self.poolings[i]))
            self.nb_channels.append(int(self.nb_channels[-1] // self.poolings[i]))

            # print(f'Layer {i}, maxpooling: {self.poolings[i]}')

        # Last block without maxpooling
        # print(f'Last layer {i}: {self.filters[-2]} -> {self.filters[-1]}, kernels = {self.kernels[-1]}')
        layers.append(nn.Conv2d(self.filters[-2], self.filters[-1], kernel_size=(min(self.kernels[-1], self.nb_channels[-1]), self.kernels[-1]),
                                padding='same'))
        layers.append(BatchNorm2d(num_features=self.filters[-1]))
        layers.append(nn.ReLU(inplace=True))

        # layers.append(nn.Conv2d(self.filters[-1], self.filters[-1], kernel_size = (1,self.kernels[-1]),
        # padding='same'))
        # layers.append(BatchNorm2d(num_features = self.filters[i]))
        # layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)
    '''

    def _build_decoder(self):
        layers = []
        for i in range(1, self.depth):
            # layers.append(nn.Upsample(scale_factor=(1,2)))
            layers.append(nn.Upsample(size=(1, self.sizes[::-1][i])))
            layers.append(nn.Conv2d(self.filters[-i] + self.filters[-i - 1], self.filters[-i - 1],
                                    kernel_size=(1, self.kernels[-i]), padding='same'))
            layers.append(BatchNorm2d(num_features=self.filters[-i - 1]))
            layers.append(nn.ReLU(inplace=True))

            # layers.append(nn.Conv2d(self.filters[-i-1], self.filters[-i-1], kernel_size=(1,self.kernels[-i]),
            # padding='same'))
            # layers.append(BatchNorm1d(num_features = self.filters[-i-1]))
            # layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _build_classifier(self):
        layers = [Conv2d(self.filters[0], self.n_classes, kernel_size=(1, 1))]
        layers.append(BatchNorm2d(num_features=self.n_classes))

        if self.n_classes == 1:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Softmax(dim=1))

        return nn.Sequential(*layers)


    '''
    def forward(self, x):
        # Encoder
        encoder_outputs = []
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.MaxPool2d):
                encoder_outputs.append(x)
            x = layer(x)

        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.Upsample):
                x = layer(x)

                res_connection = encoder_outputs.pop()
                res_connection = torch.mean(res_connection, dim=2)
                a, b, c = res_connection.shape
                res_connection = res_connection.reshape((a, b, 1, c))
                x = torch.cat([x, res_connection], dim=1)

            else:
                x = layer(x)

        # Dense classification
        for layer in self.classifier:
            x = layer(x)

        return x
        '''

    def forward(self, x):
        # Encoder
        x, encoder_outputs = self.forward_encoder(x, self.encoder2D)
        x = torch.mean(x, dim=2)
        a, b, c = x.shape
        x = x.reshape((a, b, 1, c))

        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.BatchNorm2d):
                x = self.apply_batchnorm(x, layer)

            elif isinstance(layer, nn.Upsample):
                x = layer(x)

                res_connection = encoder_outputs.pop()
                res_connection = torch.mean(res_connection, dim=2)
                a, b, c = res_connection.shape
                res_connection = res_connection.reshape((a, b, 1, c))

                x = torch.cat([x, res_connection], dim=1)

            elif isinstance(layer, nn.Conv2d):
                x, residual = self.forward_conv(x, layer)

            elif isinstance(layer, nn.ReLU):
                x = (layer(x) + residual)/2

            elif isinstance(layer, nn.Dropout):
                x = layer(x)

            else:
                raise Exception("I forgot a kind of possible layer in the Decoder")


        # Dense classification
        for layer in self.classifier:
            x = layer(x)

        return x
