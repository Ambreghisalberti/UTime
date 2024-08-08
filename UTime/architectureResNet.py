import torch
import torch.nn as nn
from torch.nn import (MaxPool2d, Conv2d, Upsample, BatchNorm2d)
from UTime.EvaluateAndPred import Model
from UTime.CommonArchitecture import Architecture
import numpy as np

class UTime(Architecture):
    def __init__(self, n_classes,
                 n_time,
                 nb_moments,
                 nb_channels_spectro,
                 depth,
                 filters,
                 kernels,
                 poolings,
                 **kwargs):

        super(UTime, self).__init__(n_classes, n_time, nb_moments, nb_channels_spectro, depth,
                                           filters, kernels, poolings, **kwargs)

        self._build_architecture(**kwargs)

    def _build_architecture(self, **kwargs):
        # Encoder layers
        self.encoder = self._build_encoder1D()
        self.encoder2D = self._build_encoder2D(**kwargs)
        self.common_encoder = self._build_common_encoder(**kwargs)

        # Decoder layers
        self.decoder = self._build_decoder()

        self.classifier = self._build_classifier()


    def _build_encoder1D(self, **kwargs):
        self.sizes = [self.n_time]
        kernel_sizes = kwargs.get('kernel_sizes', self.kernels)
        layers = []
        for i in range(self.depth - 1):
            for iter in range(self.nb_blocks_per_layer):
                layers += self._build_conv_block(i, 1, 1, self.sizes[-1], kernel_sizes[i])

            new_layers, pooling1, pooling2 = self.add_pooling(i, 1)
            layers += new_layers
            self.sizes.append(self.sizes[-1]//pooling2)

        layers += self._build_conv_block(-1, 1, 1, self.sizes[-1], kernel_sizes[-1])

        return nn.Sequential(*layers).to(self.device)

    '''
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
    '''

    def _build_decoder(self, **kwargs):
        kernel_sizes = kwargs.get('kernel_sizes', self.kernels)
        layers = []
        for i in range(1, self.depth):
            # layers.append(nn.Upsample(scale_factor=(1,2)))
            layers.append(nn.Upsample(size=(1, self.sizes[::-1][i])))

            for iter in range(self.nb_blocks_per_layer):
                layers.append(nn.Conv2d(self.filters[-i] + 2 * self.filters[-i - 1], self.filters[-i - 1],
                                        kernel_size=(1, kernel_sizes[-i]), padding='same'))
                if self.batch_norm:
                    layers.append(BatchNorm2d(num_features=self.filters[-i - 1]))
                layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)


    def forward_conv(self, x, layer):
        residual = x
        _, nb, _, _ = residual.shape
        x = layer(x)
        _, nb2, _, _ = x.shape
        dimension_matching_conv = nn.Conv2d(nb, nb2, kernel_size=(1, 1), padding='same').double().to(
            self.device)
        residual = dimension_matching_conv(residual)
        if self.batch_norm:
            norm = BatchNorm2d(num_features=nb2).double().to(self.device)  # Replace by apply_batchnorm?
            residual = norm(residual)

        return x, residual

    def forward_encoder(self, x, encoder):
        encoder_outputs = []
        for i, layer in enumerate(encoder):
            if isinstance(layer, nn.BatchNorm2d):
                x = self.apply_batchnorm(x, layer)

            elif isinstance(layer, nn.MaxPool2d):
                # encoder_outputs.append(x.detach().numpy())
                encoder_outputs.append(x)
                x = layer(x)

            elif isinstance(layer, nn.Conv2d):
                x, residual = self.forward_conv(x, layer)

            elif isinstance(layer, nn.ReLU):
                x = (layer(x) + residual) / 2

            elif isinstance(layer, nn.Dropout):
                x = layer(x)

            else:
                raise Exception("I forgot a kind of possible layer in the Encoder")

        #return x, torch.Tensor(encoder_outputs)
        return x, encoder_outputs

    def forward(self, x):
        moments, spectro = x

        # Encoder
        moments, encoder_moments_outputs = self.forward_encoder(moments, self.encoder)
        spectro, encoder_spectro_outputs = self.forward_encoder(spectro, self.encoder2D)

        # Concatenate moments and spectro encoder info
        spectro = torch.mean(spectro, dim=2)
        ''' These following lines ensure that if there are still severl channels in the spectro, 
        it will be transformed into something of the same shape as the moments resultss, to be concatenated'''
        a, b, c = spectro.shape
        spectro = spectro.reshape((a, b, 1, c))
        x = torch.cat([spectro, moments], dim=1).double()

        '''
        conv = nn.Conv2d(self.filters[-1] * 2, self.filters[- 1], kernel_size=(1, self.kernels[-1]),
                         padding='same').double().to(self.device)
        x = conv(x.double())
        '''
        # Common encoder
        for layer in self.common_encoder:
            if isinstance(layer, nn.BatchNorm2d):
                x = self.apply_batchnorm(x, layer)
            else:
                x = layer(x)


        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.BatchNorm2d):
                x = self.apply_batchnorm(x, layer)

            elif isinstance(layer, nn.Upsample):
                x = layer(x)

                res_connection_spectro = encoder_spectro_outputs.pop()
                # res_connection_spectro = torch.Tensor(encoder_spectro_outputs.pop())
                res_connection_spectro = torch.mean(res_connection_spectro, dim=2)
                a, b, c = res_connection_spectro.shape
                res_connection_spectro = res_connection_spectro.reshape((a, b, 1, c))
                res_connection_moments = encoder_moments_outputs.pop()
                # res_connection_moments = torch.Tensor(encoder_moments_outputs.pop())

                x = torch.cat([x, res_connection_spectro, res_connection_moments], dim=1)

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
