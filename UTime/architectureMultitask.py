import torch
import torch.nn as nn
from torch.nn import (MaxPool2d, Conv2d, Upsample, BatchNorm2d)
from UTime.EvaluateAndPred import Model
from UTime.CommonArchitecture import Architecture

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

        # Encoder layers
        self.encoder = self._build_encoder1D()
        self.encoder2D = self._build_encoder2D()
        self.common_encoder = self._build_common_encoder()

        # Decoder layers
        self.decoder = self._build_decoder()

        self.classifiers = [self._build_classifier(nb_classes_classifier=1) for i in range(n_classes)]


    def _build_encoder2D(self):
        layers = []
        for i in range(self.depth - 1):
            layers += self._build_conv_block(i, self.nb_channels_spectro[i], self.kernels[i], self.sizes[i],
                                           self.kernels[i])
            new_layers, pooling1, pooling2 = self.add_pooling(i, self.nb_channels_spectro[-1])
            layers += new_layers
            self.nb_channels_spectro.append(int(self.nb_channels_spectro[-1] // pooling1))

        # Last block without maxpooling
        layers += self._build_conv_block(-1, self.nb_channels_spectro[-1], self.kernels[-1],
                                         self.sizes[-1], self.kernels[-1])

        return nn.Sequential(*layers)

    def _build_common_encoder(self):
        layers = []
        layers.append(nn.Conv2d(self.filters[-1] * 2, self.filters[- 1], kernel_size=(1, self.kernels[-1]),
                     padding='same'))
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

    def forward(self, x):
        moments, spectro = x

        # Encoders
        moments, encoder_moments_outputs = self.forward_encoder(moments, self.encoder)
        spectro, encoder_spectro_outputs = self.forward_encoder(spectro, self.encoder2D)

        # Squish spectro encoder output in 1D
        ''' These following lines ensure that if there are still several channels in the spectro, 
        it will be transformed into something of the same shape as the moments results, to be concatenated'''
        spectro = torch.mean(spectro, dim=2)
        a, b, c = spectro.shape
        spectro = spectro.reshape((a, b, 1, c))

        # Concatenate moments and spectro encoder info
        x = torch.cat([spectro, moments], dim=1).double()
        # This version is only for the case where common encoder is just one convolution, not keeping encoding!
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
                res_connection_spectro = torch.mean(res_connection_spectro, dim=2)
                a, b, c = res_connection_spectro.shape
                res_connection_spectro = res_connection_spectro.reshape((a, b, 1, c))
                res_connection_moments = encoder_moments_outputs.pop()

                x = torch.cat([x, res_connection_spectro, res_connection_moments], dim=1)

            else:
                x = layer(x)

        # Dense classification
        outs = []
        for classifier in self.classifiers:
            out = x
            for layer in classifier:
                out = layer(out)
            outs.append(out)
        outs = torch(outs)

        return outs
