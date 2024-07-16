import torch
import torch.nn as nn
from torch.nn import (MaxPool2d, Conv2d, Upsample, BatchNorm2d)
from UTime.EvaluateAndPred import Model
from UTime.CommonArchitecture import Architecture
from UTime.architectureResNet import UTime as UTimeResNet

class UTime(UTimeResNet):
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
        self.encoder = self._build_encoder2D(nb_in_channels=self.nb_moments+1)

        # Decoder layers
        self.decoder = self._build_decoder()

        self.classifier = self._build_classifier()


    def _build_encoder2D(self, **kwargs):
        layers = []
        for i in range(self.depth - 1):
            layers += self._build_conv_block(i, self.nb_channels[i], self.kernels[i],
                                             self.sizes[i], self.kernels[i], **kwargs)
            new_layers, pooling1, pooling2 = self.add_pooling(i, self.nb_channels[-1])
            layers += new_layers
            self.nb_channels.append(int(self.nb_channels[-1] // pooling1))

        # Last block without maxpooling
        layers += self._build_conv_block(-1, self.nb_channels[-1], self.kernels[-1],
                                         self.sizes[-1], self.kernels[-1], **kwargs)

        return nn.Sequential(*layers)


    def _build_decoder(self):
        layers = []
        for i in range(1, self.depth):
            # layers.append(nn.Upsample(scale_factor=(1,2)))
            layers.append(nn.Upsample(size=(1, self.sizes[::-1][i])))
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Conv2d(self.filters[-i] + self.filters[-i - 1], self.filters[-i - 1],
                                    kernel_size=(1, self.kernels[-i]), padding='same'))
            if self.batch_norm:
                layers.append(BatchNorm2d(num_features=self.filters[-i - 1]))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):

        # Encoders
        x, encoder_outputs = self.forward_encoder(x, self.encoder)

        # Squish spectro encoder output in 1D
        ''' These following lines ensure that if there are still several channels in the spectro, 
        it will be transformed into something of the same shape as the moments results, to be concatenated'''
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

            else:
                x = layer(x)

        # Dense classification
        for layer in self.classifier:
            x = layer(x)

        return x
