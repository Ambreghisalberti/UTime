import torch
import torch.nn as nn
from torch.nn import (MaxPool2d, Conv2d, Upsample, BatchNorm2d)
from .EvaluateAndPred import Model
from UTime.architecture import Architecture


class UTime(Architecture):
    def __init__(self, n_classes,
                 n_time,
                 nb_moments,
                 depth,
                 filters,
                 kernels,
                 poolings, **kwargs):

        super(UTime, self).__init__(n_classes, n_time, nb_moments, 0, depth,
                                           filters, kernels, poolings, **kwargs)

        # Encoder layers
        self.encoder = self._build_encoder1D()
        # Decoder layers
        self.decoder = self._build_decoder()
        self.classifier = self._build_classifier()

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

        # Encoder
        x, encoder_outputs = self.forward_encoder(x, self.encoder)

        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.BatchNorm2d):
                x = self.apply_batchnorm(x, layer)

            elif isinstance(layer, nn.Upsample):
                x = layer(x)
                res_connection = encoder_outputs.pop()
                x = torch.cat([x, res_connection], dim=1)
            else:
                x = layer(x)

        # Dense classification
        for layer in self.classifier:
            x = layer(x)

        return x
