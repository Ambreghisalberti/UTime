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

        super(Architecture, self).__init__(n_classes, n_time, nb_moments, 0, depth,
                                           filters, kernels, poolings, **kwargs)

        # Encoder layers
        self.encoder = self._build_encoder1D()
        # Decoder layers
        self.decoder = self._build_decoder()
        self.classifier = self._build_classifier()


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
