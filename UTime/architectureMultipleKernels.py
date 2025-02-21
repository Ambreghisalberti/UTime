import torch
import torch.nn as nn
from torch.nn import (MaxPool2d, Conv2d, Upsample, BatchNorm2d)
from UTime.architectureResNet import UTime as UTimeResNet
import numpy as np

class UTime(UTimeResNet):
    def __init__(self, n_classes,
                 n_time,
                 nb_moments,
                 nb_channels_spectro,
                 depth,
                 filters,
                 kernels,
                 poolings, dilatations,
                 **kwargs):

        self.kernel_size_decoder = kwargs.get('kernel_size_decoder', 5)
        self.dilatations = dilatations
        super(UTime, self).__init__(n_classes, n_time, nb_moments, nb_channels_spectro, depth,
                 filters, kernels, poolings, **kwargs)


    def _build_architecture(self, **kwargs):

        for dilatation in self.dilatations:

            # Encoder layers
            self.__setattr__(f'encoder1D_{dilatation}', self._build_encoder1D(kernel_sizes=self.kernels,
                                                                              dilatation=dilatation).double().to(self.device))
            self.__setattr__(f'encoder2D_{dilatation}', self._build_encoder2D(kernel_sizes=self.kernels,
                                                                              dilatation=dilatation).double().to(self.device))
            self.__setattr__(f'commonEncoder_{dilatation}', self._build_common_encoder(self.kernels,
                                                                                       dilatation=dilatation).double().to(self.device))

        # Decoder layers
        self.decoder = self._build_decoder()
        self.classifier = self._build_classifier()


    def _build_common_encoder(self, kernel_size, **kwargs):
        dilatation = kwargs.get('dilatation',1)
        layers = []
        layers.append(nn.Conv2d(self.filters[-1] * 2, self.filters[- 1], kernel_size=(1, kernel_size),
                     padding='same', dilatation=dilatation))
        return nn.Sequential(*layers).double().to(self.device)


    def _build_decoder(self):
        layers = []
        layers.append(nn.Conv2d(self.filters[-1] * len(self.kernels), self.filters[- 1],
                                kernel_size=(1, self.kernel_size_decoder), padding='same'))
        if self.batch_norm:
            layers.append(BatchNorm2d(num_features=self.filters[- 1]))
        layers.append(nn.ReLU(inplace=True))

        for i in range(1, self.depth):
            # layers.append(nn.Upsample(scale_factor=(1,2)))
            layers.append(nn.Upsample(size=(1, self.sizes[::-1][i])))
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Conv2d(self.filters[-i] + 2 * self.filters[-i - 1] * len(self.kernels), self.filters[-i - 1],
                                    kernel_size=(1, self.kernel_size_decoder), padding='same'))
            if self.batch_norm:
                layers.append(BatchNorm2d(num_features=self.filters[-i - 1]))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        initial_moments, initial_spectro = x
        encoded_outputs = []
        all_encoder_moments_outputs = []
        all_encoder_spectro_outputs = []

        for kernel in self.kernels:
            encoder1D = getattr(self, f'encoder1D_{kernel}')
            encoder2D = getattr(self, f'encoder2D_{kernel}')
            commonEncoder = getattr(self, f'commonEncoder_{kernel}')

            # Encoders
            moments, encoder_moments_outputs = self.forward_encoder(initial_moments, encoder1D.double().to(self.device))
            spectro, encoder_spectro_outputs = self.forward_encoder(initial_spectro, encoder2D.double().to(self.device))
            all_encoder_moments_outputs.append(encoder_moments_outputs)
            all_encoder_spectro_outputs.append(encoder_spectro_outputs)

            # Squish spectro encoder output in 1D
            ''' These following lines ensure that if there are still several channels in the spectro, 
            it will be transformed into something of the same shape as the moments results, to be concatenated'''
            spectro = torch.mean(spectro, dim=2)
            a, b, c = spectro.shape
            spectro = spectro.reshape((a, b, 1, c))

            # Concatenate moments and spectro encoder info
            x = torch.cat([spectro, moments], dim=1).double()
            # This version is only for the case where common encoder is just one convolution, not keeping encoding!
            for layer in commonEncoder:
                if isinstance(layer, nn.BatchNorm2d):
                    x = self.apply_batchnorm(x, layer)
                else:
                    x = layer(x)

            encoded_outputs.append(torch.Tensor.cpu(x).detach().numpy())

        encoded_outputs = torch.tensor(np.array(encoded_outputs)).to(self.device)
        nb_kernel_sizes,n_batch,n_features,height,width = encoded_outputs.shape
        x = encoded_outputs.transpose(0,1).reshape((n_batch,nb_kernel_sizes*n_features,height,width))  # Vérifier que ça c'est bon

        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.BatchNorm2d):
                x = self.apply_batchnorm(x, layer)

            elif isinstance(layer, nn.Upsample):
                x = layer(x)

                for nbr in range(len(self.kernels)):
                    res_connection_spectro = all_encoder_spectro_outputs[nbr].pop()
                    res_connection_spectro = torch.mean(res_connection_spectro, dim=2)
                    a, b, c = res_connection_spectro.shape
                    res_connection_spectro = res_connection_spectro.reshape((a, b, 1, c))
                    res_connection_moments = all_encoder_moments_outputs[nbr].pop()     # Check that it actually removes the element!

                    x = torch.cat([x, res_connection_spectro, res_connection_moments], dim=1)

            else:
                x = layer(x)

        # Dense classification
        for layer in self.classifier:
            x = layer(x)

        return x
