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
        self.kernel_size_decoder = kwargs.get('kernel_size_decoder', 5)

        self.encoders = []
        self.encoders2D = []
        self.common_encoders = []

        for kernel_size in self.kernels:

            # Encoder layers
            self.encoders.append(self._build_encoder1D(kernel_size))
            self.encoders2D.append(self._build_encoder2D(kernel_size))
            self.common_encoders.append(self._build_common_encoder(kernel_size))

        # Decoder layers
        self.decoder = self._build_decoder()
        self.classifier = self._build_classifier()


    def _build_encoder1D(self, kernel_size):
        layers = []
        for i in range(self.depth - 1):
            layers += self._build_conv_block(i, 1, 1, self.sizes[-1], kernel_size)
            new_layers, pooling1, pooling2 = self.add_pooling(i, 1)
            layers += new_layers
            self.sizes.append(int(self.sizes[-1] // pooling2))

        layers += self._build_conv_block(-1, 1, 1, self.sizes[-1], kernel_size)

        return nn.Sequential(*layers)

    def _build_encoder2D(self, kernel_size):
        layers = []
        for i in range(self.depth - 1):
            layers = self._build_conv_block(i, self.nb_channels_spectro[i], kernel_size, self.sizes[i],
                                           kernel_size)
            new_layers, pooling1, pooling2 = self.add_pooling(i, self.nb_channels_spectro[-1])
            layers += new_layers
            self.nb_channels_spectro.append(int(self.nb_channels_spectro[-1] // pooling1))

        # Last block without maxpooling
        layers += self._build_conv_block(-1, self.nb_channels_spectro[-1], kernel_size,
                                         self.sizes[-1], kernel_size)

        return nn.Sequential(*layers)

    def _build_common_encoder(self, kernel_size):
        layers = []
        layers.append(nn.Conv2d(self.filters[-1] * 2, self.filters[- 1], kernel_size=(1, kernel_size),
                     padding='same'))
        return nn.Sequential(*layers)

    def _build_decoder(self):
        layers = []
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

        for nbr in range(len(self.kernels)):

            # Encoders
            moments, encoder_moments_outputs = self.forward_encoder(initial_moments, self.encoders[nbr])
            spectro, encoder_spectro_outputs = self.forward_encoder(initial_spectro, self.encoders2D[nbr])
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
            for layer in self.common_encoders[nbr]:
                if isinstance(layer, nn.BatchNorm2d):
                    x = self.apply_batchnorm(x, layer)
                else:
                    x = layer(x)

            encoded_outputs.append(x)

        encoded_outputs = torch.tensor(encoded_outputs)
        nb_kernel_sizes,n_batch,n_features,height,width = encoded_outputs.shape
        x = encoded_outputs.transpose(0,1).reshape((n_batch,nb_kernel_sizes*n_features,height,width))  # Vérifier que ça c'est bon

        all_encoder_moments_outputs = torch.tensor(all_encoder_moments_outputs).transpose(0,1)
        n_batch, nb_kernel_sizes, n_features, height, width = all_encoder_moments_outputs.shape
        all_encoder_moments_output = all_encoder_moments_outputs.reshape((n_batch, nb_kernel_sizes*n_features, height, width))
        all_encoder_spectro_outputs = torch.tensor(all_encoder_spectro_outputs).transpose(0,1)
        n_batch, nb_kernel_sizes, n_features, height, width = all_encoder_spectro_outputs.shape
        all_encoder_spectro_outputs = all_encoder_spectro_outputs.reshape((n_batch, nb_kernel_sizes*n_features, height, width))

        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.BatchNorm2d):
                x = self.apply_batchnorm(x, layer)

            elif isinstance(layer, nn.Upsample):
                x = layer(x)

                res_connection_spectro = all_encoder_spectro_outputs.pop()
                res_connection_spectro = torch.mean(res_connection_spectro, dim=2)
                a, b, c = res_connection_spectro.shape
                res_connection_spectro = res_connection_spectro.reshape((a, b, 1, c))
                res_connection_moments = all_encoder_moments_output.pop()

                x = torch.cat([x, res_connection_spectro, res_connection_moments], dim=1)

            else:
                x = layer(x)

        # Dense classification
        for layer in self.classifier:
            x = layer(x)

        return x