import torch
import torch.nn as nn
from torch.nn import (MaxPool2d, Conv2d, Upsample, BatchNorm2d)
from UTime.EvaluateAndPred import Model
from UTime.architecture import get_kernel_size, get_pooling_size, apply_batchnorm
from UTime.architecture import Architecture

class UTime(nn.Module, Model, UTime):
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
        self.nb_blocks_per_layer = kwargs.get('nb_blocks_per_layer',1)

        # Encoder layers
        self.encoder = self._build_encoder1D()
        self.encoder2D = self._build_encoder2D()
        # print(self.encoder)
        # print(self.encoder2D)

        # Decoder layers
        self.decoder = self._build_decoder()
        # print(self.decoder)

        self.classifier = self._build_classifier()


    def _build_encoder1D(self):
        layers = []
        for i in range(self.depth - 1):
            for iter in range(self.nb_blocks_per_layer):
                if i == 0:
                    layers.append(
                        nn.Conv2d(self.nb_moments, self.filters[i], kernel_size=(1, self.kernels[i]), padding='same'))
                else:
                    layers.append(
                        nn.Conv2d(self.filters[i - 1], self.filters[i], kernel_size=(1, self.kernels[i]), padding='same'))
                if self.batch_norm:
                    layers.append(BatchNorm2d(num_features=self.filters[i]))

                for j in range(self.nb_blocks_per_layer-1):
                    layers.append(
                        nn.Conv2d(self.filters[i], self.filters[i], kernel_size=(1, self.kernels[i]), padding='same'))
                    if self.batch_norm:
                        layers.append(BatchNorm2d(num_features=self.filters[i]))

                layers.append(nn.ReLU(inplace=True))

            new_layers, pooling1, pooling2 = self.add_pooling(self, i, 1)
            layers.append(new_layers)
            self.sizes.append(self.sizes[-1]//pooling2)

        layers.append(nn.Conv2d(self.filters[-2], self.filters[-1], kernel_size=(self.kernels[-1], self.kernels[-1]),
                                padding='same'))
        if self.batch_norm:
            layers.append(BatchNorm2d(num_features=self.filters[-1]))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)


    def _build_encoder2D(self):
        layers = []
        for i in range(self.depth - 1):
            for iter in range(self.nb_blocks_per_layer):
                kernel_size1 = get_kernel_size(self.nb_channels_spectro[-1], self.kernels[i])
                kernel_size2 = get_kernel_size(self.sizes[i] , self.kernels[i])
                layers.append(self._build_conv_block(i, kernel_size1, kernel_size2))

            new_layers, pooling1, pooling2 = self.add_pooling(self, i, self.nb_channels_spectro[-1])
            layers.append(new_layers)
            self.nb_channels_spectro.append(int(self.nb_channels_spectro[-1] // pooling1))

        # Last block without maxpooling
        kernel_size1 = min(self.kernels[-1], self.nb_channels_spectro[-1])
        kernel_size2 = get_kernel_size(self.sizes[i], self.kernels[i])
        layers.append(self._build_conv_block(-1, kernel_size1, kernel_size2))

        if self.batch_norm:
            layers.append(BatchNorm2d(num_features=self.filters[-1]))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _build_decoder(self):
        layers = []
        for i in range(1, self.depth):
            # layers.append(nn.Upsample(scale_factor=(1,2)))
            layers.append(nn.Upsample(size=(1, self.sizes[::-1][i])))

            for iter in range(self.nb_blocks_per_layer):
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
            if isinstance(layer, nn.BatchNorm2d):
                a, b, c, d = moments.shape
                if c * d > 1:
                    moments = layer(moments)

            elif isinstance(layer, nn.MaxPool2d):
                encoder_moments_outputs.append(moments)
                moments = layer(moments)

            elif isinstance(layer, nn.Conv2d):
                residual = moments
                _,nb,_,_ = residual.shape
                moments = layer(moments)
                _,nb2,_,_ = moments.shape
                dimension_matching_conv = nn.Conv2d(nb, nb2, kernel_size=(1, 1), padding='same').double().to(self.device)
                residual = dimension_matching_conv(residual)
                if self.batch_norm:
                    norm = BatchNorm2d(num_features=nb2).double().to(self.device)
                    residual = norm(residual)

            elif isinstance(layer, nn.ReLU):
                moments = (layer(moments)+residual)/2

            else:
                raise Exception("I forgot a kind of possible layer in the Encoder1d")

        # Encoder 2D
        encoder_spectro_outputs = []
        for i, layer in enumerate(self.encoder2D):
            if isinstance(layer, nn.BatchNorm2d):
                a, b, c, d = spectro.shape
                if c * d > 1:
                    spectro = layer(spectro)

            elif isinstance(layer, nn.MaxPool2d):
                encoder_spectro_outputs.append(spectro)
                spectro = layer(spectro)

            elif isinstance(layer, nn.Conv2d):
                residual = spectro
                _, nb, _, _ = residual.shape
                spectro = layer(spectro)
                _, nb2, _, _ = spectro.shape
                dimension_matching_conv = nn.Conv2d(nb, nb2, kernel_size=(1, 1), padding='same').double().to(self.device)
                residual = dimension_matching_conv(residual)
                if self.batch_norm:
                    norm = BatchNorm2d(num_features=nb2).double().to(self.device)
                    residual = norm(residual)

            elif isinstance(layer, nn.ReLU):
                spectro = (layer(spectro) + residual)/2

            else:
                raise Exception("I forgot a kind of possible layer in the Encoder2d")

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
            if isinstance(layer, nn.BatchNorm2d):
                a, b, c, d = x.shape
                if c * d > 1:
                    x = layer(x)

            elif isinstance(layer, nn.Upsample):
                x = layer(x)

                res_connection_spectro = encoder_spectro_outputs.pop()
                res_connection_spectro = torch.mean(res_connection_spectro, dim=2)
                a, b, c = res_connection_spectro.shape
                res_connection_spectro = res_connection_spectro.reshape((a, b, 1, c))
                res_connection_moments = encoder_moments_outputs.pop()

                x = torch.cat([x, res_connection_spectro, res_connection_moments], dim=1)

            elif isinstance(layer, nn.Conv2d):
                residual = x
                _, nb, _, _ = residual.shape
                x = layer(x)
                _, nb2, _, _ = x.shape
                dimension_matching_conv = nn.Conv2d(nb, nb2, kernel_size=(1, 1), padding='same').double().to(self.device)
                residual = dimension_matching_conv(residual)
                if self.batch_norm:
                    norm = BatchNorm2d(num_features=nb2).double().to(self.device)
                    residual = norm(residual)

            elif isinstance(layer, nn.ReLU):
                x = (layer(x) + residual)/2

            else:
                raise Exception("I forgot a kind of possible layer in the Decoder")


        # Dense classification
        for layer in self.classifier:
            x = layer(x)

        return x
