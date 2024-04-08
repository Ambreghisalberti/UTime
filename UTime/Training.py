import numpy as np
from datetime import datetime
import torch
import time
import matplotlib.pyplot as plt
import torch.optim as optim
from .CostFunctions import WeightedMSE, WeightedBCE
from .EarlyStopping import EarlyStopping


class Training():

    def __init__(self, model, epochs, dltrain, **kwargs):
        self.model = model
        self.epochs = epochs
        self.current_epoch = 0
        self.dltrain = dltrain
        self.dltest = kwargs.get('dltest')
        self.training_loss = []
        self.mirrored = kwargs.get("mirrored", True)

        self.validation = kwargs.get('validation', False)
        self.dlval = kwargs.get('dlval')
        self.val_loss = []
        self.lr = kwargs.get('lr', 0.001)
        self.optimizer = kwargs.get('optimizer', optim.Adam(self.model.parameters(), self.lr))

        self.verbose = kwargs.get('verbose', False)
        self.verbose_plot = kwargs.get('verbose_plot', False)

        # self.train_criterion = kwargs.get('train_criterion',nn.CrossEntropyLoss(reduction='mean'))
        # Look into it. Rather choose a weighted function later.
        # self.val_criterion = kwargs.get('val_criterion',nn.CrossEntropyLoss(reduction='mean'))
        # Look into it. Rather choose a weighted function later.
        # self.test_criterion = kwargs.get('test_criterion',nn.CrossEntropyLoss(reduction='mean'))
        # Look into it. Rather choose a weighted function later.

        self.train_criterion = kwargs.get('train_criterion', WeightedMSE(self.dltrain))
        if self.validation:
            self.val_criterion = kwargs.get('val_criterion', WeightedMSE(self.dlval))
        self.test_criterion = kwargs.get('test_criterion', WeightedMSE(self.dltest))

    def backward_propagation(self, batch, labels):
        self.optimizer.zero_grad()
        outputs = self.model.forward(batch.double())
        loss = self.train_criterion(torch.flatten(outputs), torch.flatten(labels)).double()

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()

        return loss.detach()

    def epoch(self, dl, **kwargs):
        # Training
        time1_seconds = time.time()
        self.model.train()
        loss = 0
        count = 0
        for i, inputs, labels in dl:
            loss += self.backward_propagation(inputs.double(), labels.double()).double()
            count += 1

        if self.mirrored:
            for i, inputs, labels in dl:
                loss += self.backward_propagation(inputs.flip(-1).double(), labels.flip(-1).double()).double()
                count += 1

        self.training_loss.append(loss / count)  # On one batch
        self.current_epoch += 1
        time2_seconds = time.time()

        verbose = kwargs.get("verbose", False)
        if verbose & (self.current_epoch % 10 == 0):
            print(
                f'Epoch [{self.current_epoch}/{self.epochs}], Loss: {self.training_loss[-1]:.4f}, took '
                f'{time2_seconds - time1_seconds} seconds')

        if self.validation:
            self.val_loss.append(self.model.evaluate(self.dlval, self.val_criterion, mirrored=self.mirrored))

    def fit(self, **kwargs):
        self.current_epoch = 0
        self.verbose = kwargs.get("verbose", False)
        early_stop = kwargs.get('early_stop', False)
        t_begin = time.time()

        if early_stop:
            name = kwargs.get('name', str(datetime.now())[:10])
            patience = kwargs.get('patience', 10)
            early_stopping = EarlyStopping(self.model,
                                           patience=patience,
                                           path=f"/home/ghisalberti/BL_encoder_decoder/checkpoints_models/{name}.pt",
                                           verbose=self.verbose)

            while (self.current_epoch < self.epochs) & (early_stopping.early_stop == False):
                self.epoch(self.dltrain, **kwargs)
                early_stopping(self.val_loss[-1], self.model)

        else:
            for e in range(self.epochs):
                self.epoch(self.dltrain, **kwargs)

        t_end = time.time()
        if self.verbose_plot:
            print(f"Total training done in {t_end - t_begin} seconds and {self.current_epoch} epochs.")

            plt.figure()
            plt.plot(np.arange(self.current_epoch), torch.tensor(self.training_loss).detach().numpy(),
                     label='Trainset')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            name = kwargs.get('name', str(datetime.now())[:10])
            plt.title(f'{name}\nnum_epochs = {self.current_epoch}.')
            if self.validation:
                plt.plot(np.arange(self.current_epoch), torch.tensor(self.val_loss).detach().numpy(),
                         label='Validation set')
            if early_stop:
                plt.axvline(early_stopping.stop_epoch, linestyle='--', color='r', label='Stopping Checkpoint')
            plt.legend()
