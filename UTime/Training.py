import numpy as np
from datetime import datetime
import torch
import time
import matplotlib.pyplot as plt
import torch.optim as optim
from .CostFunctions import WeightedMSE, WeightedBCE
from .EarlyStopping import EarlyStopping
from torch.nn import MSELoss
from IPython import display


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

        self.model = self.model.to(self.model.device)

        self.train_criterion = kwargs.get('train_criterion', MSELoss())
        if self.validation:
            self.val_criterion = kwargs.get('val_criterion', MSELoss())
        self.test_criterion = kwargs.get('test_criterion', MSELoss())

        self.name = kwargs.get('name', str(datetime.now())[:10])


    def backward_propagation(self, batch, labels):
        if isinstance(batch, tuple):
            a,b = batch
            batch = (a.double().to(self.model.device), b.double().to(self.model.device))
        else:
            batch = batch.to(self.model.device)
        labels = labels.to(self.model.device)
        self.optimizer.zero_grad()
        outputs = self.model.forward(batch)
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
            if isinstance(inputs, list):
                a,b = inputs[0], inputs[1]
                inputs = a.double(), b.double()
            else:
                inputs = inputs.double()
            loss += self.backward_propagation(inputs, labels.double()).double()
            count += 1

        if self.mirrored:
            for i, inputs, labels in dl:
                if isinstance(inputs, list):
                    a, b = inputs[0], inputs[1]
                    flipped_inputs = a.flip(-1).double(), b.flip(-1).double()
                else:
                    flipped_inputs = inputs.flip(-1).double()
                loss += self.backward_propagation(flipped_inputs, labels.flip(-1).double()).double()
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

        if self.verbose_plot:
            self.info()


    def fit(self, **kwargs):
        self.current_epoch = 0
        self.verbose = kwargs.get("verbose", False)
        early_stop = kwargs.get('early_stop', False)
        t_begin = time.time()

        if self.verbose_plot:
            if 'fig' in kwargs and 'ax' in kwargs:
                fig = kwargs['fig']
                ax = kwargs['ax']
            else:
                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5,5))

        if early_stop:
            patience = kwargs.get('patience', 10)
            early_stopping = EarlyStopping(self.model,
                                           patience=patience,
                                           path=f"/home/ghisalberti/BL_encoder_decoder/checkpoints_models/{self.name}.pt",
                                           verbose=self.verbose)

            while (self.current_epoch < self.epochs) & (early_stopping.early_stop == False):
                self.epoch(self.dltrain, **kwargs)
                early_stopping(self.val_loss[-1], self.model)
            self.stop_epoch = early_stopping.stop_epoch

        else:
            for e in range(self.epochs):
                self.epoch(self.dltrain, **kwargs)
            self.stop_epoch = self.epochs

        t_end = time.time()
        if self.verbose_plot:
            print(f"Total training done in {t_end - t_begin} seconds and {self.stop_epoch} epochs.")
            if early_stop:
                self.info(early_stopping=early_stopping, fig=fig, ax=ax)
            else:
                self.info(fig=fig, ax=ax)
            plt.show()

    def info(self, **kwargs):
        if 'fig' in kwargs and 'ax' in kwargs:
            fig,ax = kwargs['fig'], kwargs['ax']
        else:
            fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(5,5))

        #plt.cla()
        ax.plot(np.arange(self.current_epoch), torch.tensor(self.training_loss).detach().numpy(),
                 label='Trainset')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.title.set_text(f'{self.name}\nnum_epochs = {self.current_epoch}.')

        if self.validation:
            ax.plot(np.arange(self.current_epoch), torch.tensor(self.val_loss).detach().numpy(),
                     label='Validation set')

        if 'early_stopping' in kwargs:
            early_stopping = kwargs['early_stopping']
            ax.axvline(early_stopping.stop_epoch, linestyle='--', color='r', label='Stopping Checkpoint')
            ax.title.set_text(f'{self.name}\nnum_epochs = {early_stopping.stop_epoch}.')
        plt.legend()
        #display.clear_output(wait=True)
        #display.display(plt.gcf())
        #display.display(fig)
