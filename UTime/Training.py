import numpy as np
from datetime import datetime
import torch, gc
import time
import matplotlib.pyplot as plt
import torch.optim as optim
from .CostFunctions import WeightedMSE, WeightedBCE
from .EarlyStopping import EarlyStopping
from torch.nn import MSELoss
from IPython import display
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import auc

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
        self.weight_decay = kwargs.get('weight_decay', 0)
        self.optimizer = kwargs.get('optimizer', optim.Adam(self.model.parameters(), self.lr,
                                                            weight_decay=self.weight_decay))
        self.variable_lr = kwargs.get('variable_lr',False)
        if self.variable_lr :
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)

        self.verbose = kwargs.get('verbose', False)
        self.verbose_plot = kwargs.get('verbose_plot', False)

        self.model = self.model.to(self.model.device)

        self.train_criterion = kwargs.get('train_criterion', MSELoss())
        if self.validation:
            self.val_criterion = kwargs.get('val_criterion', MSELoss())
        self.test_criterion = kwargs.get('test_criterion', MSELoss())

        self.name = kwargs.get('name', str(datetime.now())[:10])
        self.make_movie = kwargs.get('make_movie', False)

    def backward_propagation(self, batch, labels, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()

        if isinstance(batch, tuple):
            a,b = batch
            batch = (a.double().to(self.model.device), b.double().to(self.model.device))
        else:
            batch = batch.to(self.model.device)
        labels = labels.to(self.model.device)
        self.optimizer.zero_grad()
        outputs = self.model.forward(batch)

        method = kwargs.get('method','flatten')

        if method == 'flatten':
            loss = self.train_criterion(torch.flatten(outputs), torch.flatten(labels)).double()
        elif method == 'nothing':
            loss = self.train_criterion(outputs, labels).double()
        elif method == 'by_class':
            ''' Here make a modification to consider multiclass or multitask prediction? 
            For example, making the sum of train_criterion for preds of different classes, 
            so they all have the same order of magnitude of participation in the loss, not depending 
            on the number of points in each class.
            '''
            _, nb_classes, _, _ = outputs.shape
            weights = kwargs.get('weights', [1/nb_classes for i in range(nb_classes)])
            loss = 0

            for i in range(nb_classes):
                outs = outputs[:,i,:,:]
                labs = labels[:,i,:,:]
                loss += weights[i]*self.train_criterion(torch.flatten(outs), torch.flatten(labs)).double()
            loss = loss

        else:
            raise Exception('Method should be flatten, nothing, or by_class.')

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
            loss += self.backward_propagation(inputs, labels.double(), **kwargs).double()
            count += 1

        if self.mirrored:
            for i, inputs, labels in dl:
                if isinstance(inputs, list):
                    a, b = inputs[0], inputs[1]
                    flipped_inputs = a.flip(-1).double(), b.flip(-1).double()
                else:
                    flipped_inputs = inputs.flip(-1).double()
                loss += self.backward_propagation(flipped_inputs, labels.flip(-1).double(),**kwargs).double()
                count += 1

        self.training_loss.append(loss / count)  # On one batch
        self.current_epoch += 1
        time2_seconds = time.time()

        verbose = kwargs.get("verbose", False)
        if verbose & (self.current_epoch % 10 == 0):
            print(
                f'Epoch [{self.current_epoch}/{self.epochs}], Loss: {self.training_loss[-1]:.4f}, took '
                f'{round(time2_seconds - time1_seconds, 2)} seconds')

        if self.validation:
            self.val_loss.append(self.model.evaluate(self.dlval, self.val_criterion, mirrored=self.mirrored))

        if self.verbose_plot:
            fig, ax = kwargs.pop('fig'), kwargs.pop('ax')
            self.info(fig=fig, ax=ax, **kwargs)

        if self.variable_lr:
            self.scheduler.step()


    def fit(self, **kwargs):

        self.current_epoch = 0
        self.verbose = kwargs.get("verbose", False)
        early_stop = kwargs.get('early_stop', False)
        t_begin = time.time()
        label = kwargs.pop('label', True)

        if self.verbose_plot:
            '''
            if 'fig' in kwargs and 'ax' in kwargs:
                fig = kwargs.pop('fig')
                ax = kwargs.pop('ax')
            else:
                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(3,3))
            '''
            if 'fig' not in kwargs or 'ax' not in kwargs:
                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(3,3))
                kwargs['fig'], kwargs['ax'] = fig, ax

        if early_stop:
            patience = kwargs.get('patience', 10)
            early_stopping = EarlyStopping(self.model,
                                           patience=patience,
                                           path=f"/home/ghisalberti/BL_encoder_decoder/checkpoints_models/{self.name}.pt",
                                           verbose=self.verbose)

            while (self.current_epoch < self.epochs) & (early_stopping.early_stop == False):
                #self.epoch(self.dltrain, fig=fig, ax=ax, **kwargs)
                '''if label and self.current_epoch == 0:
                    label = True
                else:
                    label = False'''
                self.epoch(self.dltrain, label=label, **kwargs)
                early_stopping(self.val_loss[-1], self.model)
            self.stop_epoch = early_stopping.stop_epoch

        else:
            for e in range(self.epochs):
                #self.epoch(self.dltrain, fig=fig, ax=ax, **kwargs)
                '''if label and e == 0:
                    label = True
                else:
                    label = False'''
                self.epoch(self.dltrain, label=label, **kwargs)

            self.stop_epoch = self.epochs

        t_end = time.time()

        if self.verbose_plot:
            print(f"Total training done in {t_end - t_begin} seconds and {self.stop_epoch} epochs.")
            fig=kwargs.pop('fig')
            ax=kwargs.pop('ax')
            if early_stop:
                self.info(early_stopping=early_stopping, fig=fig, ax=ax, label=True, **kwargs)
            else:
                self.info(fig=fig, ax=ax, label=True, **kwargs)

        plt.tight_layout()
        #plt.close()

    def info(self, **kwargs):
        if 'fig' in kwargs and 'ax' in kwargs:
            fig,ax = kwargs['fig'], kwargs['ax']
        else:
            fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(3,3))

        ax.cla()
        label = kwargs.get('label', True)
        ax.plot(np.arange(self.current_epoch), torch.tensor(self.training_loss).detach().numpy(), color='blue',
                 label='Trainset' if label else '_nolegend_')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.title.set_text(f'{self.name}\nnum_epochs = {self.current_epoch}.')

        if self.validation:
            ax.plot(np.arange(self.current_epoch), torch.tensor(self.val_loss).detach().numpy(), color='orange',
                     label='Validation set' if label else '_nolegend_')

        if 'early_stopping' in kwargs:
            early_stopping = kwargs['early_stopping']
            ax.axvline(early_stopping.stop_epoch-1, linestyle='--', color='r', label='Stopping Checkpoint')
            ax.title.set_text(f'{self.name}\nnum_epochs = {early_stopping.stop_epoch}.')
        ax.legend(loc='upper center', bbox_to_anchor = (0.5, -0.2), fancybox=True, shadow=True)

        if 'ax_ROC' in kwargs:
            ax = kwargs['ax_ROC']
            ax.cla()
            n_classes = self.model.n_classes
            pred, target = self.model.compute_pred_and_target(self.dltest)
            title = 'ROC'
            if n_classes==1:
                pred=[pred]
                target=[target]
            for i in range(n_classes):
                pred_i = pred[i]
                target_i = target[i]
                FPR, TPR = self.model.ROC(pred=pred_i, target=target_i, verbose=False)
                name_class = self.model.label_names[i].split('_')[1]
                ax.scatter(FPR, TPR, s=0.1, label=name_class)
                title += f'\nAUC {name_class} = {round(auc(FPR, TPR),3)}'
            ax.set_title(title)
            ax.plot(np.linspace(0,1,100),np.linspace(0,1,100),linestyle='--', alpha=0.5,
                        color='grey')
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR')
            ax.legend(loc='center left', bbox_to_anchor = (1, 0), fancybox=True, shadow=True)
        plt.tight_layout()
        plt.draw()
        if self.make_movie:
            path = '/home/ghisalberti/BL_encoder_decoder/model/movies/' + self.name + '_{:04d}.png'.format(self.current_epoch)
            plt.savefig(path)
        display.clear_output(wait=True)
        display.display(plt.gcf())
