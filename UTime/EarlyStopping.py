import numpy as np
import torch
import matplotlib.pyplot as plt


class EarlyStopping():
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, model, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0

        """
        self.model = model
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.min_loss = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.current_epoch = 0
        self.stop_epoch = None
        self.val_loss = []

    def __call__(self, val_loss, model):
        val_loss = val_loss.item()
        self.current_epoch += 1
        self.val_loss.append(val_loss)

        if self.min_loss is None:
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.min_loss - self.delta:
            # Worse or only a little better. If delta = 0, only worse cases are considered
            self.counter += 1
            if self.verbose:
                print("EarlyStopping counter: " + str(self.counter) + " out of " + str(self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                self.stopped()

        else:
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose & (self.min_loss is not None):
            print("Validation loss decreased (" + str(round(self.min_loss, 6)) + " --> " + str(round(val_loss, 6)) +
                  ").  Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.min_loss = val_loss

    def stopped(self):
        # load the last checkpoint with the best model
        self.model.load_state_dict(torch.load(self.path))
        if self.early_stop:
            stop_epoch = self.current_epoch - self.counter
        else:
            stop_epoch = self.current_epoch
        self.stop_epoch = stop_epoch

        if self.verbose:
            print("Early stopping at epoch " + str(self.stop_epoch))

'''
    def info(self, name, dt):
        print("Total training done in " + str(dt) + ' seconds and ' + str(self.stop_epoch) + ' epochs.')

        plt.figure()
        plt.plot(np.arange(1, self.current_epoch + 1), self.model.train_loss, label='Trainset')
        plt.plot(np.arange(1, self.current_epoch + 1), self.val_loss, label='Validation set')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.title('num_epochs = ' + str(self.stop_epoch))
        # find position of lowest validation loss
        # minposs = np.argmin(loss_over_iterations_val)
        plt.axvline(self.stop_epoch, linestyle='--', color='r', label='Stopping Checkpoint')
        plt.legend()
        plt.savefig("models_regions/figures/CNN_" + name + ".png")

'''
