import matplotlib.pyplot as plt
from torch.nn import MSELoss, CrossEntropyLoss, BCELoss
from .CostFunctions import WeightedMSE, WeightedBCE, DiceLoss
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import auc
import numpy as np
import scipy
import copy
from .Training import Training
from IPython import display
from datetime import datetime

def cross_validation(architecture, windows, nb_iter, loss_function, **kwargs):
    if 'fig' in kwargs and 'ax' in kwargs:
        fig = kwargs['fig']
        axes = kwargs['ax']
    else:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

    architecture = architecture.double()
    precisions, recalls, F1_scores, FPRs, TPRs, AUCs, models, trainings = [], [], [], [], [], [], [], []
    name = kwargs.pop('name', str(datetime.now())[:10])

    for iter in range(nb_iter):
        print(f'\nIteration {iter} :')
        model = copy.deepcopy(architecture)

        if "pretrained" in kwargs:
            model = initialize_pretrained_model(model, kwargs["pretrained"])

        dl_train, dl_test = make_dataloaders(windows, test_ratio=kwargs.get('test_ratio', 0.2),
                                             batch_size=kwargs.get('batch_size',10))

        precisions, recalls, F1_scores, FPRs, TPRs, AUCs, models, trainings = train_one_iter(model, iter, loss_function, dl_train,
                                                                                  dl_test, models, trainings, precisions,
                                                                                  recalls,F1_scores, FPRs, TPRs, AUCs,
                                                                                  fig, axes, **kwargs)

    if kwargs.get('verbose', True):
        plot_mean_loss(models, fig=fig, ax=axes[0])
        plot_mean_ROC(FPRs, TPRs, AUCs, fig=fig, ax=axes[1])
    plt.tight_layout()
    plt.draw()

    if kwargs.get('savefig', False):
        plt.savefig('/home/ghisalberti/BL_encoder_decoder/model/diagnostics/'+name+'_cross_val.png')

    return precisions, recalls, F1_scores, FPRs, TPRs, AUCs, models


def train_one_iter(model, iter, loss_function, dl_train, dl_test, models, trainings, precisions, recalls, F1_scores,
                   FPRs, TPRs, AUCs, fig, axes, **kwargs):
    train_loss, test_loss = get_loss_functions(loss_function, dl_train, dl_test)

    name = kwargs.pop('name', str(datetime.now())[:10])
    training = Training(model, 2000, dl_train, dltest=dl_test, dlval=dl_test, validation=True,
                        # To make it more general, get those parameters from kwargs?
                        train_criterion=train_loss, val_criterion=test_loss,
                        learning_rate=kwargs.get('lr', 0.001), verbose_plot=True, mirrored=True,
                        name=name + f'_iter{iter}', **kwargs)


    if kwargs.get('plot_ROC', False):
        training.fit(verbose=False, early_stop=kwargs.get('early_stop', True), patience=kwargs.get('patience', 40),
                     fig=fig, ax=axes[0], ax_ROC=axes[1], label=True)

    else:
        training.fit(verbose=False, early_stop=kwargs.get('early_stop', True), patience=kwargs.get('patience', 40),
                     fig=fig, ax=axes[0], label=True)
    precisions, recalls, F1_scores, FPRs, TPRs, AUCs = add_scores(model, dl_test, precisions, recalls, F1_scores,
                                                                  FPRs, TPRs, AUCs)
    models.append(training.model.to('cpu'))
    trainings.append(training.to('cpu'))

    return precisions, recalls, F1_scores, FPRs, TPRs, AUCs, models, trainings



def get_loss_functions(loss_function, dl_train, dl_test):
    if loss_function == 'CEL':
        train_loss, test_loss = CrossEntropyLoss(reduction='mean'), CrossEntropyLoss(reduction='mean')
    elif loss_function == 'MSE':
        train_loss, test_loss = MSELoss(), MSELoss()
    elif loss_function == 'WeightedMSE':
        train_loss, test_loss = WeightedMSE(dl=dl_train), WeightedMSE(dl=dl_test)
    elif loss_function == 'BCE':
        train_loss, test_loss = BCELoss(), BCELoss()
    elif loss_function == 'WeightedBCE':
        train_loss, test_loss = WeightedBCE(dl=dl_train), WeightedBCE(dl=dl_test)
    elif loss_function == 'DiceLoss':
        train_loss, test_loss = DiceLoss(), DiceLoss()
    return train_loss, test_loss


def make_dataloaders(windows, test_ratio=0.2, batch_size=10):
    train, test = random_split(windows, [1 - test_ratio, test_ratio])
    dl_train = DataLoader(train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(test, shuffle=True)
    return dl_train, dl_test


def add_scores(model, dl, precisions, recalls, F1_scores, FPRs, TPRs, AUCs):
    p, r, F1 = model.scores(dl=dl)
    FPR, TPR = model.ROC(dl=dl, verbose=False)
    AUC = auc(FPR, TPR)

    precisions.append(p)
    recalls.append(r)
    F1_scores.append(F1)
    TPRs.append(TPR)
    FPRs.append(FPR)
    AUCs.append(AUC)

    return precisions, recalls, F1_scores, FPRs, TPRs, AUCs

def plot_mean(reference_x, x_list, y_list, **kwargs):
    x_list, y_list = np.array(x_list), np.array(y_list)

    if 'fig' in kwargs and 'ax' in kwargs:
        fig = kwargs['fig']
        ax = kwargs['ax']
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.cla()

    interpolated_y = y_list.copy()
    for i in range(len(y_list)):
        interpolated_y[i] = scipy.interpolate.interp1d(x_list[i], y_list[i])(reference_x)
        ax.plot(x_list[i], y_list[i], color=kwargs.get('color','blue'), linewidth=0.5)

    ax.fill_between(reference_x, np.mean(interpolated_y, axis=0) - np.std(interpolated_y, axis=0),
                    np.mean(interpolated_y, axis=0) + np.std(interpolated_y, axis=0), alpha=0.5)
    ax.plot(interpolated_y, np.mean(interpolated_y, axis=0), linewidth=2, color=kwargs.get('color_mean','red'),
            label = kwargs.get('label', '_nolegend_'))
    ax.legend()

    return fig, ax


def plot_mean_ROC(FPRs, TPRs, AUCs, **kwargs):
    fig, ax = plot_mean(np.linspace(0, 1, 100), FPRs, TPRs, **kwargs)
    ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), linestyle='--', color='grey', alpha=0.5)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.title.set_text(f"ROC for the cross-validation\n AUC = {round(np.mean(np.array(AUCs)), 2)} "
                      f"+/- {round(np.std(np.array(AUCs)),3)}")
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.tight_layout()

    #plt.close()

def plot_mean_loss(trainings, **kwargs):
    current_epochs, epochs, train_losses, val_losses = [], [], [], []
    for training in trainings:
        current_epochs.append(training.current_epoch)
        epochs += [list(np.arange(training.current_epoch))]
        train_losses += [list(training.val_loss)]
        val_losses += [list(training.val_loss)]
    reference_epochs = np.arange(np.min(current_epochs))

    fig, ax = plot_mean(reference_epochs, epochs, train_losses, color='green', color_mean = 'blue', **kwargs)
    fig, ax = plot_mean(reference_epochs, epochs, val_losses, color='orange', color_mean='red', fig=fig, ax=ax)

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.title.set_text()    # To fill
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.tight_layout()


def initialize_pretrained_model(new_model, pretrained_model):
    weights = pretrained_model.state_dict()
    weights = {key: weights[key] for key in weights.keys() if key.startswith('encoder') or key.startswith('decoder')}
    model_dict = new_model.state_dict()
    model_dict.update(weights)
    new_model.load_state_dict(model_dict)
    return new_model


def write_scores(text, scores, variable, value, precisions, recalls, F1_scores, AUCs):
    mean_precision, std_precision = np.mean(np.array(precisions)), np.std(np.array(precisions))
    mean_recalls, std_recalls = np.mean(np.array(recalls)), np.std(np.array(recalls))
    mean_F1, std_F1 = np.mean(np.array(F1_scores)), np.std(np.array(F1_scores))
    mean_AUC, std_AUC = np.mean(np.array(AUCs)), np.std(np.array(AUCs))

    scores.loc[f'{variable}={value}', :] = [mean_precision, std_precision, mean_recalls, std_recalls, mean_F1, std_F1,
                                            mean_AUC, std_AUC]

    text += f'For {variable} = {value}:\n\n'
    text += f'Precision : mean = {round(mean_precision * 100, 2)}%, std = {round(100 * std_precision, 2)}%.\n'
    text += f'Recall : mean = {round(100 * mean_recalls, 2)}%, std = {round(100 * std_recalls, 2)}%.\n'
    text += f'F1 : mean = {round(mean_F1, 2)}, std = {round(std_F1, 2)}.\n'
    text += f'AUC : mean = {round(mean_AUC, 2)}, std = {round(std_AUC, 2)}.\n'
    text += '\n'

    return text, scores
