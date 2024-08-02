import matplotlib.pyplot as plt
from torch.nn import MSELoss, CrossEntropyLoss, BCELoss
from .CostFunctions import (WeightedMSE, WeightedBCE, DiceLoss, IntersectionOverUnion, FocalLoss,
                            maxF1, roc_star)
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import auc
import numpy as np
import scipy
import copy
from .Training import Training
from IPython import display
from datetime import datetime
import torch
import random as rd
from datetime import timedelta
import pandas as pd


def initialize_empty_scores(windows):
    precisions, recalls, F1_scores, FPRs, TPRs, AUCs, max_F1s = (
        {f"{windows.label[i].split('_')[1]}": [] for i in range(len(windows.label))} for i in range(7))
    return {'models':[], 'precisions':precisions, 'recalls':recalls, 'F1_scores':F1_scores, 'FPRs':FPRs, 'TPRs':TPRs,
            'AUCs':AUCs, 'max_F1s':max_F1s, 'train_losses':[], 'val_losses':[], 'last_epochs':[], 'dl_tests':[]}


def cross_validation(architecture, windows, nb_iter, loss_function, **kwargs):
    if 'fig' in kwargs and 'ax' in kwargs:
        fig = kwargs.pop('fig')
        axes = kwargs.pop('ax')
    else:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

    architecture = architecture.double()
    dict = initialize_empty_scores(windows)
    name = kwargs.pop('name', str(datetime.now())[:10])

    for iter in range(nb_iter):
        print(f'\nIteration {iter} :')
        model = copy.deepcopy(architecture)

        if "pretrained" in kwargs:
            model = initialize_pretrained_model(model, kwargs["pretrained"])

        dl_train, dl_test = make_dataloaders(windows, **kwargs)

        dict = train_one_iter(model, iter, loss_function, dl_train, dl_test, dict, fig, axes, name=name, **kwargs)

    if kwargs.get('verbose', True):
        plot_mean_loss(dict['train_losses'], dict['val_losses'], dict['last_epochs'], fig=fig, ax=axes[0], **kwargs)
        plot_mean_ROC(dict['FPRs'], dict['TPRs'], dict['AUCs'], fig=fig, ax=axes[1])
    plt.tight_layout()
    plt.draw()

    if kwargs.get('savefig', False):
        plt.savefig('/home/ghisalberti/BL_encoder_decoder/model/diagnostics/'+name+'_cross_val.png')

    return dict


def train_one_iter(model0, iter, loss_function, dl_train, dl_test, dict, fig, axes, **kwargs):

    model = copy.deepcopy(model0)
    train_loss, test_loss = get_loss_functions(loss_function, dl_train, dl_test)

    name = kwargs.pop('name', str(datetime.now())[:10])
    training = Training(model, 2000, dl_train, dltest=dl_test, dlval=dl_test, validation=True,
                        # To make it more general, get those parameters from kwargs?
                        train_criterion=train_loss, val_criterion=test_loss,
                        learning_rate=kwargs.get('lr', 0.001), verbose_plot=True, mirrored=True,
                        name=name + f'_iter{iter}', **kwargs)


    if kwargs.get('plot_ROC', False):
        training.fit(verbose=kwargs.pop('verbose',False), early_stop=kwargs.pop('early_stop', True), patience=kwargs.pop('patience', 40),
                     fig=fig, ax=axes[0], ax_ROC=axes[1], label=True, name=name + f'_iter{iter}', **kwargs)

    else:
        training.fit(verbose=kwargs.pop('verbose',False), early_stop=kwargs.pop('early_stop', True), patience=kwargs.pop('patience', 40),
                     fig=fig, ax=axes[0], label=True, name=name + f'_iter{iter}', **kwargs)

    dict = add_scores(training.model.to(training.model.device), dl_test, dict)
    dict['train_losses'] += [list(torch.Tensor(training.training_loss).numpy())]
    dict['val_losses'] += [list(torch.Tensor(training.val_loss).numpy())]
    dict['last_epochs'] += [training.current_epoch]

    return dict


def get_loss_functions(loss_function, dl_train, dl_test):
    if loss_function == 'CEL':
        train_loss, test_loss = CrossEntropyLoss(reduction='mean'), CrossEntropyLoss(reduction='mean')
    elif loss_function == 'MSE':
        train_loss, test_loss = MSELoss(), MSELoss()
    elif loss_function == 'WeightedMSE':
        train_loss, test_loss = WeightedMSE(dl_train), WeightedMSE(dl_test)
    elif loss_function == 'BCE':
        train_loss, test_loss = BCELoss(), BCELoss()
    elif loss_function == 'WeightedBCE':
        train_loss, test_loss = WeightedBCE(dl_train), WeightedBCE(dl_test)
    elif loss_function == 'DiceLoss':
        train_loss, test_loss = DiceLoss(), DiceLoss()
    elif loss_function == 'IoU':
        train_loss, test_loss = IntersectionOverUnion(), IntersectionOverUnion()
    elif loss_function == 'FocalLoss':
        train_loss, test_loss = FocalLoss(alpha=0.75), FocalLoss(alpha=0.75)
    elif loss_function == 'maxF1':
        train_loss, test_loss = maxF1(), maxF1()
    elif loss_function == 'roc_star':
        train_loss, test_loss = roc_star(), roc_star()
    else:
        raise Exception("Unknown loss : loss has not been given, or is not in the list allowed by function get_loss_function!")
    return train_loss, test_loss


def make_dataloaders(windows, **kwargs):
    stride = windows.stride
    if stride >= windows.win_length:
        return make_dataloaders_without_stride(windows, **kwargs)
    else:
        return make_dataloaders_with_stride(windows, **kwargs)


def make_dataloaders_without_stride(windows, batch_size=10, **kwargs):
    test_ratio = kwargs.get("test_ratio", 0.2)
    train_ratio = kwargs.get('train_ratio', 1-test_ratio)
    if test_ratio + train_ratio < 1:
        train, test, rest = random_split(windows, [train_ratio, test_ratio, 1 - train_ratio - test_ratio])
    else:
        train, test = random_split(windows, [train_ratio, test_ratio])

    dl_train = DataLoader(train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(test, shuffle=True)
    return dl_train, dl_test


def make_windows_groups(windows, interval=timedelta(days=1), **kwargs):
    all_start, all_stop = windows.dataset.index.values[0], windows.dataset.index.values[-1]
    starts_intervals = pd.date_range(start=all_start, end=all_stop, freq=interval)
    stops_intervals = starts_intervals + interval

    windows.dataset['nbr_window'] = 0
    windows.dataset.iloc[windows.windows_indices, -1] = 1
    windows.dataset['nbr_window'] = windows.dataset['nbr_window'].values.cumsum() * windows.dataset['nbr_window']

    groups = []
    for i, (start, stop) in enumerate(zip(starts_intervals, stops_intervals)):
        subdf = windows.dataset.loc[start:stop]
        windows_stops = subdf.index.values[subdf['nbr_window'].values != 0]
        windows_starts = windows_stops - windows.win_duration
        group = list(subdf.loc[windows_stops, 'nbr_window'].values[windows_starts >= start] - 1)
        if len(group) > 1:
            groups.append(group)

    test = np.array([values for x in groups for values in x])
    assert np.max(test) < len(windows), "The maximum numero is too high compared to the number of windows!"

    return groups


def make_dataloaders_with_stride(windows, **kwargs):
    groups = make_windows_groups(windows, **kwargs)
    rd.shuffle(groups)

    test_ratio = kwargs.get('test_ratio', 0.2)
    train_ratio = kwargs.get('train_ratio',1-test_ratio)

    train_groups = groups[:int(len(groups) * train_ratio)]
    train_indices = [t for tx in train_groups for t in tx]
    test_groups = groups[-int(len(groups) * test_ratio):]
    test_indices = [t for tx in test_groups for t in tx]

    dl_train = DataLoader([windows[i] for i in train_indices], batch_size=kwargs.get('batch_size', 10), shuffle=True)
    dl_test = DataLoader([windows[i] for i in test_indices], shuffle=True)

    return dl_train, dl_test


def add_scores(model, dl, dict):
    dict['models'] += [model.to('cpu')]
    model = model.to(model.device)

    n_classes = model.n_classes
    pred, target = model.compute_pred_and_target(dl)

    i, X, y = next(iter(dl))
    if isinstance(X, list):
        dl = DataLoader([(i, [x.to('cpu') for x in X], y.to('cpu')) for i, X, y in dl])
    else:
        dl = DataLoader([(i, X.to('cpu'), y.to('cpu')) for i, X, y in dl])
    dict['dl_tests'] += [dl]  # Would need to save this datalaoder on cpu device instead of gpu?

    for i in range(n_classes):
        pred_i = pred[i]
        target_i = target[i]

        p, r, F1 = model.scores(prediction=pred_i, target=target_i)
        FPR, TPR = model.ROC(pred=pred_i, target=target_i, verbose=False)
        AUC = auc(FPR, TPR)
        prec_maxF1, recall_maxF1, max_F1 = model.max_F1(prediction=pred_i, target=target_i)
        name_class = model.label_names[i].split('_')[1]
        dict['precisions'][name_class] += [p]
        dict['recalls'][name_class] += [r]
        dict['F1_scores'][name_class] += [F1]
        dict['TPRs'][name_class] += [TPR]
        dict['FPRs'][name_class] += [FPR]
        dict['AUCs'][name_class] += [AUC]
        dict['max_F1s'][name_class] += [{'max_F1':max_F1, 'prec_maxF1':prec_maxF1, 'recall_maxF1':recall_maxF1}]

    return dict



def plot_mean(reference_x, x_list, y_list, **kwargs):
    x_list, y_list = np.array(x_list), np.array(y_list)

    if 'fig' in kwargs and 'ax' in kwargs:
        fig = kwargs['fig']
        ax = kwargs['ax']
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1)

    interpolated_y = [[] for i in range(y_list.shape[0])]
    for i in range(len(y_list)):
        interpolated_y[i] = scipy.interpolate.interp1d(x_list[i], y_list[i])(reference_x)
        ax.plot(x_list[i], y_list[i], color=kwargs.get('color','blue'), linewidth=0.5)
    interpolated_y = np.array(interpolated_y)

    ax.fill_between(reference_x, np.mean(interpolated_y, axis=0) - np.std(interpolated_y, axis=0),
                    np.mean(interpolated_y, axis=0) + np.std(interpolated_y, axis=0), alpha=kwargs.get('alpha',0.5),
                    color=kwargs.get('color','blue'))
    ax.plot(reference_x, np.mean(interpolated_y, axis=0), linewidth=2, color=kwargs.get('color_mean','red'),
            label = kwargs.get('label', '_nolegend_'))
    ax.legend()

    return fig, ax


def plot_mean_ROC(FPRs, TPRs, AUCs, fig, ax, **kwargs):
    ax.cla()
    title = f"ROC for the cross-validation"
    for key in FPRs.keys():
        fig, ax = plot_mean(np.linspace(0, 1, 1000), FPRs[key], TPRs[key], fig=fig, ax=ax, label = key)
        title += f"\n{key} AUC = {round(np.mean(np.array(AUCs[key])), 2)}"
    ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), linestyle='--', color='grey', alpha=0.5)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.title.set_text(title)
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.tight_layout()

    #plt.close()


def plot_mean_loss(train_losses, val_losses, last_epochs, fig, ax, **kwargs):
    max_epoch = np.min(last_epochs)
    epochs = [list(np.arange(max_epoch)) for e in last_epochs]
    reference_epochs = np.arange(max_epoch)
    train_losses = [train_loss[:max_epoch] for train_loss in train_losses]
    val_losses = [val_loss[:max_epoch] for val_loss in val_losses]

    ax.cla()
    fig, ax = plot_mean(reference_epochs, epochs, train_losses, color='green', color_mean = 'blue', label='Train loss',
                        fig=fig, ax=ax)
    fig, ax = plot_mean(reference_epochs, epochs, val_losses, color='orange', color_mean='red', label='Test loss',
                        fig=fig, ax=ax)

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.title.set_text("Mean loss during training")
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
    key = 'BL'
    mean_precision, std_precision = np.mean(np.array(precisions[key])), np.std(np.array(precisions[key]))
    mean_recalls, std_recalls = np.mean(np.array(recalls[key])), np.std(np.array(recalls[key]))
    mean_F1, std_F1 = np.mean(np.array(F1_scores[key])), np.std(np.array(F1_scores[key]))
    mean_AUC, std_AUC = np.mean(np.array(AUCs[key])), np.std(np.array(AUCs[key]))

    scores.loc[f'{variable}={value}', :] = [mean_precision, std_precision, mean_recalls, std_recalls, mean_F1, std_F1,
                                            mean_AUC, std_AUC]

    text += f'For {variable} = {value}:\n\n'
    text += f'Precision : mean = {round(mean_precision * 100, 2)}%, std = {round(100 * std_precision, 2)}%.\n'
    text += f'Recall : mean = {round(100 * mean_recalls, 2)}%, std = {round(100 * std_recalls, 2)}%.\n'
    text += f'F1 : mean = {round(mean_F1, 2)}, std = {round(std_F1, 2)}.\n'
    text += f'AUC : mean = {round(mean_AUC, 2)}, std = {round(std_AUC, 2)}.\n'
    text += '\n'

    return text, scores
