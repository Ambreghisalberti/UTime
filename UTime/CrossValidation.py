import matplotlib.pyplot as plt
from torch.nn import MSELoss, CrossEntropyLoss, BCELoss
from .CostFunctions import WeightedMSE, WeightedBCE
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import auc
import numpy as np
import scipy
import copy
from .Training import Training


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

    return train_loss, test_loss


def make_dataloaders(windows, test_ratio=0.2):
    train, test = random_split(windows, [1 - test_ratio, test_ratio])
    dl_train = DataLoader(train, batch_size=10, shuffle=True)
    dl_test = DataLoader(test, shuffle=True)
    return dl_train, dl_test


def add_scores(model, dl, precisions, recalls, F1_scores, TPRs, FPRs, AUCs):
    p, r, F1 = model.scores(dl=dl)
    TPR, FPR = model.ROC(dl=dl, verbose=False)
    AUC = auc(FPR, TPR)

    precisions.append(p)
    recalls.append(r)
    F1_scores.append(F1)
    TPRs.append(TPR)
    FPRs.append(FPR)
    AUCs.append(AUC)

    return precisions, recalls, F1_scores, TPRs, FPRs, AUCs


def plot_mean_ROC(FPRs, TPRs, AUCs):
    FPRs, TPRs = np.array(FPRs), np.array(TPRs)
    plt.figure()
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), linestyle='--', color='grey', alpha=0.5)
    reference_FPR = np.linspace(0, 1, 1000)
    interpolated_TPRs = TPRs.copy()
    for i in range(len(TPRs)):
        interpolated_TPRs[i] = scipy.interpolate.interp1d(FPRs[i], TPRs[i])(reference_FPR)
        plt.plot(FPRs[i], TPRs[i], color='blue', linewidth=0.5)

    plt.fill_between(reference_FPR, np.mean(interpolated_TPRs, axis=0) - np.std(interpolated_TPRs, axis=0),
                     np.mean(interpolated_TPRs, axis=0) + np.std(interpolated_TPRs, axis=0), alpha=0.5)
    plt.plot(reference_FPR, np.mean(interpolated_TPRs, axis=0), linewidth=2, color='red')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC for the cross-validation : mean AUC = {round(np.mean(np.array(AUCs)), 2)}")
    plt.show()


def cross_validation(architecture, windows, nb_iter, loss_function, **kwargs):
    architecture = architecture.double()
    precisions, recalls, F1_scores, TPRs, FPRs, AUCs, models = [], [], [], [], [], [], []

    for iter in range(nb_iter):
        print(f'\nIteration {iter} :')
        model = copy.deepcopy(architecture)

        dl_train, dl_test = make_dataloaders(windows, test_ratio=kwargs.get('test_ratio', 0.2))

        train_loss, test_loss = get_loss_functions(loss_function, dl_train, dl_test)
        training = Training(model, 2000, dl_train, dltest=dl_test, dlval=dl_test, validation=True,
                            # To make it more general, get those parameters from kwargs?
                            train_criterion=train_loss, val_criterion=test_loss,
                            learning_rate=0.001, verbose_plot=True if iter == 0 else False, mirrored=True)

        '''training = Training(model, 2000, dl_train, dltest = dl_test, dlval=dl_test, validation=True,     # To make it more general, get those parameters from kwargs?
                                       train_criterion = train_loss,val_criterion = test_loss,
                                       learning_rate=0.001, verbose_plot = True, mirrored = True)'''
        name = loss_function + f', lr = {training.lr}, n={training.current_epoch}, early_stopping, n°{iter}'  # To make it more general, get early stopping from kwargs?
        training.fit(verbose=False, name=name, early_stop=True, patience=40)
        precisions, recalls, F1_scores, TPRs, FPRs, AUCs = add_scores(model, dl_test, precisions, recalls, F1_scores,
                                                                      TPRs, FPRs, AUCs)
        models.append(training.model.to('cpu'))

        del model, training

    if kwargs.get('verbose', True):
        plot_mean_ROC(FPRs, TPRs, AUCs)

    return precisions, recalls, F1_scores, TPRs, FPRs, AUCs, models
