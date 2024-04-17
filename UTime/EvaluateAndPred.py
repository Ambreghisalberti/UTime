import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, auc
from torch.nn import MSELoss, BCELoss, CrossEntropyLoss
from .CostFunctions import WeightedMSE, WeightedBCE
import copy
from torch.utils.data import random_split, DataLoader
from .Training import Training


class Model():

    # def __init__(self,model,**kwargs):
    # self.model = model

    def evaluate(self, dl, criterion, mirrored=True):  # on dlval or dltest
        # self.eval()
        loss = 0
        count = 0
        with torch.no_grad():
            for i, inputs, labels in dl:
                if isinstance(inputs, list):
                    inputs = [inputs[0].to(self.device).double(), inputs[1].to(self.device).double()]
                else:
                    inputs = inputs.to(self.device).double()
                labels = labels.to(self.device)
                count += 1
                loss += criterion(torch.flatten(self.forward(inputs)), torch.flatten(labels.double())).detach()

        if mirrored:
            for i, inputs, labels in dl:
                if isinstance(inputs, list):
                    flipped_inputs = [inputs[0].to(self.device).double().flip(-1), inputs[1].to(self.device).double().flip(-1)]
                else:
                    flipped_inputs = inputs.to(self.device).double().flip(-1)
                labels = labels.to(self.device)
                count += 1
                loss += criterion(torch.flatten(self.forward(flipped_inputs)),
                                  torch.flatten(labels.flip(-1).double())).detach()

        return loss / count

    def compute_pred_and_target(self, dl, mirrored=False):
        target = np.array([])
        pred = np.array([])
        for i, X, y in dl:
            if isinstance(X, tuple):
                a,b = X
                X = (a.to(self.device), b.to(self.device))
            elif isinstance(X, list):
                X = [X[0].to(self.device), X[1].to(self.device)]
            else:
                X = X.to(self.device)
            target = np.concatenate((target, torch.flatten(y).numpy()))
            pred = np.concatenate((pred, torch.Tensor.cpu(torch.flatten(self.forward(X))).detach().numpy()))

        if mirrored:
            for i, X, y in dl:
                if isinstance(X, tuple):
                    a, b = X
                    X = (a.to(self.device), b.to(self.device))
                elif isinstance(X, list):
                    flipped_X = [X[0].to(self.device).flip(-1), X[1].to(self.device).flip(-1)]
                else:
                    flipped_X = X.to(self.device).flip(-1)
                target = np.concatenate((target, torch.flatten(y.flip(-1)).numpy()))
                pred = np.concatenate((pred, torch.Tensor.cpu(torch.flatten(self.forward(flipped_X))).detach().numpy()))

        return pred, target

    def confusion_matrix(self, threshold=0.5, **kwargs):
        if 'dl' in kwargs:
            dl = kwargs.get('dl')
            pred, target = self.compute_pred_and_target(dl)
        else:
            if ('prediction' not in kwargs) | ('target' not in kwargs):
                raise Exception("Either a dataloader or the prediction and target must be given.")
            pred = kwargs.get('prediction')
            target = kwargs.get('target')

        pred_class = pred > threshold

        TP, TN, FP, FN = 0, 0, 0, 0
        TP += (target * pred_class).sum()  # target = 1 and pred = 1
        TN += ((1 - target) * (1 - pred_class)).sum()  # target = 0 and pred_class = 0
        FP += ((1 - target) * pred_class).sum()  # target = 0 and pred_class = 1
        FN += (target * (1 - pred_class)).sum()  # target = 1 and pred_class = 0
        TP, FP, TN, FN = TP.item(), FP.item(), TN.item(), FN.item()
        cm = [[TP,FN],[FP,TN]]

        verbose = kwargs.get('verbose',True)
        if verbose:
            disp = ConfusionMatrixDisplay(np.array(cm), display_labels=['BL', 'not_BL'])
            disp.plot()

        return cm

    def scores(self, verbose=True, **kwargs):
        if ('TP' not in kwargs) | ('FP' not in kwargs) | ('TN' not in kwargs) | ('FN' not in kwargs):
            [[TP,FN],[FP,TN]] = self.confusion_matrix(verbose=False, **kwargs)
        else:
            TP = kwargs.get('TP')
            FP = kwargs.get('FP')
            TN = kwargs.get('TN')
            FN = kwargs.get('FN')

        if (TP + FN) == 0:
            raise Exception("There is no positive target, the recall is not defined.")
        if (TP + FP) == 0:
            raise Exception("There is no predicted positive class, the precision is not defined.")

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = TP / (TP + (FN + FP) / 2)

        if verbose:
            print(
                f"Precision = {round(precision * 100, 2)}%, recall = {round(recall * 100, 2)}%, "
                f"F1 score = {round(F1, 3)}")

        return precision, recall, F1


    def scatter_threshold_on_ROC(self, threshold, FPR, TPR, thresholds, **kwargs):
        threshold_plus = thresholds[thresholds >= threshold][0]  # First threshold above wanted value
        threshold_minus = thresholds[thresholds <= threshold][-1]  # Last threshold below wanted value
        if threshold_plus - threshold < threshold - threshold_minus:
            plt.scatter(np.array(FPR)[thresholds >= threshold][0], np.array(TPR)[thresholds >= 0.5][0], s=30, **kwargs)
        else:
            plt.scatter(np.array(FPR)[thresholds <= 0.5][-1], np.array(TPR)[thresholds <= 0.5][-1], s=30, **kwargs)

    def ROC(self, dl, verbose=True, **kwargs):
        pred, target = self.compute_pred_and_target(dl)

        TPR = []
        FPR = []
        thresholds = np.linspace(0, 1, 1000)
        for threshold in thresholds:
            [[TP,FN],[FP,TN]] = self.confusion_matrix(prediction=pred, target=target, threshold=threshold, verbose=False)
            TPR.append(TP / (TP + FN))
            FPR.append(FP / (FP + TN))

        if verbose:
            plt.figure()
            plt.scatter(FPR, TPR, s=0.5)
            plt.plot(thresholds, thresholds, color='grey', alpha=0.5, linestyle='--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")

            # Scatter a dot corresponding to a threshold close to 0.5
            self.scatter_threshold_on_ROC(0.5, FPR, TPR, thresholds, color='red')
            best_threshold = self.find_best_threshold(dl, **kwargs)
            self.scatter_threshold_on_ROC(best_threshold, FPR, TPR, thresholds, color='green')

            plt.title(f"ROC, AUC = {round(auc(FPR, TPR),2)}\nbest_threshold = {round(best_threshold,2)}")

        return FPR, TPR

    def find_best_threshold(self, dl, **kwargs):
        pred, target = self.compute_pred_and_target(dl)

        F1_scores = []
        thresholds = np.linspace(0, 1, 1000)
        for threshold in thresholds:
            [[TP, FN], [FP, TN]] = self.confusion_matrix(prediction=pred, target=target, threshold=threshold,
                                                         verbose=False)
            F1_scores.append(2*TP / (2*TP + FN + FP))

        return thresholds[np.argmax(F1_scores)]

    def get_loss_functions(self, loss_function, dl_train, dl_test):

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

    def cross_validation(self, windows, nb_iter, loss_function):
        # Créer une copie du modèle
        architecture = self.double()
        precisions, recalls, F1_scores = [], [], []
        best_precisions, best_recalls, best_F1_scores = [], [], []
        models = []

        for iter in range(nb_iter):
            print(f'Iteration {iter} :')
            model = copy.deepcopy(architecture)

            # Data
            train, test = random_split(windows, [0.8, 0.2])
            dl_train = DataLoader(train, batch_size=10, shuffle=True)
            dl_test = DataLoader(test, shuffle=True)

            # Training
            train_loss, test_loss = self.get_loss_functions(loss_function, dl_train, dl_test)
            training = Training(model, 2000, dl_train, dltest=dl_test, dlval=dl_test, validation=True,
                                # To make it more general, get those parameters from kwargs?
                                train_criterion=train_loss, val_criterion=test_loss,
                                learning_rate=0.001, verbose_plot=True if iter == 0 else False, mirrored=True)
            name = loss_function + f', lr = {training.lr}, n={training.epoch}, early_stopping, n°{iter}'  # To make it more general, get early stopping from kwargs?
            training.fit(verbose=False, name=name, early_stop=True, patience=200)

            p, r, F1 = training.model.scores(dl=dl_test)

            precisions.append(p)
            recalls.append(r)
            F1_scores.append(F1)
            models.append(model)

        print('\n')
        print(
            f'Precision : mean = {round(np.mean(np.array(precisions)) * 100, 2)}%, std = {round(100 * np.std(np.array(precisions)), 2)}%.')
        print(
            f'Recall : mean = {round(100 * np.mean(np.array(recalls)), 2)}%, std = {round(100 * np.std(np.array(recalls)), 2)}%.')
        print(f'F1 : mean = {round(np.mean(np.array(F1_scores)), 2)}, std = {round(np.std(np.array(F1_scores)), 2)}.')


        return precisions, recalls, F1_scores, models



