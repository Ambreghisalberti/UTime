import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, auc
from torch.nn import MSELoss, BCELoss, CrossEntropyLoss
from .CostFunctions import WeightedMSE, WeightedBCE
import copy
from torch.utils.data import random_split, DataLoader
from .Training import Training
from .CrossValidation import get_loss_functions

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

        target = torch.Tensor([])
        pred = torch.Tensor([])

        for i, X, y in dl:
            if isinstance(X, tuple):
                a,b = X
                X = (a.to(self.device), b.to(self.device))
            elif isinstance(X, list):
                X = [X[0].to(self.device), X[1].to(self.device)]
            else:
                X = X.to(self.device)
            target = torch.concat((target, y))
            pred = torch.concat((pred, torch.Tensor.cpu(self.forward(X))))

        if mirrored:
            for i, X, y in dl:
                if isinstance(X, tuple):
                    a, b = X
                    X = (a.to(self.device), b.to(self.device))
                elif isinstance(X, list):
                    flipped_X = [X[0].to(self.device).flip(-1), X[1].to(self.device).flip(-1)]
                else:
                    flipped_X = X.to(self.device).flip(-1)
                target = torch.concat((target, y.flip(-1)))
                pred = torch.concat((pred, torch.Tensor.cpu(self.forward(flipped_X))))

        pred = pred.transpose(0,1)
        target=target.transpose(0,1)

        return pred, target

    def confusion_matrix(self, threshold=0.5, **kwargs):
        if 'dl' in kwargs:
            if 'dl' not in kwargs:
                raise Exception("Either pred and target must be given to ROC, or dl.")
            dl = kwargs.get('dl')
            pred, target = self.compute_pred_and_target(dl)
        else:
            if ('prediction' not in kwargs) | ('target' not in kwargs):
                raise Exception("Either a dataloader or the prediction and target must be given.")
            pred = kwargs.get('prediction')
            target = kwargs.get('target')

        pred = pred.detach().numpy().flatten()
        target = target.numpy().flatten()

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
            print("There is no positive target, the recall is not defined.")
            recall = -1
        else:
            recall = TP / (TP + FN)
        if (TP + FP) == 0:
            print("There is no predicted positive class, the precision is not defined.")
            precision = -1
        else:
            precision = TP / (TP + FP)
        if (TP + (FN + FP) / 2) == 0:
            F1 = -1
        else:
            F1 = TP / (TP + (FN + FP) / 2)

        if verbose:
            print(
                f"Precision = {round(precision * 100, 2)}%, recall = {round(recall * 100, 2)}%, "
                f"F1 score = {round(F1, 3)}")

        return precision, recall, F1

    def max_F1(self, **kwargs):
        precisions, recalls, F1s = [], [], []
        thresholds = np.linspace(0, 1, 1000)
        for threshold in thresholds:
            precision, recall, F1 = self.scores(verbose=False, threshold=threshold, **kwargs)
            precisions.append(precision)
            recalls.append(recall)
            F1s.append(F1)
        F1s = np.array(F1s)

        return precisions[np.argmax(F1s)], recalls[np.argmax(F1s)], np.max(F1s)

    def scatter_threshold_on_ROC(self, threshold, FPR, TPR, thresholds, **kwargs):
        threshold_plus = thresholds[thresholds >= threshold][0]  # First threshold above wanted value
        threshold_minus = thresholds[thresholds <= threshold][-1]  # Last threshold below wanted value
        if threshold_plus - threshold < threshold - threshold_minus:
            plt.scatter(np.array(FPR)[thresholds >= threshold][0], np.array(TPR)[thresholds >= 0.5][0], s=30, **kwargs)
        else:
            plt.scatter(np.array(FPR)[thresholds <= 0.5][-1], np.array(TPR)[thresholds <= 0.5][-1], s=30, **kwargs)

    def ROC(self, verbose=True, **kwargs):
        if 'pred' not in kwargs or 'target' not in kwargs:
            if 'dl' not in kwargs:
                raise Exception("Either pred and target must be given to ROC, or dl.")
            dl = kwargs['dl']
            pred, target = self.compute_pred_and_target(dl)
        else:
            pred, target = kwargs['pred'], kwargs['target']


        TPR = []
        FPR = []
        thresholds = np.linspace(0, 1, 1000)
        for threshold in thresholds:
            [[TP,FN],[FP,TN]] = self.confusion_matrix(prediction=pred, target=target, threshold=threshold, verbose=False)
            TPR.append(TP / (TP + FN) if TP+FN > 0 else np.nan)
            FPR.append(FP / (FP + TN) if FP+TN > 0 else np.nan)

        if verbose:
            plt.figure()
            plt.scatter(FPR, TPR, s=0.5)
            plt.plot(thresholds, thresholds, color='grey', alpha=0.5, linestyle='--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")

            # Scatter a dot corresponding to a threshold close to 0.5
            self.scatter_threshold_on_ROC(0.5, FPR, TPR, thresholds, color='red')
            plt.title(f"ROC, AUC = {round(auc(FPR, TPR),2)}")

            #best_threshold = self.find_best_threshold(dl, **kwargs)
            #self.scatter_threshold_on_ROC(best_threshold, FPR, TPR, thresholds, color='green')

            #plt.title(f"ROC, AUC = {round(auc(FPR, TPR),2)}\nbest_threshold = {round(best_threshold,2)}")

        return FPR, TPR

    def find_best_threshold(self, **kwargs):
        if 'pred' not in kwargs or 'target' not in kwargs:
            dl = kwargs.get('dl')
            pred, target = self.compute_pred_and_target(dl)
        else:
            pred, target = kwargs['pred'], kwargs['target']

        F1_scores = []
        thresholds = np.linspace(0, 1, 1000)
        for threshold in thresholds:
            [[TP, FN], [FP, TN]] = self.confusion_matrix(prediction=pred, target=target, threshold=threshold,
                                                         verbose=False)
            F1_scores.append(2*TP / (2*TP + FN + FP))

        return thresholds[np.argmax(F1_scores)]

