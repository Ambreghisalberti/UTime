import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, auc

class Model():

    # def __init__(self,model,**kwargs):
    # self.model = model

    def evaluate(self, dl, criterion, mirrored=True):  # on dlval or dltest
        # self.eval()
        loss = 0
        count = 0
        with torch.no_grad():
            for i, inputs, labels in dl:
                count += 1
                loss += criterion(torch.flatten(self.forward(inputs.double())), torch.flatten(labels.double())).detach()

        if mirrored:
            for i, inputs, labels in dl:
                count += 1
                loss += criterion(torch.flatten(self.forward(inputs.flip(-1).double())),
                                  torch.flatten(labels.flip(-1).double())).detach()

        return loss / count

    def compute_pred_and_target(self, dl, mirrored=True):
        target = np.array([])
        pred = np.array([])
        for i, X, y in dl:
            target = np.concatenate((target, torch.flatten(y).numpy()))
            pred = np.concatenate((pred, torch.flatten(self.forward(X)).detach().numpy()))

        if mirrored:
            for i, X, y in dl:
                target = np.concatenate((target, torch.flatten(y.flip(-1)).numpy()))
                pred = np.concatenate((pred, torch.flatten(self.forward(X.flip(-1))).detach().numpy()))

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
            plt.scatter(np.array(FPR)[thresholds >= threshold][0], np.array(TPR)[thresholds >= 0.5][0], s=10, **kwargs)
        else:
            plt.scatter(np.array(FPR)[thresholds <= 0.5][-1], np.array(TPR)[thresholds <= 0.5][-1], s=10, **kwargs)

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
            self.scatter_threshold_on_ROC(self, 0.5, FPR, TPR, thresholds, color='red')
            best_threshold = self.find_best_threshold(dl, **kwargs)
            self.scatter_threshold_on_ROC(self, best_threshold, FPR, TPR, thresholds, color='green')

            plt.title(f"ROC, AUC = {round(auc(FPR, TPR),2)}, best_threshold = {best_threshold}")

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
