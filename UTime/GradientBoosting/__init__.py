import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier as Gbc
from sklearn.ensemble import HistGradientBoostingClassifier as Hgbc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import timedelta
import random as rd
from sklearn.metrics import auc


def split(all_data, columns, **kwargs):
    method_split = kwargs.get('method_split', 'random')
    if method_split == 'random':
        Xtrain, Xtest, ytrain, ytest = train_test_split(all_data.loc[:, columns].values, all_data.label_BL.values,
                                                        test_size=kwargs.get('test_size', 0.2))
    elif method_split == 'temporal':
        Xtrain, Xtest, ytrain, ytest = temporal_split(all_data.loc[:, columns + ['label_BL']], columns, ['label_BL'],
                                                      test_size=kwargs.get('test_size', 0.2))
    else:
        raise Exception(f"Split method should be 'random' or 'temporal', but is {method_split}.")
    return Xtrain, Xtest, ytrain.astype('int'), ytest.astype('int')


def temporal_split(data, columns, label_columns=None, test_size=0.2):
    if label_columns is None:
        label_columns = ['label_BL']
    dftrain, dftest = pd.DataFrame([], columns=data.columns), pd.DataFrame([], columns=data.columns)
    months = pd.date_range(start=data.index.values[0], end=data.index.values[-1], freq=timedelta(days=30))
    for i in range(len(months) - 1):
        temp = data[months[i]:months[i + 1]]
        len_test = int(len(temp) * test_size)
        indice = rd.randint(0, len(temp) - len_test)
        temp_test = temp.iloc[indice:indice + len_test]
        dftest = pd.concat((dftest, pd.DataFrame(temp_test.values, index=temp_test.index.values, columns=data.columns)))
        temp_train = pd.concat((temp.iloc[:indice], temp.iloc[indice + len_test:]))
        dftrain = pd.concat(
            (dftrain, pd.DataFrame(temp_train.values, index=temp_train.index.values, columns=data.columns)))

    assert len(
        dftrain[dftrain.index.isin(dftest.index)]) == 0, "Trainset and testset should not have any point in common!"
    return (dftrain.loc[:, columns].values, dftest.loc[:, columns].values,
            dftrain.loc[:, label_columns].values, dftest.loc[:, label_columns].values)


def compute_scores(pred, truth):
    TP = (pred * truth).sum()
    FP = (pred * (1 - truth)).sum()
    TN = ((1 - pred) * (1-truth)).sum()
    FN = ((1 - pred) * truth).sum()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return TP, FP, TN, FN, precision, recall


def train_model(df, columns, **kwargs):
    scaler = StandardScaler()
    data = df.copy().dropna()
    data.loc[:, columns] = scaler.fit_transform(data.loc[:, columns].values)

    method_split = kwargs.pop('method_split', 'temporal')
    verbose = kwargs.pop('verbose', True)
    model_verbose = kwargs.pop('model_verbose', True)
    test_size = kwargs.pop('test_size', 0.2)

    precisions, recalls, AUCs = [], [], []
    model = kwargs.pop('model', 'GBC')

    n_iter = kwargs.pop('n_iter', 1)
    for i in range(n_iter):
        assert n_iter > 0, f"n_iter must be strictly positive but is {n_iter}"
        xtrain, xtest, ytrain, ytest = split(data, columns, method_split=method_split, test_size=test_size)

        if model == 'GBC':
            gb = Gbc(verbose=model_verbose, **kwargs)
        elif model == 'HGBC':
            gb = Hgbc(verbose=model_verbose, **kwargs)
        else:
            raise Exception("Model should be GBC or HGBC")
        gb.fit(xtrain, ytrain)

        pred = gb.predict(xtest)
        ytest = ytest.flatten()
        TP, FP, TN, FN, precision, recall = compute_scores(pred, ytest)

        if verbose:
            print(f'Precision = {round(precision * 100, 2)}%, Recall = {round(recall * 100, 2)}%.')

        precisions.append(precision)
        recalls.append(recall)

        FPRs, TPRs, auc_value = ROC(gb, xtest, ytest)
        AUCs.append(auc_value)

    if verbose:
        fig, ax = plt.subplots(ncols=2)
        _ = ax[0].hist(precisions, bins=20)
        ax[0].set_xlabel('Precision')
        _ = ax[1].hist(recalls, bins=20)
        ax[1].set_xlabel('Recall')
        plt.show()

    return precisions, recalls, AUCs, xtrain, xtest, ytrain, ytest, gb


def ROC(model, xtest, ytest, **kwargs):
    proba = model.predict_proba(xtest)[:, 1]
    ytest = ytest.flatten()

    FPRs, TPRs = [], []
    for threshold in np.linspace(0, 1, 100):
        pred = proba > threshold
        TP, FP, TN, FN, _, _ = compute_scores(pred, ytest)
        FPR = FP / (FP + TN)
        TPR = TP / (TP + FN)
        FPRs.append(FPR)
        TPRs.append(TPR)

    auc_value = auc(FPRs, TPRs)
    if kwargs.get('verbose', False):
        plt.figure()
        plt.scatter(FPRs, TPRs)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), linestyle='--', color='grey', alpha=0.5)
        plt.title(f'ROC, AUC = {round(auc_value, 3)}')
        plt.show()
    return FPRs, TPRs, auc_value


def recall_precision_curve(model, xtest, ytest):
    proba = model.predict_proba(xtest)[:, 1]
    ytest = ytest.flatten()

    precisions, recalls = [], []
    for threshold in np.linspace(0, 1, 100):
        pred = proba > threshold
        _, _, _, _, precision, recall = compute_scores(pred, ytest)
        precisions.append(precision)
        recalls.append(recall)

    plt.figure()
    plt.scatter(recalls, precisions)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axhline(ytest.sum() / len(ytest), linestyle='--', color='grey', alpha=0.5)
    plt.show()
    return recalls, precisions


def learning_curve(model, xtrain, ytrain, xtest, ytest):
    ytrain = ytrain.flatten()
    train_loss = [np.mean((np.array(proba[:, 1]) - np.array(ytrain)) ** 2) for proba in
                  model.staged_predict_proba(xtrain)]

    ytest = ytest.flatten()
    test_loss = [np.mean((np.array(proba[:, 1]) - np.array(ytest)) ** 2) for proba in
                 model.staged_predict_proba(xtest)]

    plt.figure()
    plt.plot(np.arange(len(train_loss)), train_loss, label='Train loss')
    plt.plot(np.arange(len(test_loss)), test_loss, label='Test loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Loss through training')
    plt.show()

    return train_loss, test_loss
