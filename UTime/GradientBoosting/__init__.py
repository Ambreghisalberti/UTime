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
import warnings
from datetime import datetime


def split(all_data, columns, **kwargs):
    method_split = kwargs.get('method_split', 'random')
    if method_split == 'random':
        all_data['time'] = all_data.index.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(all_data.loc[:, list(columns)+['time']].values,
                                                        all_data.label_BL.values,
                                                        test_size=kwargs.get('test_size', 0.2))
        timestrain = Xtrain[:,-1]
        timestest = Xtest[:,-1]
        Xtrain = Xtrain[:,:-1]
        Xtest = Xtest[:,:-1]

    elif method_split == 'temporal':
        Xtrain, Xtest, ytrain, ytest, timestrain, timestest = (
            temporal_split(all_data.loc[:, list(columns) + ['label_BL']], columns,
                           ['label_BL'], test_size=kwargs.get('test_size', 0.2)))
    else:
        raise Exception(f"Split method should be 'random' or 'temporal', but is {method_split}.")
    return Xtrain, Xtest, ytrain.astype('int'), ytest.astype('int'), timestrain, timestest


def temporal_split(data, columns, label_columns=None, test_size=0.2):
    if label_columns is None:
        label_columns = ['label_BL']
    dftrain, dftest = pd.DataFrame([], columns=data.columns), pd.DataFrame([], columns=data.columns)
    months = pd.date_range(start=data.index.values[0], end=data.index.values[-1], freq=timedelta(days=30))
    for i in range(len(months) - 1):
        temp = data[months[i]:months[i + 1]].iloc[:-1,:]
        # The goal is to not take the last point, as it will also be part of the next month
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
            dftrain.loc[:, label_columns].values, dftest.loc[:, label_columns].values,
            dftrain.index.values, dftest.index.values)


def compute_scores(pred, truth):
    TP = (pred * truth).sum()
    FP = (pred * (1 - truth)).sum()
    TN = ((1 - pred) * (1-truth)).sum()
    FN = ((1 - pred) * truth).sum()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return TP, FP, TN, FN, precision, recall


def train_model(df, columns, **kwargs):
    warnings.filterwarnings("ignore")
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
        xtrain, xtest, ytrain, ytest, timestrain, timestest = split(data, columns,
                                                                    method_split=method_split,
                                                                    test_size=test_size)

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

    return precisions, recalls, AUCs, xtrain, xtest, ytrain, ytest, timestrain, timestest, gb


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
        if 'ax' not in kwargs:
            _, ax = plt.subplots()
        else:
            ax = kwargs['ax']

        ax.scatter(FPRs, TPRs)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), linestyle='--', color='grey', alpha=0.5)
        ax.set_title(f'ROC, AUC = {round(auc_value, 3)}')
    return FPRs, TPRs, auc_value


def recall_precision_curve(model, xtest, ytest, **kwargs):
    proba = model.predict_proba(xtest)[:, 1]
    ytest = ytest.flatten()

    precisions, recalls = [], []
    for threshold in np.linspace(0, 1, 100):
        pred = proba > threshold
        _, _, _, _, precision, recall = compute_scores(pred, ytest)
        precisions.append(precision)
        recalls.append(recall)

    if 'ax' not in kwargs:
        _, ax = plt.subplots()
    else:
        ax = kwargs['ax']
    ax.scatter(recalls, precisions)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.axhline(ytest.sum() / len(ytest), linestyle='--', color='grey', alpha=0.5)
    return recalls, precisions


def learning_curve(model, xtrain, ytrain, xtest, ytest, **kwargs):
    ytrain = ytrain.flatten()
    train_loss = [np.mean((np.array(proba[:, 1]) - np.array(ytrain)) ** 2) for proba in
                  model.staged_predict_proba(xtrain)]

    ytest = ytest.flatten()
    test_loss = [np.mean((np.array(proba[:, 1]) - np.array(ytest)) ** 2) for proba in
                 model.staged_predict_proba(xtest)]

    if 'ax' not in kwargs:
        _, ax = plt.subplots()
    else:
        ax = kwargs['ax']

    ax.plot(np.arange(len(train_loss)), train_loss, label='Train loss')
    ax.plot(np.arange(len(test_loss)), test_loss, label='Test loss')
    ax.legend()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE')
    ax.set_title('Loss through training')

    return train_loss, test_loss


def diagnostic(gb, xtrain, ytrain, xtest, ytest, **kwargs):
    fig, ax = plt.subplots(ncols=3, figsize=(15,5))
    train_loss, test_loss = learning_curve(gb, xtrain, ytrain, xtest, ytest, ax=ax[0])
    recalls, precisions = recall_precision_curve(gb, xtest, ytest, ax=ax[1])
    FPRs, TPRs, auc_value = ROC(gb, xtest, ytest, verbose=True, ax=ax[2])
    diag = {'train_loss': train_loss, 'test_loss': test_loss,
            'recalls': recalls, 'precisions': precisions,
            'FPRs': FPRs, 'TPRs': TPRs, 'auc_value': auc_value}
    fig.tight_layout()
    name= kwargs.get('name',str(datetime.now())[:10])
    fig.savefig(f'/home/ghisalberti/GradientBoosting/diagnostics/diag_{name}.jpg')
    return diag


def order_feature_importances(model, columns):
    df = pd.DataFrame(model.feature_importances_, columns=['feature_importance'], index=columns)
    return df.sort_values(by='feature_importance')


def plot_feature_importance(model, columns, **kwargs):
    importances = order_feature_importances(model, columns)
    if 'ax' in kwargs:
        ax = kwargs['ax']
    else:
        _, ax = plt.subplots()

    ax.scatter(np.arange(len(importances)), importances.values, s=50, marker='+')
    _ = ax.xticks(range(len(importances)), importances.index.values, fontsize=10, rotation=45)


def all_pred(model, df, columns):
    scaler = StandardScaler()
    data = df[columns].copy().dropna()
    data.loc[:, :] = scaler.fit_transform(data.loc[:, :].values)

    pred = model.predict_proba(data.values)[:, 1]
    pred_df = pd.DataFrame(pred, index=data.index.values, columns=['pred_proba_gradboost'])
    pred_df = pred_df.resample('5S').mean()
    return pred_df


def effect_trainset_size(train_proportions, df, columns, **kwargs):
    verbose_trainset_effect = kwargs.pop('verbose_trainset_effect', False)
    n_iter = kwargs.pop('n_iter', 5)

    median_precisions, median_recalls, median_aucs = [], [], []
    std_precisions, std_recalls, std_aucs = [], [], []
    train_sizes = []
    for tp in train_proportions:
        print(f'Train proportion = {tp}:')
        results = train_model(df, columns, method_split='temporal',
                              n_iter=n_iter, test_size=1 - tp, **kwargs)

        precisions, recalls, AUCs, Xtrain, Xtest, ytrain, ytest, timestrain, timestest, gb = results
        train_sizes.append(len(Xtrain))
        median_precisions.append(np.median(np.array(precisions)))
        median_recalls.append(np.median(np.array(recalls)))
        median_aucs.append(np.median(np.array(AUCs)))
        std_precisions.append(np.std(np.array(precisions)))
        std_recalls.append(np.std(np.array(recalls)))
        std_aucs.append(np.std(np.array(AUCs)))

    if verbose_trainset_effect:
        if 'ax' in kwargs:
            ax = kwargs['ax']
        else:
            _, ax = plt.subplots()
        ax.errorbar(train_sizes, median_precisions, yerr=std_precisions, label='Precision')
        ax.errorbar(train_sizes, median_recalls, yerr=std_recalls, label='Recall')
        ax.errorbar(train_sizes, median_aucs, yerr=std_aucs, label='AUC')
        ax.legend()
        ax.set_xlabel('Number of points in trainset')
        ax.set_ylabel('Scores')
        ax.set_title('Effect on trainset size on performance')

    return (median_precisions, median_recalls, median_aucs,
            std_precisions, std_recalls, std_aucs, train_sizes)


def get_all_feature_combinaisons(features):
    all_combinaisons = [[]]
    for _ in features:
        all_combinaisons = ([comb + [0] for comb in all_combinaisons] +
                            [comb + [1] for comb in all_combinaisons])

    features = np.array(features)
    all_combinaisons = np.array(all_combinaisons)
    combinaisons_features = []
    for comb in all_combinaisons:
        combinaisons_features += [list(features[comb.astype('bool')])]
    return combinaisons_features
