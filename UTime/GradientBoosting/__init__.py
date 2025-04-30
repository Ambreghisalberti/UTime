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
from scipy.signal import medfilt
import shap


def split(all_data, columns, **kwargs):
    method_split = kwargs.get('method_split', 'random')
    if method_split == 'random':
        all_data['time'] = all_data.index.values
        Xtrain, Xtest, ytrain, ytest = train_test_split(all_data.loc[:, list(columns)+['time']].values,
                                                        all_data.label_BL.values,
                                                        test_size=kwargs.get('test_size', 0.2))
        timestrain = Xtrain[:, -1]
        timestest = Xtest[:, -1]
        Xtrain = Xtrain[:, :-1]
        Xtest = Xtest[:, :-1]

    elif method_split == 'temporal':
        Xtrain, Xtest, ytrain, ytest, timestrain, timestest = (
            temporal_split(all_data.loc[:, list(columns) + ['label_BL']], columns,
                           ['label_BL'], test_size=kwargs.pop('test_size', 0.2),
                           **kwargs))
    else:
        raise Exception(f"Split method should be 'random' or 'temporal', but is {method_split}.")
    return Xtrain, Xtest, ytrain.astype('int'), ytest.astype('int'), timestrain, timestest


'''
def temporal_split(data, columns, label_columns=None, test_size=0.2, **kwargs):
    if label_columns is None:
        label_columns = ['label_BL']
    dftrain, dftest = pd.DataFrame([], columns=data.columns), pd.DataFrame([], columns=data.columns)
    months = pd.date_range(start=data.index.values[0], end=data.index.values[-1],
                           freq=kwargs.get('freq_split', timedelta(days=30)))
    for i in range(len(months) - 1):
        temp = data[months[i]:months[i + 1]].iloc[:-1, :]
        # The goal is to not take the last point, as it will also be part of the next month
        len_test = int(len(temp) * test_size)
        indice = rd.randint(0, len(temp) - len_test)
        temp_test = temp.iloc[indice:indice + len_test]
        dftest = pd.concat((dftest, pd.DataFrame(temp_test.values, index=temp_test.index.values, columns=data.columns)))
        temp_train = pd.concat((temp.iloc[:indice], temp.iloc[indice + len_test:]))
        dftrain = pd.concat(
            (dftrain, pd.DataFrame(temp_train.values, index=temp_train.index.values, columns=data.columns)))
        if len(temp_test) > 0:
            assert len(data[temp_test.index.values[0]:temp_test.index.values[-1]]) == len(temp_test), \
                ("The monthly testset portion is missing values from the dataset (it has more holes than "
                 "the original dataset)")

    assert len(
        dftrain[dftrain.index.isin(dftest.index)]) == 0, "Trainset and testset should not have any point in common!"

    timestest = dftest.index.values
    for i in range(len(months) - 1):
        test = timestest[timestest >= months[i]]
        test = test[test < months[i + 1]]
        if len(test) > 0:
            temp = data[test[0]:test[-1]]
            assert len(temp) == len(test), f"In the month {i}, some testset dates have holes."

    return (dftrain.loc[:, columns].values, dftest.loc[:, columns].values,
            dftrain.loc[:, label_columns].values, dftest.loc[:, label_columns].values,
            dftrain.index.values, dftest.index.values)
'''


def temporal_split(data, columns, label_columns=None, test_size=0.2,
                   resolution=np.timedelta64(5,'s'), **kwargs):
    data['date'] = data.index.values
    data = pd.DataFrame(data.values, columns=data.columns.values)

    if label_columns is None:
        label_columns = ['label_BL']
    timestrain, timestest = [], []
    months = np.array(list(pd.date_range(start=data.date.values[0], end=data.date.values[-1],
                                         freq=kwargs.get('freq_split', timedelta(days=30)))) + [
                          pd.to_datetime(data.date.values[-1] + resolution)])
    for i in range(len(months) - 1):
        temp = data[np.logical_and(data.date.values >= months[i], data.date.values < months[i + 1])].index.values
        # The goal is to not take the last point, as it will also be part of the next month
        len_test = int(len(temp) * test_size)
        indice = rd.randint(0, len(temp) - len_test)
        temp_test = list(temp[indice:indice + len_test])
        timestest += temp_test
        temp_train = list(temp[:indice]) + list(temp[indice + len_test:])
        timestrain += temp_train
        assert len(temp_train) + len(temp_test) == len(
            temp), f"Dataset should be split between train and test but {len(temp) - (len(temp_train) + len(temp_test))}/{len(temp)} points are left out for month {i}."

        """
        if len(temp_test) > 0:
            assert len(data[temp_test[0]:temp_test[-1]]) == len(temp_test), \
                ("The monthly testset portion is missing values from the dataset (it has more holes than "
                 "the original dataset)")
        """
    timestrain, timestest = np.array(timestrain), np.array(timestest)

    dftrain = data.loc[timestrain]  # Here what happens when the same date is here twice? Should give number indices instead of dates
    dftest = data.loc[timestest]

    dftrain = pd.DataFrame(dftrain.values, index=dftrain.date.values, columns=dftrain.columns.values).drop(
        columns=['date'])
    dftest = pd.DataFrame(dftest.values, index=dftest.date.values, columns=dftest.columns.values).drop(columns=['date'])

    if 'sat' in dftrain.columns:
        for sat in np.unique(dftrain.sat.values):
            subtrain = dftrain[dftrain.sat.values == sat]
            subtest = dftest[dftest.sat.values == sat]
            assert len(
                subtrain[subtrain.index.isin(subtest.index)]) == 0, \
                f"Trainset and testset should not have any point in common for {sat}!"
    else:
        assert len(dftrain[dftrain.index.isin(dftest.index)]) == 0, \
            "Trainset and testset should not have any point in common!"

    assert len(dftrain) + len(dftest) == len(data), (f"Dataset should be split between train and test but "
                                                     f"{len(data) - (len(dftrain) + len(dftest))}/{len(data)} "
                                                     f"points are left out.")

    return (dftrain.loc[:, columns].values, dftest.loc[:, columns].values,
            dftrain.loc[:, label_columns].values, dftest.loc[:, label_columns].values,
            timestrain, timestest)


def compute_scores(pred, truth, threshold=0.5):
    pred = (pred > threshold).astype(int)
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
    freq_split = kwargs.pop('freq_split', timedelta(days=30))

    precisions, recalls, AUCs = [], [], []
    xtests, xtrains, ytests, ytrains, timestests, timestrains, gbs = [], [], [], [], [], [], []

    model = kwargs.pop('model', 'GBC')

    n_iter = kwargs.pop('n_iter', 1)
    assert n_iter > 0, f"n_iter must be strictly positive but is {n_iter}"

    for i in range(n_iter):
        xtrain, xtest, ytrain, ytest, timestrain, timestest = split(data, columns,
                                                                    method_split=method_split,
                                                                    test_size=test_size,
                                                                    freq_split=freq_split)
        xtrains.append(xtrain)
        ytrains.append(ytrain)
        timestrains.append(timestrain)
        xtests.append(xtest)
        ytests.append(ytest)
        timestests.append(timestest)

        if model == 'GBC':
            gb = Gbc(verbose=model_verbose, **kwargs)
        elif model == 'HGBC':
            gb = Hgbc(verbose=model_verbose, **kwargs)
        else:
            raise Exception("Model should be GBC or HGBC")
        gb.fit(xtrain, ytrain)
        gbs.append(gb)

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

    return precisions, recalls, AUCs, xtrains, xtests, ytrains, ytests, timestrains, timestests, gbs


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


def recall_precision_curve(pred_proba, label, **kwargs):
    if 'pred' in kwargs:
        pred_proba = kwargs['pred']
    elif 'model' in kwargs and 'xtest' in kwargs:
        pred_proba = kwargs['model'].predict_proba(kwargs['xtest'])[:, 1]
    label = label.flatten()

    precisions, recalls = [], []
    for threshold in np.linspace(0, 1, 100):
        _, _, _, _, precision, recall = compute_scores(pred_proba, label, threshold=threshold)
        precisions.append(precision)
        recalls.append(recall)

    if 'ax' not in kwargs:
        _, ax = plt.subplots()
    else:
        ax = kwargs['ax']
    ax.scatter(recalls, precisions)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.axhline(label.sum() / len(label), linestyle='--', color='grey', alpha=0.5)
    return recalls, precisions


def find_best_threshold(pred_proba, label, **kwargs):
    thresholds = np.linspace(0, 1, 100)
    recalls, precisions = recall_precision_curve(pred_proba, label, **kwargs)
    i = np.nanargmax(np.array(precisions)*np.array(recalls)).item()
    return thresholds[i]


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
    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
    train_loss, test_loss = learning_curve(gb, xtrain, ytrain, xtest, ytest, ax=ax[0])
    recalls, precisions = recall_precision_curve(gb, xtest, ytest, ax=ax[1])
    FPRs, TPRs, auc_value = ROC(gb, xtest, ytest, verbose=True, ax=ax[2])
    diag = {'train_loss': train_loss, 'test_loss': test_loss,
            'recalls': recalls, 'precisions': precisions,
            'FPRs': FPRs, 'TPRs': TPRs, 'auc_value': auc_value}
    fig.tight_layout()
    name = kwargs.get('name', str(datetime.now())[:10])
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

        precisions, recalls, AUCs, Xtrains, _, _, _, _, _, _ = results
        train_sizes.append(len(Xtrains[0]))
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


def scores_median_pred(pred, ytest, kernel_size, verbose=False):
    pred_test_med = medfilt(pred, kernel_size=kernel_size)
    pred_test_med = pred_test_med > 0.5

    TP_med = np.logical_and(pred_test_med, ytest.astype('bool')).sum()
    FP_med = np.logical_and(pred_test_med, np.logical_not(ytest.astype('bool'))).sum()
    TN_med = np.logical_and(np.logical_not(pred_test_med), np.logical_not(ytest.astype('bool'))).sum()
    FN_med = np.logical_and(np.logical_not(pred_test_med), ytest.astype('bool')).sum()
    assert len(pred) == TP_med + FP_med + FN_med + TN_med, "Points should be in TP, FP, TN or FN!"

    precision = TP_med / (TP_med + FP_med)
    recall = TP_med / (TP_med + FN_med)

    if verbose:
        print(
            f'After a median filter with kernel size = {kernel_size}, the {len(pred)} points in testset are '
            f'{TP_med} points of true positives, {FP_med} point of false positives, '
            f'{FN_med} points of false negatives, and {TN_med} points of true negatives,'
            f'\ngiving a precision of {round(precision * 100, 2)}% and a recall of {round(recall * 100, 2)}%.')

    return precision, recall


def get_all_feature_choices(features_yes_or_no, features_multiple_choice):
    all_combinaisons = [[]]
    for _ in features_yes_or_no:
        all_combinaisons = ([comb + [0] for comb in all_combinaisons] +
                            [comb + [1] for comb in all_combinaisons])

    for choices in features_multiple_choice:
        nb_choices = len(choices)
        all_new_combinaisons = []
        for i in range(nb_choices):
            all_new_combinaisons += [comb + [i] for comb in all_combinaisons]
        all_combinaisons = all_new_combinaisons

    return all_combinaisons


def get_all_feature_combinaisons_multiple_choices(features_yes_or_no, features_multiple_choice, features_mandatory):
    all_combinaisons = np.array(get_all_feature_choices(features_yes_or_no, features_multiple_choice))

    combinaisons_features = []
    for comb in all_combinaisons:
        comb_features = features_mandatory.copy()
        for i, features in enumerate(features_yes_or_no):
            if comb[i] == 1:
                comb_features += features
        for i, choices in enumerate(features_multiple_choice):
            choice = comb[len(features_yes_or_no) + i]
            comb_features += choices[choice]
        combinaisons_features += [comb_features]
    return combinaisons_features


def get_name_choice(choice):
    temp = str(choice)[1:-1].split(', ')
    temp2 = ''
    for t in temp:
        temp2 += t
    return temp2


def add_scores(ytest, precisions, recalls, threshold=0.5, **kwargs):
    if 'pred' in kwargs:
        pred = kwargs['pred']
    elif ('model' in kwargs) and ('xtest' in kwargs):
        pred = kwargs['model'].predict_proba(kwargs['xtest'])[:,1]

    ytest = ytest.flatten()
    TP, FP, TN, FN, precision, recall = compute_scores(pred, ytest, threshold=threshold)
    precisions.append(precision)
    recalls.append(recall)
    return precisions, recalls


def split_and_fit(data, columns, method_split, test_size, gbs, **kwargs):
    xtrain, xtest, ytrain, ytest, timestrain, timestest = split(data, columns,
                                                                method_split=method_split,
                                                                test_size=test_size,
                                                                freq_split=kwargs.get('freq_split', timedelta(days=30)))

    model = kwargs.get('model', 'GBC')
    verbose = kwargs.get('model_verbose', True)
    if model == 'GBC':
        gb = Gbc(verbose=verbose)
    elif model == 'HGBC':
        gb = Hgbc(verbose=verbose, max_iter=kwargs.get('max_iter', 50))
    else:
        raise Exception("Model should be GBC or HGBC")
    gb.fit(xtrain, ytrain)
    gbs.append(gb)

    return xtest, ytest, gbs


def x_y_set(df, scaler, columns):
    y = df.label_BL.values
    x = df[columns]
    x.loc[:, :] = scaler.transform(x.values)
    return x, y


def fit_and_assess_on_different_and_common_testsets(df, testset2, columns, **kwargs):
    warnings.filterwarnings("ignore")
    scaler = StandardScaler()
    data = df.copy().dropna()
    data.loc[:, columns] = scaler.fit_transform(data.loc[:, columns].values)

    xtest2, ytest2 = x_y_set(testset2, scaler, columns)

    method_split = kwargs.pop('method_split', 'temporal')
    test_size = kwargs.pop('test_size', 0.2)

    precisions, recalls = [], []
    gbs = []
    precisions2, recalls2 = [], []

    n_iter = kwargs.pop('n_iter', 1)
    assert n_iter > 0, f"n_iter must be strictly positive but is {n_iter}"

    for i in range(n_iter):
        xtest, ytest, gbs = split_and_fit(data, columns, method_split, test_size, gbs, **kwargs)
        precisions, recalls = add_scores(ytest, precisions, recalls, model=gbs[-1], xtest=xtest)
        precisions2, recalls2 = add_scores(ytest2, precisions2, recalls2, model=gbs[-1],
                                           xtest=xtest2)

    if kwargs.get('verbose', False):
        print(
            f'Over {len(precisions)} runs, the precision has '
            f'a standard deviation of {round(np.std(precisions) * 100, 2)}% and '
            f'a span of {round((np.max(precisions) - np.min(precisions)) * 100, 2)}%, and '
            f'the recall has a standard deviation of {round(np.std(recalls) * 100, 2)}% and '
            f'a span of {round((np.max(recalls) - np.min(recalls)) * 100, 2)}%.')
        print(
            f'On a common testset, the precision has a standard deviation of '
            f'{round(np.std(precisions2) * 100, 2)}% and '
            f'a span of {round((np.max(precisions2) - np.min(precisions2)) * 100, 2)}%, '
            f'and the recall has a standard deviation of {round(np.std(recalls2) * 100, 2)}% '
            f'and a span of {round((np.max(recalls2) - np.min(recalls2)) * 100, 2)}%.')

    return precisions, recalls, precisions2, recalls2, gbs


def measure_variability_scores(subdata, columns, **kwargs):
    all_precisions, all_recalls, all_precisions_common, all_recalls_common = [], [], [], []

    nb_trys = kwargs.get('nb_trys', 5)
    for i in range(nb_trys):
        _, _, _, _, df_times, testset2_times = temporal_split(subdata, ['label_BL'],
                                                              test_size=0.2,
                                                              freq_split=timedelta(days=60))
        testset2 = subdata.loc[testset2_times]
        df = subdata.loc[df_times]

        (precisions, recalls,
         precisions2, recalls2, gbs) = fit_and_assess_on_different_and_common_testsets(df, testset2,
                                                                                       columns, **kwargs)
        print('\n')
        all_precisions += [precisions]
        all_recalls += [recalls]
        all_precisions_common += [precisions2]
        all_recalls_common += [recalls2]

    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))

    for i, (precisions, recalls, precisions2, recalls2) in enumerate(
            zip(all_precisions, all_recalls, all_precisions_common, all_recalls_common)):
        ax[0].scatter(precisions, precisions2, label='Precisions', color='blue', s=3)
        ax[0].scatter(recalls, recalls2, label='Recalls', color='orange', s=3)
        if i == 0:
            ax[0].legend()
    ax[0].plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), linestyle='--', color='grey', alpha=0.5)
    ax[0].set_xlabel('On random testset')
    ax[0].set_ylabel('On same unseen testset')

    ax[1].errorbar(np.arange(len(all_precisions)), [np.median(np.array(precisions)) for precisions in all_precisions],
                   yerr=[np.std(np.array(precisions)) for precisions in all_precisions],
                   label='On corresponding testsets')
    ax[1].errorbar(np.arange(len(all_precisions)),
                   [np.median(np.array(precisions)) for precisions in all_precisions_common],
                   yerr=[np.std(np.array(precisions)) for precisions in all_precisions_common],
                   label='On common testset')
    ax[1].set_xlabel("Different common testsets")
    ax[1].set_ylabel("Precision")
    ax[1].legend()

    ax[2].errorbar(np.arange(len(all_recalls)), [np.median(np.array(recalls)) for recalls in all_recalls],
                   yerr=[np.std(np.array(recalls)) for recalls in all_recalls], label='On corresponding testsets')
    ax[2].errorbar(np.arange(len(all_recalls)), [np.median(np.array(recalls)) for recalls in all_recalls_common],
                   yerr=[np.std(np.array(recalls)) for recalls in all_recalls_common], label='On common testset')
    ax[2].set_xlabel("Different common testsets")
    ax[2].set_ylabel("Precision")
    ax[2].legend()

    fig.suptitle(f"For a train/test split on every chunk of {str(kwargs.get('freq_split', timedelta(days=30)))}.")
    fig.tight_layout()

    median_precision_std = np.median(np.array([np.std(np.array(precisions)) for precisions in all_precisions]))
    median_precision_common_std = np.median(
        np.array([np.std(np.array(precisions)) for precisions in all_precisions_common]))
    median_recall_std = np.median(np.array([np.std(np.array(recalls)) for recalls in all_recalls]))
    median_recall_common_std = np.median(np.array([np.std(np.array(recalls)) for recalls in all_recalls_common]))

    print(
        f'The precision on random testsets has a standard deviation of {round(median_precision_std * 100, 2)}%, '
        f'and of {round(median_precision_common_std * 100, 2)}% on common testset.')
    print(
        f'The recall on random testsets has a standard deviation of {round(median_recall_std * 100, 2)}%, '
        f'and of {round(median_recall_common_std * 100, 2)}% on common testset.')

    return (all_precisions, all_recalls,
            all_precisions_common, all_recalls_common,
            median_precision_std, median_precision_common_std,
            median_recall_std, median_recall_common_std)


def check_effect_freq_split_on_variability(subdata, columns, split_frequencies, **kwargs):
    all_median_precision_std = []
    all_median_precision_common_std = []
    all_median_recall_std = []
    all_median_recall_common_std = []
    all_median_precision_span = []
    all_median_precision_common_span = []
    all_median_recall_span = []
    all_median_recall_common_span = []

    results = {}
    for freq_split in split_frequencies:
        (all_precisions, all_recalls,
         all_precisions_common, all_recalls_common,
         median_precision_std, median_precision_common_std,
         median_recall_std, median_recall_common_std) = measure_variability_scores(subdata,
                                                                                   columns,
                                                                                   freq_split=freq_split,
                                                                                   **kwargs)
        all_median_precision_std += [median_precision_std]
        all_median_precision_common_std += [median_precision_common_std]
        all_median_recall_std += [median_recall_std]
        all_median_recall_common_std += [median_recall_common_std]
        all_median_precision_span += [
            np.median(np.array([np.max(precisions) - np.min(precisions) for precisions in all_precisions]))]
        all_median_precision_common_span += [
            np.median(np.array([np.max(precisions) - np.min(precisions) for precisions in all_precisions_common]))]
        all_median_recall_span += [np.median(np.array([np.max(recalls) - np.min(recalls) for recalls in all_recalls]))]
        all_median_recall_common_span += [
            np.median(np.array([np.max(recalls) - np.min(recalls) for recalls in all_recalls_common]))]
        results = {'description': f'On {kwargs.get("n_iter", 100)} runs, assessment of precision and '
                                  f'recall standard deviation, '
                                  'on each testset, and on a common testset. These results are averaged over '
                                  f'{kwargs.get("nb_trys", 5)} repetitions.', 'frequencies': split_frequencies,
                   'all_median_precision_std': all_median_precision_std,
                   'all_median_precision_common_std': all_median_precision_common_std,
                   'all_median_recall_std': all_median_recall_std,
                   'all_median_recall_common_std': all_median_recall_common_std,
                   'all_median_precision_span': all_median_precision_span,
                   'all_median_precision_common_span': all_median_precision_common_span,
                   'all_median_recall_span': all_median_recall_span,
                   'all_median_recall_common_span': all_median_recall_common_span}

        pd.to_pickle(results, '/home/ghisalberti/GradientBoosting/diagnostics/'
                              'effect_freq_split_on_variability.pkl')

    if kwargs.get('verbose', True):
        text = ''
        for i in range(len(split_frequencies)):
            text += (f'For a split period of {split_frequencies[i]}, we have a precision standard deviation of '
                     f'{round(all_median_precision_std[i] * 100, 2)}% on normal testsets, and of '
                     f'{round(all_median_precision_common_std[i] * 100, 2)}% on a common testset;')
            text += (f'and a recall standard deviation of {round(all_median_recall_std[i] * 100, 2)}% '
                     f'on normal testsets, and of {round(all_median_recall_common_std[i] * 100, 2)}% '
                     f'on a common testset.\n')
        print(text)

    return results


def plot_effect_freq_split_on_variability(results, **kwargs):
    frequencies = results['frequencies']
    all_median_precision_std = results['all_median_precision_std']
    all_median_precision_common_std = results['all_median_precision_common_std']
    all_median_recall_std = results['all_median_recall_std']
    all_median_recall_common_std = results['all_median_recall_common_std']
    all_median_precision_span = results['all_median_precision_span']
    all_median_precision_common_span = results['all_median_precision_common_span']
    all_median_recall_span = results['all_median_recall_span']
    all_median_recall_common_span = results['all_median_recall_common_span']

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

    ax[0].plot([freq.days for freq in frequencies], all_median_precision_std, label='Precision variability')
    ax[0].plot([freq.days for freq in frequencies], all_median_precision_common_std,
               label='Precision variability, common testset')
    ax[0].plot([freq.days for freq in frequencies], all_median_recall_std, label='Recall variability')
    ax[0].plot([freq.days for freq in frequencies], all_median_recall_common_std,
               label='Recall variability, common testset')
    ax[0].legend()
    ax[0].set_xlabel('Split period, in days')
    ax[0].set_ylabel('Standard deviation')

    ax[1].plot([freq.days for freq in frequencies], all_median_precision_span, label='Precision variability')
    ax[1].plot([freq.days for freq in frequencies], all_median_precision_common_span,
               label='Precision variability, common testset')
    ax[1].plot([freq.days for freq in frequencies], all_median_recall_span, label='Recall variability')
    ax[1].plot([freq.days for freq in frequencies], all_median_recall_common_span,
               label='Recall variability, common testset')
    ax[1].legend()
    ax[1].set_xlabel('Split period, in days')
    ax[1].set_ylabel('Score span')

    fig.suptitle('Effect of temporal split period on variability of scores')
    fig.savefig(f'/home/ghisalberti/GradientBoosting/diagnostics/score_variability_'
                f'depending_on_freq_split_{kwargs.get("name",str(datetime.now())[:10])}.pkl')


def ensemble_learning_on_same_model(data, columns, **kwargs):
    ensemble_precisions, ensemble_recalls, median_precisions, median_recalls = [], [], [], []
    n_iter = kwargs.get('n_models_ensemble', 10)
    n_trys = kwargs.get('n_repetitions', 30)
    verbose = kwargs.pop('verbose', False)

    for i in range(n_trys):
        _, _, _, _, df_times, testset2_times = temporal_split(data, ['label_BL'],
                                                              test_size=0.2,
                                                              freq_split=timedelta(days=60))
        testset2 = data.loc[testset2_times]
        df = data.loc[df_times]
        scaler = StandardScaler()
        scaler.fit(df.loc[:, columns].values)
        xtest2, ytest2 = x_y_set(testset2, scaler, columns)

        (precisions, recalls,
         precisions2, recalls2, gbs) = fit_and_assess_on_different_and_common_testsets(df, testset2,
                                                                                       columns,
                                                                                       n_iter=n_iter,
                                                                                       verbose=False,
                                                                                       **kwargs)
        pred = np.zeros(len(testset2))
        for gb in gbs:
            pred += gb.predict_proba(xtest2)[:, 1]
        pred = pred / n_iter

        ensemble_precisions, ensemble_recalls = add_scores(ytest2, ensemble_precisions, ensemble_recalls, pred=pred)
        median_precisions += [np.median(np.array(precisions2))]
        median_recalls += [np.median(np.array(recalls2))]

        if verbose:
            print(f'For ensemble learning: precision = {round(ensemble_precisions[-1] * 100, 2)}% '
                  f'and recall = {round(ensemble_recalls[-1] * 100, 2)}%, from models '
                  f'with average precision = {round(np.median(np.array(precisions2)) * 100, 2)}% '
                  f'and average recall = {round(np.median(np.array(recalls2)) * 100, 2)}%.')

    if verbose:
        plt.figure()
        _ = plt.hist(np.array(ensemble_precisions) - np.array(median_precisions), bins=50, alpha=0.5,
                     label='precisions')
        _ = plt.hist(np.array(ensemble_recalls) - np.array(median_recalls), bins=50, alpha=0.5, label='recalls')
        plt.legend()
        plt.xlabel('Scores')
        plt.ylabel('Count')
        plt.title(f'Score gain from single models to ensemble models from {n_iter} single ones')

    return (np.array(ensemble_precisions), np.array(ensemble_recalls),
            np.array(median_precisions), np.array(median_recalls))


def ensemble_learning_on_different_features(data, list_features, **kwargs):
    warnings.filterwarnings("ignore")

    (ensemble_precisions, ensemble_recalls, median_precisions, median_recalls,
     best_model_precisions, best_model_recalls, timestests) = [], [], [], [], [], [], []
    verbose = kwargs.pop('verbose', False)
    model = kwargs.get('model', 'HGBC')
    n_repetitions = kwargs.get('n_repetitions', 10)

    for i in range(n_repetitions):
        models = []
        _, _, _, _, timestrain, timestest = temporal_split(data, ['label_BL'],
                                                           test_size=0.2,
                                                           freq_split=kwargs.get('freq_split', timedelta(days=30)))
        timestests += [timestest]
        testset = data.loc[timestest]
        trainset = data.loc[timestrain]
        precisions, recalls = [], []

        pred = np.zeros(len(testset))
        for columns in list_features:
            scaler = StandardScaler()
            scaler.fit(trainset.loc[:, columns].values)
            xtest, ytest = x_y_set(testset, scaler, columns)
            xtrain, ytrain = x_y_set(trainset, scaler, columns)

            if model == 'GBC':
                gb = Gbc(verbose=False)
            elif model == 'HGBC':
                gb = Hgbc(verbose=False, max_iter=kwargs.get('max_iter', 50))
            else:
                raise Exception("Model should be GBC or HGBC")

            gb.fit(xtrain, ytrain)
            precisions, recalls = add_scores(ytest, precisions, recalls, model=gb, xtest=xtest)
            pred += gb.predict_proba(xtest)[:, 1]
            models += [gb]

        pred = pred / len(list_features)

        ensemble_precisions, ensemble_recalls = add_scores(ytest, ensemble_precisions, ensemble_recalls, pred=pred)
        median_precisions += [np.median(np.array(precisions))]
        median_recalls += [np.median(np.array(recalls))]
        best_model_precisions += [np.max(np.array(precisions))]
        best_model_recalls += [np.array(recalls)[np.argmax(np.array(precisions))]]

        if verbose:
            print(
                f'For ensemble learning: precision = {round(ensemble_precisions[-1] * 100, 2)}% '
                f'and recall = {round(ensemble_recalls[-1] * 100, 2)}%, '
                f'from models with average precision = {round(np.median(np.array(precisions)) * 100, 2)}% '
                f'and average recall = {round(np.median(np.array(recalls)) * 100, 2)}%,\n'
                f'and a best model with precision = {round(np.max(np.array(precisions)) * 100, 2)}% '
                f'and recall = {round(np.array(recalls)[np.argmax(np.array(precisions))] * 100, 2)}%.')

    if kwargs.get('plot_verbose', True):
        fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

        _ = ax[0].hist(np.array(ensemble_precisions) - np.array(best_model_precisions), bins=50, alpha=0.5,
                       label='precisions')
        _ = ax[0].hist(np.array(ensemble_recalls) - np.array(best_model_recalls), bins=50, alpha=0.5, label='recalls')
        ax[0].legend()
        ax[0].set_xlabel('Scores')
        ax[0].set_ylabel('Count')

        ax[1].scatter(np.array(ensemble_precisions), np.array(best_model_precisions), s=3, label='precisions')
        ax[1].scatter(np.array(ensemble_recalls), np.array(best_model_recalls), s=3, label='recalls')
        ax[1].legend()
        ax[1].plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), linestyle='--', color='grey', alpha=0.5)
        ax[1].set_xlabel('Ensemble scores')
        ax[1].set_ylabel('Best model scores')

        fig.suptitle(f'Score gain from best model to ensemble model from {len(list_features)} single ones')

    return (np.array(ensemble_precisions), np.array(ensemble_recalls),
            np.array(median_precisions), np.array(median_recalls),
            np.array(best_model_precisions), np.array(best_model_recalls), np.array(models),
            np.array(timestests))


def get_all_features_ensemble_model(model_results):
    all_features = []
    for features in model_results['all_features']:
        all_features += features
    all_features = list(np.unique(np.array(all_features)).astype('str'))
    return all_features


def pred_ensemble(x, model_results):
    pred = np.zeros(len(x))
    for model, scaler, features in zip(model_results['models'], model_results['scalers'],
                                       model_results['all_features']):
        pred += model.predict_proba(scaler.transform(x[features]))[:, 1]
    pred = pred / len(model_results['models'])
    return pred


def compute_shap_values_ensemble_learning(model_results, data, **kwargs):
    start = kwargs.get('start', data.index.values[0])
    stop = kwargs.get('stop', data.index.values[-1])

    all_features = get_all_features_ensemble_model(model_results)
    shap_values = pd.DataFrame(np.zeros(data.loc[start:stop, all_features].shape),
                               index=data.loc[start:stop].index.values, columns=all_features)
    count = 0
    for model, scaler, features in zip(model_results['models'], model_results['scalers'],
                                       model_results['all_features']):
        count += 1

        def pred(x):
            return model.predict_proba(scaler.transform(x))[:, 1]

        explainer = shap.Explainer(pred, data[features])
        vals = explainer.shap_values(data.loc[start:stop, features])
        shap_values[features] = shap_values[features].values + vals
        print(f'Model {count} Shapley values done.')

    shap_values = shap_values / len(model_results['models'])

    return shap_values


def fit_ensemble(dftrain, model_results):
    """
    Fits an ensemble of models, with the features and scalers given in model_results.
    """
    models = []

    for scaler, features in zip(model_results['scalers'], model_results['all_features']):
        # Scale THA data with MMS scaler
        values = scaler.transform(dftrain[np.array(features)].values)

        # fit the model
        model = Hgbc()
        model = model.fit(values, dftrain.label_BL.values)
        models += [model]

    return models


def refit_ensemble(dftrain, model_results):
    """
    Applies transfert learning to an ensemble of models,
    with the models, features and scalers given in model_results.
    """
    models = []

    for scaler, features, model in zip(model_results['scalers'], model_results['all_features'],
                                       model_results['models']):
        # Scale THA data with MMS scaler
        values = scaler.transform(dftrain[np.array(features)].values)

        # fit the model
        model = model.fit(values, dftrain.label_BL.values)
        models += [model]

    return models


def scores(pred, label, threshold=0.5):
    pred_class = pred > threshold

    TP = (pred_class * label).sum()
    FP = (pred_class * (1 - label)).sum()
    FN = ((1 - pred_class) * label).sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return precision, recall


def make_learning_curve(df, model_results):
    precisions, recalls = [], []
    test_sizes = np.linspace(0.05, 0.95, 50)
    all_features = get_all_features_ensemble_model(model_results)

    for test_size in test_sizes:
        Xtrain, Xtest, ytrain, ytest, timestrain, timestest = split(df, all_features,
                                                                    method_split='temporal',
                                                                    test_size=test_size)
        Xtrain = pd.DataFrame(Xtrain, columns=all_features)
        Xtrain['label_BL'] = ytrain
        Xtest = pd.DataFrame(Xtest, columns=all_features)

        models = fit_ensemble(Xtrain, model_results)
        pred = pred_ensemble(Xtest, {'all_features': model_results['all_features'],
                                     'scalers': model_results['scalers'],
                                     'models': models})
        precision, recall = scores(pred, ytest)
        precisions += [precision]
        recalls += [recall]

    plt.figure()
    plt.plot(test_sizes, precisions, label='precision')
    plt.plot(test_sizes, recalls, label='recall')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()


def get_features_from_choice(choice, mandatory, features_yes_or_no, features_multiple_choice):
    features = list(np.array(mandatory).copy())
    for i,c in enumerate(choice):
        if i < len(features_yes_or_no):
            if c == '1':
                features += features_yes_or_no[i]
        else:
            c = int(c)
            features += features_multiple_choice[i-len(features_yes_or_no)][c]
    return features
