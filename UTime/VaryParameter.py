import matplotlib.pyplot as plt
import pandas as pd
from .architectureResNet import UTime
from .CrossValidation import cross_validation, write_scores
import numpy as np
import collections.abc

# Cannot work for window_size yet
def duplicate_to_list(data, nb_values):
    if not (isinstance(data, list)) and not (isinstance(data, (collections.abc.Sequence, np.ndarray))):
        data = [data for i in range(nb_values)]
    if isinstance(data, str):
        data = [data for i in range(nb_values)]
    return data

'''
def transform_in_lists(dfs):
    for df in dfs:
        if isinstance(df, list):
            nb_values = len(df)
    for i in range(len(dfs)):
        dfs[i] = duplicate_to_list(dfs[i], nb_values)

    return dfs
'''

def get_variable(parameters):
    variable = ''
    nb_values = 1
    index = -1
    for key in parameters.keys():
        if not(isinstance(parameters[key], str)) and (isinstance(parameters[key], list) or isinstance(parameters[key], (collections.abc.Sequence, np.ndarray))):
            variable = key
            values = parameters[key]
            nb_values = len(values)
    return variable, values, nb_values


def transform_in_lists(parameters):
    variable, values, nb_values = get_variable(parameters)
    for key in parameters.keys():
        parameters[key] = duplicate_to_list(parameters[key], nb_values)
    return parameters, variable, values


def write_file(text, path):
    f = open(path, "w")
    f.write(text)
    f.close()


def vary_parameter(windows, **kwargs):
    plt.ion()
    depth = kwargs.pop('depth', 5)
    kernel_size = kwargs.pop('kernel_size', 5)
    nb_filters = kwargs.pop('nb_filters', 16)
    train_proportion = kwargs.get('train_proportion', 0.8)
    test_proportion = kwargs.get('test_proportion', 1 - train_proportion)
    if isinstance(train_proportion, (collections.abc.Sequence, np.ndarray)):
        test_proportion = [min(test_p, 1 - train_p) for test_p, train_p in zip(test_proportion, train_proportion)]
    else:
        test_proportion = min(test_proportion, 1 - train_proportion)

    loss_function = kwargs.pop('loss_function', 'MSE')
    if 'description' in kwargs:
        description = '_' + kwargs['description']
    else:
        description = ''

    text = ''
    scores = pd.DataFrame([],
                          columns=["mean_precision", "std_precision", "mean_recall", "std_recall",
                                   "mean_F1", "std_F1", "mean_AUC", "std_AUC"])

    parameters = {'depth': depth, 'kernel_size': kernel_size, 'nb_filters': nb_filters,
                  'test_proportion': test_proportion, 'train_proportion':train_proportion,
                  'loss_function': loss_function}
    parameters, variable, values = transform_in_lists(parameters)
    depth, kernel_size, nb_filters, test_proportion, train_proportion, loss_function = parameters.values()

    fig, axes = plt.subplots(ncols=2, nrows=len(depth), figsize=(6, 3 * len(depth)))
    plt.draw()
    nb_iter = kwargs.pop('nb_iter', 5)

    for i, (d, ks, nf, te_p, tr_p, lf) in enumerate(zip(depth, kernel_size, nb_filters, test_proportion, train_proportion, loss_function)):
        value = values[i]
        name = f'depth={d}_filters={nf}_kernel={ks}_{lf}_testsize={te_p}_trainsize={tr_p}' + description
        architecture = UTime(1, windows.win_length, len(windows.moments_features),
                             len(windows.spectro_features), d, nf, ks, 2)
        cv = cross_validation(architecture, windows, nb_iter, lf, test_ratio=te_p, train_ratio=tr_p, fig=fig,
                              ax=axes[i, :], name=f'{variable} = {value}', **kwargs)
        pd.to_pickle(cv,f'/home/ghisalberti/BL_encoder_decoder/model/diagnostics/{name}.pkl')
        precisions, recalls, F1_scores = cv['precisions'], cv['recalls'], cv['F1_scores']
        TPRs, FPRs, AUCs = cv['TPRs'], cv['FPRs'], cv['AUCs']
        text, scores = write_scores(text, scores, variable, value, precisions, recalls, F1_scores, AUCs)

        print(f'{variable} = {value} done.')
        print('\n\n')

    path = f"/home/ghisalberti/BL_encoder_decoder/model/diagnostics/{variable}_effect{description}"
    write_file(text, path + ".txt")
    plt.tight_layout()
    plt.ioff()
    plt.savefig(path + ".png")
    plt.close()

    compare_scores(scores, description=description)

    return scores


def compare_scores(scores, **kwargs):
    plt.figure()
    plt.errorbar(np.arange(len(scores)) - 0.1, scores.mean_precision.values,
                 yerr=scores.std_precision.values, fmt="o", label='Precision')
    plt.errorbar(np.arange(len(scores)), scores.mean_recall.values, yerr=scores.std_recall.values,
                 fmt="o", label='Recall')
    plt.errorbar(np.arange(len(scores)) + 0.1, scores.mean_AUC.values, yerr=scores.std_AUC.values,
                 fmt="o", label='AUC')

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    labels = np.array(scores.index.values)
    plt.xticks(ticks=np.arange(len(scores)), labels=labels, rotation=45)
    plt.ylabel("Scores, in %")

    variable = labels[0].split('=')[0]
    plt.xlabel(variable)
    plt.title(f"Effect of {variable} on scores")

    '''
    for i in range(len(scores)):
        plt.axhline(scores.mean_precision.values[i], alpha=0.2, linestyle='--', color='b')
        plt.axhline(scores.mean_recall.values[i], alpha=0.4, linestyle='--', color='orange')
    '''
    plt.savefig(f"/home/ghisalberti/BL_encoder_decoder/model/diagnostics/"
                f"{variable}_effect_on_scores{kwargs.get('description','')}.png")
    plt.show()