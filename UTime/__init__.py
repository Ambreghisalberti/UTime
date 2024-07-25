import torch
from torch.utils.data import Dataset
from swapp.windowing.make_windows import prepare_df
from swapp.windowing.make_windows.utils import select_windows, durationToNbrPts, time_resolution, nbr_windows
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataForWindows(Dataset):

    def __init__(self, all_data, position, omni_data, win_duration, moments_features = [], spectro_features = [],
                 label = ['label_BL'], **kwargs):
        self.win_duration = win_duration
        self.win_length = durationToNbrPts(win_duration, time_resolution(all_data))
        self.stride = kwargs.get('stride',self.win_length)
        self.moments_features = moments_features
        self.spectro_features = spectro_features
        self.ml_features = self.moments_features + self.spectro_features
        self.label = label

        is_prepared = kwargs.get('is_prepared', False)
        if is_prepared:
            self.df, self.pos, self.omni = all_data, position, omni_data
        else:
            label_paths = kwargs["label_paths"]
            labelled_days = kwargs["labelled_days"]
            self.df, self.pos, self.omni = prepare_df(all_data, position, omni_data, win_duration, label_paths,
                                                               labelled_days)
        self.omni = self.omni.rename(columns={col:"OMNI_"+col for col in self.omni.columns})
        self.omni = self.omni.ffill().bfill()

        self.dataset = pd.concat([self.df, self.omni], axis = 1)
        self.conditions = kwargs.get('conditions',['isDayside', 'isFull', 'encountersMSPandMSH'])
        conditions_select = [condition+'_select' for condition in self.conditions]
        self.all_dataset = select_windows(self.dataset, conditions_select)
        indices_windows = np.ones(len(self.all_dataset))
        for condition in self.conditions:
            indices_windows = np.logical_and(indices_windows, self.all_dataset[condition].values)
        self.all_windows_indices = np.arange(len(self.all_dataset))[indices_windows]

        self.labelled_condition = kwargs.get('labelled_condition', ['isLabelled'])
        labelled_conditions_select = [condition+'_select' for condition in self.labelled_condition]
        self.dataset = select_windows(self.dataset, labelled_conditions_select + conditions_select)
        indices_windows = np.ones(len(self.dataset))
        for condition in self.conditions+self.labelled_condition:
            indices_windows = np.logical_and(indices_windows, self.dataset[condition].values)
        self.windows_indices = np.arange(len(self.dataset))[indices_windows]

        self.spectro_normalization_method = kwargs.get('spectro_normalization', 'per_channel')
        self.normalize(self.spectro_normalization_method)


    def __len__(self):
        #return len(self.dataset) // self.win_length
        return len(self.windows_indices)

    def normalize_per_channel(self, features, df, **kwargs):
        if len(features)>0:
            if 'scaler' not in kwargs:
                scaler = StandardScaler()
                scaler.fit(df.loc[:, features])
            else:
                scaler = kwargs['scaler']
            df.loc[:, features] = scaler.transform(df.loc[:, features])
        else:
            scaler = None
        return df, scaler

    def normalize_overall(self, features, df, **kwargs):
        if len(features) > 0:
            if 'mean' not in kwargs or 'std' not in kwargs:
                mean = df.loc[:, features].mean()
                std = df.loc[:, features].std()
            else:
                mean, std = kwargs['mean'], kwargs['std']
            df.loc[:, features] = (df.loc[:, features] - mean) / std
        else:
            mean, std = None, None
        return df, mean, std

    def normalize_overall_spectro(self):
        self.dataset, self.scaler = self.normalize_per_channel(self.moments_features, self.dataset)
        self.all_dataset, _ = self.normalize_per_channel(self.moments_features, self.all_dataset,
                                                         scaler=self.scaler)
        self.dataset, self.mean_spectro, self.std_spectro = self.normalize_overall(self.spectro_features,
                                                                                   self.dataset)
        self.all_dataset, _, _ = self.normalize_overall(self.spectro_features, self.all_dataset,
                                                        mean=self.mean_spectro, std=self.std_spectro)

    def normalize(self, spectro_normalization, **kwargs):
        if spectro_normalization == 'per_channel':
            self.dataset, self.scaler = self.normalize_per_channel(self.ml_features, self.dataset)
            self.all_dataset, _ = self.normalize_per_channel(self.ml_features, self.all_dataset,
                                                             scaler=self.scaler)

        elif spectro_normalization == 'overall':
            self.normalize_overall_spectro()

        elif spectro_normalization == 'overall_log':
            self.dataset[self.spectro_features] = self.dataset[self.spectro_features].where(
                self.dataset[self.spectro_features]>0.001, 0.001)
            # Replace 0 by 0.001
            self.dataset.loc[:,self.spectro_features] = np.log(self.dataset.loc[:,self.spectro_features].values)
            self.all_dataset.loc[:,self.spectro_features] = np.log(self.all_dataset.loc[:,self.spectro_features].values)
            self.normalize_overall_spectro()



class Windows(DataForWindows):

    def __getitem__(self, indice):
        i = self.windows_indices[indice]
        #subdf = self.dataset.iloc[i * self.win_length : (i+1) * self.win_length][self.moments_features + self.spectro_features + self.label]
        subdf = self.dataset.iloc[i - self.win_length + 1:i + 1][self.moments_features + self.spectro_features + self.label]

        spectro = torch.tensor(np.transpose(subdf[self.spectro_features].values).reshape((1, len(self.spectro_features), self.win_length))).double()
        moments = torch.tensor(np.transpose(subdf[self.moments_features].values).reshape((len(self.moments_features), 1, self.win_length))).double()
        labels = torch.tensor(np.transpose(subdf[self.label].values).reshape((len(self.label), 1, self.win_length))).double()

        return indice, (moments, spectro), labels

    def all_pred(self, model, threshold=0.5):
        nbrWindows = len(self.all_windows_indices)
        pred = pd.DataFrame(np.zeros((len(self.all_dataset), 1 + len(model.label_names))), index=self.all_dataset.index.values,
                            columns=[f"pred_sum_{label.split('_')[1]}" for label in model.label_names] + ["pred_count"])
        print(f"Number of windows = {nbrWindows}.")

        for count in range(nbrWindows):
            i = self.all_windows_indices[count]
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            moments = self.all_dataset.iloc[i - self.win_length + 1: i + 1][self.moments_features]
            moments = torch.tensor(
                np.transpose(moments.values).reshape((1, len(self.moments_features), 1, self.win_length))).double().to(
                device)
            spectro = self.all_dataset.iloc[i - self.win_length + 1: i + 1][self.spectro_features]
            spectro = torch.tensor(
                np.transpose(spectro.values).reshape((1, 1, len(self.spectro_features), self.win_length))).double().to(
                device)
            # prediction = torch.Tensor.cpu(model.forward((moments, spectro)).flatten()).detach().numpy()
            prediction = torch.Tensor.cpu(model.forward((moments, spectro))).detach().numpy()

            for nbr,label in enumerate(model.label_names):
                pred.iloc[i - self.win_length + 1: i + 1, -1 - len(model.label_names) + nbr] = (pred.iloc[i - self.win_length + 1: i + 1, -2].values +
                                                                 prediction[:,nbr,:,:])

            pred.iloc[i - self.win_length + 1: i + 1, -1] = pred.iloc[i - self.win_length + 1: i + 1, -1].values + 1

            if count % (nbrWindows // 10) == 0:
                print(f"{round(count / nbrWindows * 100, 2)}% of windows predicted.")

        for label in model.label_names:
            pred[f"pred_{label.split('_')[1]}"] = pred[f"pred_sum_{label.split('_')[1]}"].values / pred['pred_count'].values
            pred[f"predicted_class_{label.split('_')[1]}"] = pred[f"pred_{label.split('_')[1]}"].values > threshold
            pred.loc[pred[pred[f"pred_{label.split('_')[1]}"].isna().values].index.values, f"predicted_class_{label.split('_')[1]}"] = np.nan

        return pred

    '''
    def all_pred(self, model, threshold=0.5):
        nbrWindows = len(self.all_windows_indices)
        pred = pd.DataFrame(np.zeros((len(self.all_dataset),2)), index=self.all_dataset.index.values,
                            columns=['pred_sum', 'pred_count'])
        print(f"Number of windows = {nbrWindows}.")

        for count in range(nbrWindows):
            i = self.all_windows_indices[count]
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            moments = self.all_dataset.iloc[i - self.win_length + 1 : i+1][self.moments_features]
            moments = torch.tensor(
                np.transpose(moments.values).reshape((1, len(self.moments_features), 1, self.win_length))).double().to(
                device)
            spectro = self.all_dataset.iloc[i - self.win_length + 1 : i+1][self.spectro_features]
            spectro = torch.tensor(
                np.transpose(spectro.values).reshape((1, 1, len(self.spectro_features), self.win_length))).double().to(
                device)

            pred.iloc[i - self.win_length + 1 : i+1, -2] = (pred.iloc[i - self.win_length + 1 : i+1, -2].values +
                    torch.Tensor.cpu(model.forward((moments, spectro)).flatten()).detach().numpy())

            pred.iloc[i - self.win_length + 1 : i+1, -1] = pred.iloc[i - self.win_length + 1 : i+1, -1].values + 1

            if count % (nbrWindows // 10) == 0:
                print(f"{round(count / nbrWindows * 100, 2)}% of windows predicted.")

        pred['pred'] = pred['pred_sum'].values / pred['pred_count'].values
        pred['predicted_class'] = pred.pred.values > threshold
        pred.loc[pred[pred['pred'].isna().values].index.values, 'predicted_class'] = np.nan

        return pred
    '''

class WindowsSpectro2D(DataForWindows):

    def __getitem__(self, indice):
        i = self.windows_indices[indice]
        subdf = self.dataset.iloc[i - self.win_length + 1:i + 1][self.moments_features + self.spectro_features + self.label]

        #subdf = self.dataset.iloc[i * self.win_length : (i+1) * self.win_length][self.spectro_features + self.label]

        #labels = subdf[self.label].values
        #subdf.drop(self.label, axis=1, inplace=True)
        #self.inputs = torch.tensor(np.transpose(subdf.values).reshape((1, len(self.spectro_features),self.win_length))).double()
        #self.labels = torch.tensor(labels).double()

        inputs = torch.tensor(np.transpose(subdf[self.spectro_features].values).reshape((1, len(self.spectro_features),self.win_length))).double()
        labels = torch.tensor(np.transpose(subdf[self.label].values).reshape((len(self.label), 1, self.win_length))).double()

        return indice, inputs, labels

    def all_pred(self, model, threshold=0.5):
        nbrWindows = len(self.all_windows_indices)
        pred = pd.DataFrame(np.zeros((len(self.all_dataset), 2)), index=self.all_dataset.index.values,
                            columns=['pred_sum', 'pred_count'])
        print(f"Number of windows = {nbrWindows}.")

        for count in range(nbrWindows):
            i = self.all_windows_indices[count]
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            spectro = self.all_dataset.iloc[i - self.win_length + 1: i + 1][self.spectro_features]
            spectro = torch.tensor(
                np.transpose(spectro.values).reshape((1, 1, len(self.spectro_features), self.win_length))).double().to(
                device)

            pred.iloc[i - self.win_length + 1: i + 1, -2] = (pred.iloc[i - self.win_length + 1: i + 1, -2].values +
                                                             torch.Tensor.cpu(model.forward(spectro).flatten()).detach().numpy())

            pred.iloc[i - self.win_length + 1: i + 1, -1] = pred.iloc[i - self.win_length + 1: i + 1, -1].values + 1

            if count % (nbrWindows // 10) == 0:
                print(f"{round(count / nbrWindows * 100, 2)}% of windows predicted.")

        pred['pred'] = pred['pred_sum'].values / pred['pred_count'].values
        pred['predicted_class'] = pred.pred.values > threshold
        pred.loc[pred[pred['pred'].isna().values].index.values, 'predicted_class'] = np.nan

        return pred


class WindowsMoments(DataForWindows):

    def __getitem__(self, indice):
        i = self.windows_indices[indice]
        subdf = self.dataset.iloc[i - self.win_length + 1:i + 1][self.moments_features + self.spectro_features + self.label]

        #subdf = self.dataset.iloc[i * self.win_length : (i+1) * self.win_length][self.moments_features + self.label]

        inputs = torch.tensor(np.transpose(subdf[self.moments_features].values).reshape((len(self.moments_features),1,self.win_length))).double()
        labels = torch.tensor(np.transpose(subdf[self.label].values).reshape((len(self.label), 1, self.win_length))).double()

        return indice, inputs, labels

    def all_pred(self, model, threshold=0.5):
        nbrWindows = len(self.all_windows_indices)
        pred = pd.DataFrame(np.zeros((len(self.all_dataset), 2)), index=self.all_dataset.index.values,
                            columns=['pred_sum', 'pred_count'])
        print(f"Number of windows = {nbrWindows}.")

        for count in range(nbrWindows):
            i = self.all_windows_indices[count]
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            moments = self.all_dataset.iloc[i - self.win_length + 1: i + 1, -1][self.moments_features]
            moments = torch.tensor(
                np.transpose(moments.values).reshape((1, len(self.moments_features), 1, self.win_length))).double().to(
                device)

            pred.iloc[i - self.win_length + 1: i + 1, -2] = (pred.iloc[i - self.win_length + 1: i + 1, -2].values +
                                                             torch.Tensor.cpu(
                                                                 model.forward(moments).flatten()).detach().numpy())

            pred.iloc[i - self.win_length + 1: i + 1, -1] = pred.iloc[i - self.win_length + 1: i + 1, -1].values + 1

            if count % (nbrWindows // 10) == 0:
                print(f"{round(count / nbrWindows * 100, 2)}% of windows predicted.")

        pred['pred'] = pred['pred_sum'].values / pred['pred_count'].values
        pred['predicted_class'] = pred.pred.values > threshold
        pred.loc[pred[pred['pred'].isna().values].index.values, 'predicted_class'] = np.nan

        return pred


class AutoEncoderWindows(DataForWindows):

    def __getitem__(self, indice):
        i = self.windows_indices[indice]
        subdf = self.dataset.iloc[i - self.win_length + 1:i + 1][self.moments_features + self.spectro_features + self.label]

        # subdf = self.dataset.iloc[i * self.win_length : (i+1) * self.win_length][self.spectro_features + self.label]
        inputs = torch.tensor(np.transpose(subdf[self.spectro_features].values).reshape((1, len(self.spectro_features),self.win_length))).double()

        return indice, inputs, inputs


class WindowsEntangledMomentsSpectro(DataForWindows):
    def __getitem__(self, indice):
        i = self.windows_indices[indice]
        subdf = self.dataset.iloc[i - self.win_length + 1:i + 1][self.moments_features + self.spectro_features + self.label]

        # subdf = self.dataset.iloc[i * self.win_length : (i+1) * self.win_length][self.moments_features + self.spectro_features + self.label]

        spectro = torch.tensor(np.transpose(subdf[self.spectro_features].values).reshape((1, len(self.spectro_features), self.win_length))).double()
        _, nb_channels, n_time = spectro.shape
        moments = torch.tensor(np.transpose(subdf[self.moments_features].values).reshape((len(self.moments_features), 1, self.win_length))).double()
        moments = moments.repeat(1,nb_channels,1)
        inputs = torch.concat((spectro,moments), dim = 0)
        labels = torch.tensor(np.transpose(subdf[self.label].values).reshape((len(self.label), 1, self.win_length))).double()

        return indice, inputs, labels
