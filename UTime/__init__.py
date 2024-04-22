import torch
from torch.utils.data import Dataset
from swapp.windowing.make_windows import prepare_df
from swapp.windowing.make_windows.utils import select_windows, durationToNbrPts, time_resolution, nbr_windows
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataForWindows(Dataset):

    def __init__(self, all_data, position, omni_data, win_duration, moments_features = [], spectro_features = [],
                 label = ['label'], **kwargs):
        self.win_length = durationToNbrPts(win_duration, time_resolution(all_data))
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
        self.all_dataset = select_windows(self.dataset, self.conditions)
        self.labelled_condition = kwargs.get('labelled_condition', ['isLabelled'])
        self.dataset = select_windows(self.dataset, self.labelled_condition + self.conditions)

        scaler = StandardScaler()
        self.dataset.loc[:,self.ml_features] = scaler.fit_transform(self.dataset.loc[:,self.ml_features])
        self.all_dataset.loc[:,self.ml_features] = scaler.transform(self.all_dataset.loc[:,self.ml_features])
        self.scaler = scaler

    def __len__(self):
        return len(self.dataset) // self.win_length



class Windows(DataForWindows):

    def __getitem__(self, i):
        subdf = self.dataset.iloc[i * self.win_length : (i+1) * self.win_length][self.moments_features + self.spectro_features + self.label]
        labels = subdf[self.label].values
        subdf.drop(self.label, axis=1, inplace=True)
        self.spectro = torch.tensor(np.transpose(subdf[self.spectro_features].values).reshape((1, len(self.spectro_features), self.win_length))).double()
        self.moments = torch.tensor(np.transpose(subdf[self.moments_features].values).reshape((len(self.moments_features), 1, self.win_length))).double()
        self.labels = torch.tensor(labels).double()
        self.times = subdf.index.values
        return i, (self.moments, self.spectro), self.labels

    def all_pred(self, model, threshold=0.5):
        nbrWindows = nbr_windows(self.all_dataset, self.win_length)
        pred = pd.DataFrame(-1.0 * np.arange(len(self.all_dataset)), index=self.all_dataset.index.values,
                            columns=['pred'])
        print(f"Number of windows = {nbrWindows}.")

        for i in range(nbrWindows):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            moments = self.all_dataset.iloc[i * self.win_length: (i + 1) * self.win_length][self.moments_features]
            moments = torch.tensor(
                np.transpose(moments.values).reshape((1, len(self.moments_features), 1, self.win_length))).double().to(
                device)
            spectro = self.all_dataset.iloc[i * self.win_length: (i + 1) * self.win_length][self.spectro_features]
            spectro = torch.tensor(
                np.transpose(spectro.values).reshape((1, 1, len(self.spectro_features), self.win_length))).double().to(
                device)

            pred.iloc[i * self.win_length: (i + 1) * self.win_length, -1] = torch.Tensor.cpu(model.forward(
                (moments, spectro)).flatten()).detach().numpy()
            if i % (nbrWindows // 10) == 0:
                print(f"{round(i / nbrWindows * 100, 2)}% of windows predicted.")

        pred['predicted_class'] = pred.pred.values > threshold

        return pred



class WindowsSpectro2D(DataForWindows):

    def __getitem__(self, i):
        subdf = self.dataset.iloc[i * self.win_length : (i+1) * self.win_length][self.spectro_features + self.label]
        labels = subdf[self.label].values
        subdf.drop(self.label, axis=1, inplace=True)
        self.inputs = torch.tensor(np.transpose(subdf.values).reshape((1, len(self.spectro_features),self.win_length))).double()
        self.labels = torch.tensor(labels).double()
        self.times = subdf.index.values
        return i, self.inputs, self.labels

    def all_pred(self, model, threshold=0.5):
        nbrWindows = nbr_windows(self.all_dataset, self.win_length)
        pred = pd.DataFrame(-1.0 * np.arange(len(self.all_dataset)), index=self.all_dataset.index.values,
                            columns=['pred'])
        print(f"Number of windows = {nbrWindows}.")

        for i in range(nbrWindows):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            spectro = self.all_dataset.iloc[i * self.win_length: (i + 1) * self.win_length][self.spectro_features]
            spectro = torch.tensor(
                np.transpose(spectro.values).reshape((1, 1, len(self.spectro_features), self.win_length))).double().to(
                device)

            pred.iloc[i * self.win_length: (i + 1) * self.win_length, -1] = torch.Tensor.cpu(model.forward(
                spectro).flatten()).detach().numpy()
            if i % (nbrWindows // 10) == 0:
                print(f"{round(i / nbrWindows * 100, 2)}% of windows predicted.")

        pred['predicted_class'] = pred.pred.values > threshold

        return pred


class WindowsMoments(DataForWindows):

    def __getitem__(self, i):
        subdf = self.dataset.iloc[i * self.win_length : (i+1) * self.win_length][self.moments_features + self.label]
        labels = subdf[self.label].values
        subdf.drop(self.label, axis=1, inplace=True)
        self.inputs = torch.tensor(np.transpose(subdf.values).reshape((len(self.moments_features),1,self.win_length))).double()
        self.labels = torch.tensor(labels).double()
        self.times = subdf.index.values
        return i, self.inputs, self.labels

    def all_pred(self, model, threshold=0.5):
        nbrWindows = nbr_windows(self.all_dataset, self.win_length)
        pred = pd.DataFrame(-1.0 * np.arange(len(self.all_dataset)), index=self.all_dataset.index.values,
                            columns=['pred'])
        print(f"Number of windows = {nbrWindows}.")

        for i in range(nbrWindows):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            moments = self.all_dataset.iloc[i * self.win_length: (i + 1) * self.win_length][self.moments_features]
            moments = torch.tensor(
                np.transpose(moments.values).reshape((1, len(self.moments_features), 1, self.win_length))).double().to(
                device)

            pred.iloc[i * self.win_length: (i + 1) * self.win_length, -1] = torch.Tensor.cpu(model.forward(
                moments).flatten()).detach().numpy()
            if i % (nbrWindows // 10) == 0:
                print(f"{round(i / nbrWindows * 100, 2)}% of windows predicted.")

        pred['predicted_class'] = pred.pred.values > threshold

        return pred