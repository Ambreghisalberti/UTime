import torch
from torch.utils.data import Dataset
from swapp.windowing.make_windows import prepare_df
from swapp.windowing.make_windows.utils import select_windows, durationToNbrPts, time_resolution, nbr_windows
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class Windows(Dataset):

    def __init__(self, all_data, position, omni_data, win_duration, ml_features, **kwargs):
        self.win_length = durationToNbrPts(win_duration, time_resolution(all_data))
        self.ml_features = ml_features

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
        conditions = ['isDayside', 'isFull', 'encountersMSPandMSH']
        self.all_dataset = select_windows(self.dataset, conditions)
        self.dataset = select_windows(self.dataset, ['isLabelled']+conditions)

        scaler = StandardScaler()
        self.dataset.loc[:,ml_features] = scaler.fit_transform(self.dataset.loc[:,ml_features])
        self.all_dataset.loc[:,ml_features] = scaler.transform(self.all_dataset.loc[:,ml_features])
        self.scaler = scaler

    def __getitem__(self, i):
        subdf = self.dataset.iloc[i * self.win_length : (i+1) * self.win_length][self.ml_features + ['label']]
        labels = subdf['label'].values
        subdf.drop(['label'], axis=1, inplace=True)
        self.inputs = torch.tensor(np.transpose(subdf.values).reshape((len(self.ml_features),1,self.win_length))).double()
        self.labels = torch.tensor(labels).double()
        self.times = subdf.index.values
        return i, self.inputs, self.labels

    def __len__(self):
        return len(self.dataset) // self.win_length

    def all_pred(self, model, threshold=0.5):
        nbrWindows = nbr_windows(self.all_dataset, self.win_length)
        pred = pd.DataFrame(-1.0 * np.arange(len(self.all_dataset)), index=self.all_dataset.index.values,
                            columns=['pred'])
        print(f"Number of windows = {nbrWindows}.")

        for i in range(nbrWindows):
            inputs = self.all_dataset.iloc[i * self.win_length: (i + 1) * self.win_length][self.ml_features]
            inputs = torch.tensor(
                np.transpose(inputs.values).reshape((1, len(self.ml_features), 1, self.win_length))).double()

            pred.iloc[i * self.win_length: (i + 1) * self.win_length, -1] = model.forward(
                inputs).flatten().detach().numpy()
            if i % (nbrWindows // 10) == 0:
                print(f"{round(i / nbrWindows * 100, 2)}% of windows predicted.")

        pred['predicted_class'] = pred.pred.values > threshold

        return pred
