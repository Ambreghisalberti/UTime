import torch
from torch.utils.data import Dataset
from swapp.windowing.make_windows import prepare_df
from swapp.windowing.make_windows.utils import select_windows, durationToNbrPts, time_resolution
import pandas as pd

class Windows(Dataset):

    def __init__(self, all_data, position, omni_data, win_duration, label_paths, labelled_days, ml_features):
        self.df, self.pos, self.omni = prepare_df(all_data, position, omni_data, win_duration, label_paths,
                                                               labelled_days)
        self.omni = self.omni.rename(columns={col:"OMNI_"+col for col in self.omni.columns})
        self.dataset = pd.concat([self.df, self.omni], axis = 1)
        self.dataset = select_windows(self.dataset, ['isFull', 'isLabelled', 'encountersMSPandMSH'])
        self.win_length = durationToNbrPts(win_duration, time_resolution(all_data))
        self.ml_features = ml_features

    def __getitem__(self, i):
        subdf = self.df.iloc[i * self.win_length : (i+1) * self.win_length][self.ml_features + ['label']]
        labels = subdf['label'].values
        subdf.drop(['label'], axis=1, inplace=True)
        self.inputs = torch.tensor(subdf.values).double()
        self.labels = torch.tensor(labels).double()
        self.times = subdf.index.values
        return self.times, self.inputs, self.labels

    def __len__(self):
        return len(self.df) // self.win_length
