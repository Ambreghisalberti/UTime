import torch
from torch.utils.data import Dataset
from swapp.windowing import make_windows


class Windows(Dataset):

    def __init__(self, all_data, position, omni_data, win_length, label_paths, labelled_days):
        self.df, self.pos, self.omni = make_windows.prepare_df(all_data, position, omni_data, win_length, label_paths, labelled_days)
        self.win_length = win_length

    def __getitem__(self, i):
        subdf = self.df.iloc[i:i + self.win_length]
        labels = subdf['label'].values
        subdf.drop(['label'], axis=1, inplace=True)
        self.inputs = torch.tensor(subdf.values).double()
        self.labels = torch.tensor(labels).double()
        self.times = subdf.index.values
        return self.times, self.inputs, self.labels

    def __len__(self):
        return len(self.df) // self.win_length
