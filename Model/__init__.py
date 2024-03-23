import torch
from torch.utils.data import Dataset
import sys
sys.path.append('/home/ghisalberti/Documents/PycharmProjects/windows/')
from windowing import make_windows

class Windows(Dataset):

    def __init__(self, all_data, position, omni_data, win_length, label_paths):
        self.df = make_windows.prepare_df(all_data, position, omni_data, win_length, label_paths)
        self.win_length = win_length

    def __getitem__(self, i):
        subdf = self.df.iloc[i:i + self.win_length]
        labels = subdf['label'].values
        subdf.drop(['label'], axis=1, inplace=True)
        inputs = torch.tensor(subdf.values).double()
        labels = torch.tensor(labels).double()
        return subdf.index.values, inputs, labels

    def __len__(self):
        return len(self.df) // self.win_length
