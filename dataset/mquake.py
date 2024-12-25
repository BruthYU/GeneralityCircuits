import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset

class mqauke_Dataset(Dataset):
    def __init__(
            self,
            file_path,
            size = None):

        with open(file_path, "r") as f:
            self.data = json.load(f)
        if size is not None:
            self.data = self.data[:size]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]