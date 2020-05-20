import os
import io
import json
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class Pubmed(Dataset):
    def __init__(self, data_dir, split, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.data_file = 'corpus.json'
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 2000)
        # note - 2070 is the 95th percentile of the abstract length
        # minimum times for word occurence
        self.min_occ = kwargs.get('min_occ', 3)
        self._load_data(fields=['abstract_raw'])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> T_co:
        return self.data[index]

    def _load_data(self, fields):
        with open(os.path.join(self.data_dir, self.data_file),'r') as file:
            self.data = json.load(file)
        # filtering
        new_data = []
        for d in self.data:
            d_new = {}
            for field in fields:
                d_new[field] = d[field]
            new_data.append(d_new)

        self.data = new_data



