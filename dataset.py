import pickle

import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def parse_id(id_, df):
    observed_values = df[id_*24:id_*24+192].to_numpy()
    observed_values = np.array(observed_values)
    observed_masks = ~np.isnan(observed_values)

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")

    return observed_values, observed_masks


class Solar_Dataset(Dataset):
    def __init__(self, eval_length=192, use_index_list=None, seed=0):
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice

        self.observed_values = []
        self.observed_masks = []
        path = (
            "./dataset/solar_nips"  + "_seed" + str(seed) + ".pk"
        )

        
        df = pd.read_csv('./dataset/solar_nips/csv/solar_nips.csv')
        df = df.rename(columns={df.columns[0]: 'date'})
        df = df.drop(columns='date')
        df = df.drop(0)
        l_total = len(df)
        l1 = 168
        l2 = 24
        idx_last = int((l_total - (l1 + l2))/24)

        if os.path.isfile(path) == False:  # if datasetfile is none, create
            for id_ in range(idx_last+1):
                try:
                    observed_values, observed_masks = parse_id(
                        id_, df
                    )
                    self.observed_values.append(observed_values)
                    self.observed_masks.append(observed_masks)
                except Exception as e:
                    print(id_, e)
                    continue
            self.observed_values = np.array(self.observed_values)
            self.observed_masks = np.array(self.observed_masks)

            with open(path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks], f
                )
        else:  # load datasetfile
            with open(path, "rb") as f:
                self.observed_values, self.observed_masks = pickle.load(
                    f
                )
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=None, batch_size=8):

    # only to obtain total length of dataset
    dataset = Solar_Dataset(seed=seed)
    indlist = np.arange(len(dataset))

    np.random.seed(seed)
    np.random.shuffle(indlist)

    test_index = indlist[-7:]
    valid_index = indlist[-12:-7]
    train_index = indlist[:-12]

    dataset = Solar_Dataset(
        use_index_list=train_index,  seed=seed)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Solar_Dataset(
        use_index_list=valid_index,  seed=seed)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Solar_Dataset(
        use_index_list=test_index,  seed=seed)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader, valid_loader, test_loader