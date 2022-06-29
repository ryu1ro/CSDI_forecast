import pickle

import os
from time import time
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def parse_sample(timepoint, length, df):
    sample = df.iloc[timepoint:timepoint + length].to_numpy()
    observed_values = sample[:,:-2]
    observed_masks = ~np.isnan(observed_values)
    time_covariates = sample[:,-2:]

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    time_covariates = time_covariates.astype("float32")

    return observed_values, observed_masks, time_covariates


class Hourly_Dataset(Dataset):
    def __init__(self, data_name='solar', time_length=168+24, seed=0,  use_index_list=None,):
        self.data_name = data_name
        self.eval_length = time_length
        # np.random.seed(seed)  # seed for ground truth choice

        self.observed_values = []
        self.observed_masks = []
        self.time_covariates = []
        path = (
            "./dataset/" +self.data_name+ "_seed" + str(seed) + ".pk"
        )

        if os.path.isfile(path) == False:  # if datasetfile is none, create
            df = pd.read_csv('./dataset/csv/' + self.data_name + '.csv')
            df = df.rename(columns={df.columns[0]: 'date'})
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            df['hour'] = df['date'].dt.hour

            df = df.drop(columns='date')
            # df = df.drop(0)
            total_time_steps = len(df)
            for time_point in range(total_time_steps - self.eval_length + 1):
                try:
                    observed_values, observed_masks, time_covariates = parse_sample(
                        time_point, self.eval_length, df
                    )
                    self.observed_values.append(observed_values)
                    self.observed_masks.append(observed_masks)
                    self.time_covariates.append(time_covariates)

                except Exception as e:
                    print(time_point, e)
                    continue
            self.observed_values = np.array(self.observed_values)
            self.observed_masks = np.array(self.observed_masks)
            self.time_covariates = np.array(self.time_covariates)

            with open(path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks, self.time_covariates], f
                )
        else:  # load datasetfile
            with open(path, "rb") as f:
                self.observed_values, self.observed_masks, self.time_covariates = pickle.load(
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
            "time_covariates": self.time_covariates[index],
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, data_name='solar', time_length=168+24, batch_size=8):

    # only to obtain total length of dataset
    dataset = Hourly_Dataset(data_name=data_name, time_length=time_length, seed=seed)
    indlist = np.arange(len(dataset))

    test_index = indlist[-24*7:]
    remain_index = indlist[:-24*7]

    np.random.seed(seed)
    np.random.shuffle(remain_index)

    valid_index = remain_index[-24*5:]
    train_index = remain_index[:-24*5]

    dataset = Hourly_Dataset(
        data_name=data_name, time_length=time_length, use_index_list=train_index,  seed=seed)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=1,
        num_workers=os.cpu_count()
        )

    valid_dataset = Hourly_Dataset(
        data_name=data_name, time_length=time_length, use_index_list=valid_index,  seed=seed)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=0,
        num_workers=os.cpu_count()
        )

    test_dataset = Hourly_Dataset(
        data_name=data_name, time_length=time_length, use_index_list=test_index,  seed=seed)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=0,
        num_workers=os.cpu_count()
        )

    return train_loader, valid_loader, test_loader