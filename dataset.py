import pickle

import os
from time import time
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def parse_sample(timepoint, length, df, n_covariates=2):
    sample = df.iloc[timepoint:timepoint + length].to_numpy()
    observed_values = sample[:,:-n_covariates] #exclude time covariates
    observed_masks = ~np.isnan(observed_values)
    time_covariates = sample[:,-n_covariates:]

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    time_covariates = time_covariates.astype("float32")

    return observed_values, observed_masks, time_covariates


class Hourly_Dataset(Dataset):
    def __init__(self, data_name='solar', time_length=168+24, seed=0,  use_index_list=None,):
        self.data_name = data_name
        self.eval_length = time_length

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
                    [self.observed_values, self.observed_masks, self.time_covariates], f, protocol=4
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


def hourly_dataloader(
    seed=1,
    data_name='solar',
    batch_size=8,
    test_batch_size=1,
    time_length=168+24,
    ):

    # only to obtain total length of dataset
    dataset = Hourly_Dataset(data_name=data_name, time_length=time_length, seed=seed)
    indlist = np.arange(len(dataset))

    test_index = indlist[-24*7::24]
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
        num_workers=os.cpu_count(),
        pin_memory=True
        )

    valid_dataset = Hourly_Dataset(
        data_name=data_name, time_length=time_length, use_index_list=valid_index,  seed=seed)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=0,
        num_workers=os.cpu_count(),
        pin_memory=True
        )

    test_dataset = Hourly_Dataset(
        data_name=data_name, time_length=time_length, use_index_list=test_index,  seed=seed)
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=0,
        num_workers=os.cpu_count(),
        pin_memory=True
        )

    return train_loader, valid_loader, test_loader

class Wiki_Dataset(Dataset):
    def __init__(
        self,
        seed=0,
        use_index_list=None,
        ):
        self.data_name = 'wiki'
        self.eval_length = 90+30

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
            df['day'] = df['date'].dt.day - 1
            df['month'] = df['date'].dt.month - 1

            df = df.drop(columns='date')
            # df = df.drop(0)
            total_time_steps = len(df)
            for time_point in range(total_time_steps - self.eval_length + 1):
                try:
                    observed_values, observed_masks, time_covariates = parse_sample(
                        time_point, self.eval_length, df, n_covariates=3
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
                    [self.observed_values, self.observed_masks, self.time_covariates], f, protocol=4
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


def wiki_dataloader(
    seed=1,
    batch_size=8,
    test_batch_size=1,
    ):

    # only to obtain total length of dataset
    dataset = Wiki_Dataset(seed=seed)
    indlist = np.arange(len(dataset))

    test_index = indlist[-30*5::30]
    remain_index = indlist[:-30*5]

    np.random.seed(seed)
    np.random.shuffle(remain_index)

    valid_index = remain_index[-30*5:]
    train_index = remain_index[:-30*5]

    dataset = Wiki_Dataset(
        use_index_list=train_index,  seed=seed)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=1,
        num_workers=os.cpu_count(),
        pin_memory=True
        )

    valid_dataset = Wiki_Dataset(
        use_index_list=valid_index,  seed=seed)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=0,
        num_workers=os.cpu_count(),
        pin_memory=True
        )

    test_dataset = Wiki_Dataset(
        use_index_list=test_index,  seed=seed)
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=0,
        num_workers=os.cpu_count(),
        pin_memory=True
        )

    return train_loader, valid_loader, test_loader

class Taxi_Dataset(Dataset):
    def __init__(
        self,
        seed=0,
        use_index_list=None,
        is_train=True
        ):
        self.data_name = 'taxi'
        self.eval_length = 48 + 24
        self.is_train = is_train
        data_suffix = '_train' if is_train else '_test'

        self.observed_values = []
        self.observed_masks = []
        self.time_covariates = []

        path = (
            "./dataset/" +self.data_name + data_suffix + "_seed" + str(seed) + ".pk"
        )

        if os.path.isfile(path) == False:  # if datasetfile is none, create
            df = pd.read_csv('./dataset/csv/' + self.data_name + data_suffix + '.csv')
            df = df.rename(columns={df.columns[0]: 'date'})
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            df['hour'] = df['date'].dt.hour

            df = df.drop(columns='date')
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
                    [self.observed_values, self.observed_masks, self.time_covariates], f, protocol=4
                )
        else:  # load datasetfile
            with open(path, "rb") as f:
                self.observed_values, self.observed_masks, self.time_covariates = pickle.load(
                    f
                )
        if self.is_train==False:
            self.use_index_list = np.arange(len(self.observed_values))[::24]
        elif use_index_list is None:
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

def taxi_dataloader(
    seed=1,
    batch_size=8,
    test_batch_size=1,
    ):

    # only to obtain total length of dataset
    dataset_train = Taxi_Dataset(seed=seed, is_train=True)
    indlist = np.arange(len(dataset_train))

    np.random.seed(seed)
    np.random.shuffle(indlist)

    valid_index = indlist[-24*5:]
    train_index = indlist[:-24*5]

    dataset = Taxi_Dataset(
        use_index_list=train_index,  seed=seed)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=1,
        num_workers=os.cpu_count(),
        pin_memory=True
        )

    valid_dataset = Taxi_Dataset(
        use_index_list=valid_index,  seed=seed)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=0,
        num_workers=os.cpu_count(),
        pin_memory=True
        )

    test_dataset = Taxi_Dataset(
        seed=seed, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=0,
        num_workers=os.cpu_count(),
        pin_memory=True
        )

    return train_loader, valid_loader, test_loader

def get_dataloader(
    seed=1,
    data_name='solar',
    batch_size=8,
    test_batch_size=1,
):
    if data_name == 'taxi':
        train_loader, valid_loader, test_loader = taxi_dataloader(
            seed=seed,
            batch_size=batch_size,
            test_batch_size=test_batch_size
        )
    elif data_name == 'wiki':
        train_loader, valid_loader, test_loader = wiki_dataloader(
            seed=seed,
            batch_size=batch_size,
            test_batch_size=test_batch_size
        )
    else:
        train_loader, valid_loader, test_loader = hourly_dataloader(
            seed=seed,
            data_name=data_name,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
        )

    return train_loader, valid_loader, test_loader