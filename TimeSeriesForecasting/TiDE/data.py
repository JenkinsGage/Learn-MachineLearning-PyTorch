import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm

class StockDataset(Dataset):
    def __init__(self, data_root, meta_file='symbols_valid_meta.csv', subset=None):
        self.data_root = data_root
        self.subset = subset
        self.meta_data = pd.read_csv(os.path.join(self.data_root, meta_file))
        self.meta_data.fillna('NaN', inplace=True)
        invalid_rows = 0
        for index, row in self.meta_data.iterrows():
            is_etf = row['ETF'] == 'Y'
            path = os.path.join(self.data_root, 'etfs/' if is_etf else 'stocks/', f'{row["Symbol"]}.csv')
            if not os.path.exists(path):
                self.meta_data.drop(index, inplace=True)
                invalid_rows += 1
        self.meta_data.reset_index(drop=True, inplace=True)
        print(f'Meta Data Loaded {len(self.meta_data)} | {invalid_rows} invalid rows removed.')

        self.listing_exchange_unique = self.meta_data['Listing Exchange'].unique()
        self.market_category_unique = self.meta_data['Market Category'].unique()
        self.round_lot_size_unique = self.meta_data['Round Lot Size'].unique()
        self.financial_status_unique = self.meta_data['Financial Status'].unique()
        self.next_shares_unique = self.meta_data['NextShares'].unique()

        self.listing_exchange_onehot = torch.eye(len(self.listing_exchange_unique))
        self.market_category_onehot = torch.eye(len(self.market_category_unique))
        self.round_lot_size_onehot = torch.eye(len(self.round_lot_size_unique))
        self.financial_status_onehot = torch.eye(len(self.financial_status_unique))
        self.next_shares_onehot = torch.eye(len(self.next_shares_unique))

    def __len__(self):
        if self.subset is not None:
            return self.subset
        else:
            return len(self.meta_data)

    def __getitem__(self, idx):
        symbol = self.meta_data.loc[idx, 'Symbol']

        if symbol == 'PRN':
            symbol = '_PRN' # Name PRN.csv is not allowed on windows so rename it to the actually file _PRN.csv

        is_etf = self.meta_data.loc[idx, 'ETF'] == 'Y'
        # load timeseries data for the current symbol
        timeseries_file = os.path.join(self.data_root, 'etfs/' if is_etf else 'stocks/', f'{symbol}.csv')
        timeseries_data = pd.read_csv(timeseries_file)

        dates = np.array(timeseries_data['Date'])
        open_prices = np.array(timeseries_data['Open'])
        high_prices = np.array(timeseries_data['High'])
        low_prices = np.array(timeseries_data['Low'])
        close_prices = np.array(timeseries_data['Close'])
        volumes = np.array(timeseries_data['Volume'])

        static_attributes = {
            'is_etf': torch.ones(1, ) if is_etf else torch.zeros(1, ),
            'listing_exchange_onehot': self.listing_exchange_onehot[self.listing_exchange_unique == self.meta_data.loc[idx, 'Listing Exchange']][0],
            'market_category_onehot': self.market_category_onehot[self.market_category_unique == self.meta_data.loc[idx, 'Market Category']][0],
            'round_lot_size_onehot': self.round_lot_size_onehot[self.round_lot_size_unique == self.meta_data.loc[idx, 'Round Lot Size']][0],
            'financial_status_onehot': self.financial_status_onehot[self.financial_status_unique == self.meta_data.loc[idx, 'Financial Status']][0],
            'next_shares_onehot': self.next_shares_onehot[self.next_shares_unique == self.meta_data.loc[idx, 'NextShares']][0],
        }

        return symbol, dates, open_prices, high_prices, low_prices, close_prices, volumes, static_attributes

def convert_date_to_dynamic_covariates(date:str):
    datetime_obj = datetime.datetime.strptime(date, "%Y-%m-%d")
    weekday_onehot = F.one_hot(torch.tensor(datetime_obj.weekday()), 7)
    month_onehot = F.one_hot(torch.tensor(datetime_obj.month-1), 12)
    return torch.cat([weekday_onehot, month_onehot], dim=-1)


class TimeSeriesDataset(Dataset):
    def __init__(self, stock_dataset:StockDataset, lookback_steps=360, horizon_steps=60, gap=1):
        self.stock_dataset = stock_dataset
        self.lookback_steps = lookback_steps
        self.horizon_steps = horizon_steps
        self.gap = gap
        assert self.gap>=1

        self.time_series = []
        for symbol, dates, open_prices, high_prices, low_prices, close_prices, volumes, static_attributes in tqdm(self.stock_dataset, desc='Index Time Series'):
            if np.any(np.isnan(close_prices)):
                continue
            data_len = len(dates)
            for i in range(max(0, data_len - self.lookback_steps - self.horizon_steps + 1)):
                if i % self.gap != 0:
                    continue

                start_idx = i + self.lookback_steps # Start Index of The Horizon
                end_idx = start_idx + self.horizon_steps # End Index of The Horizon

                lookback = close_prices[start_idx-self.lookback_steps:start_idx]
                horizon = close_prices[start_idx:end_idx]
                dates_data = dates[start_idx-self.lookback_steps:end_idx]

                self.time_series.append((lookback, horizon, static_attributes, dates_data))

    def __len__(self):
        return len(self.time_series)

    def __getitem__(self, idx):
        return self.time_series[idx]


def collate_fn(batch):
    lookback_batch, horizon_batch, static_attributes_batch, dates_batch = zip(*batch)
    lookback_batch = torch.from_numpy(np.stack(lookback_batch)).to(torch.float32)
    horizon_batch = torch.from_numpy(np.stack(horizon_batch)).to(torch.float32)
    static_attributes_batch = torch.stack([torch.cat([static_attribtue[key] for key in static_attribtue]) for static_attribtue in static_attributes_batch]).to(torch.float32)
    dynamic_covariates_batch = torch.stack([torch.stack([convert_date_to_dynamic_covariates(date) for date in covariate]) for covariate in dates_batch]).to(torch.float32)
    return lookback_batch, horizon_batch, static_attributes_batch, dynamic_covariates_batch