import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import pandas as pd
from tqdm import tqdm

class StockDataset(Dataset):
    def __init__(self, data_root, meta_file='symbols_valid_meta.csv', lookback_steps=270, horizon_steps=90, roll_steps=1) -> None:
        super().__init__()
        self.data_root = data_root
        self.lookback_steps = lookback_steps
        self.horizon_steps = horizon_steps
        self.roll_steps = roll_steps
        self.meta_data = pd.read_csv(os.path.join(self.data_root, meta_file))
        self.meta_data.fillna('NaN', inplace=True)

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

        invalid_rows = 0
        self.data_frames = []
        self.ts_indexes = []
        for index, row in tqdm(self.meta_data.iterrows(), total=len(self.meta_data), desc='Indexing Time Series'):
            etf = row['ETF'] == 'Y'
            path = os.path.join(self.data_root, 'etfs/' if etf else 'stocks/', f'{row["Symbol"]}.csv')
            if not os.path.exists(path):
                self.meta_data.drop(index, inplace=True)
                invalid_rows += 1
            else:
                df = pd.read_csv(path)
                df.dropna(inplace=True)
                df.reset_index(drop=True, inplace=True)
                df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype('float32')
                df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
                static_attributes = self.get_meta_static_attributes(row)
                self.data_frames.append((df, static_attributes))
                df_index = len(self.data_frames) - 1
    
                ts_len = len(df)
                if ts_len < self.lookback_steps + self.horizon_steps + -1:
                    continue
                for i in range(ts_len - self.lookback_steps - self.horizon_steps + 1):
                    if i % self.roll_steps != 0:
                        continue
                    self.ts_indexes.append((df_index, i))
                
        self.meta_data.reset_index(drop=True, inplace=True)
        print(f'Index Complete. Invalid Meta Rows: {invalid_rows}')

    def __len__(self):
        return len(self.ts_indexes)
    
    def __getitem__(self, idx):
        df_index, ts_index = self.ts_indexes[idx]
        df, static_attributes = self.data_frames[df_index]
        lookback = df.iloc[ts_index:ts_index+self.lookback_steps]
        horizon = df.iloc[ts_index+self.lookback_steps:ts_index+self.lookback_steps+self.horizon_steps]
        return lookback, horizon, static_attributes
    
    def get_meta_static_attributes(self, row):
        etf = torch.ones(1, ) if row['ETF']=='Y' else torch.zeros(1, )
        listing_exchange = self.listing_exchange_onehot[self.listing_exchange_unique == row['Listing Exchange']][0]
        market_category = self.market_category_onehot[self.market_category_unique == row['Market Category']][0]
        round_lot_size = self.round_lot_size_onehot[self.round_lot_size_unique == row['Round Lot Size']][0]
        financial_status = self.financial_status_onehot[self.financial_status_unique == row['Financial Status']][0]
        next_shares = self.next_shares_onehot[self.next_shares_unique == row['NextShares']][0]
        return torch.cat([etf, listing_exchange, market_category, round_lot_size, financial_status, next_shares])
    
def collate_fn(data_columns=['Close']):
    def collate(batch):
        lookback_batch, horizon_batch, static_attributes_batch, dynamic_covariates_batch = [], [], [], []
        for i in range(len(batch)):
            lookback, horizon, static_attributes = batch[i]

            dates = pd.concat([lookback['Date'], horizon['Date']])
            weekday_onehot = F.one_hot(torch.tensor(dates.dt.weekday.values), num_classes=7).to(torch.float32)
            month_onehot = F.one_hot(torch.tensor(dates.dt.month.values-1), num_classes=12).to(torch.float32)
            dynamic_covariates = torch.cat([weekday_onehot, month_onehot], dim=-1)

            lookback_batch.append(torch.tensor(lookback[data_columns].values).squeeze())
            horizon_batch.append(torch.tensor(horizon[data_columns].values).squeeze())
            
            static_attributes_batch.append(static_attributes)
            dynamic_covariates_batch.append(dynamic_covariates)
        return torch.stack(lookback_batch), torch.stack(horizon_batch), torch.stack(static_attributes_batch), torch.stack(dynamic_covariates_batch)

    return collate

def roll_norm(x:torch.Tensor, std_norm=True, eps=1e-12):
    x0 = x[:, 0]
    x_roll = torch.roll(x, shifts=1, dims=1)
    delta = x/(x_roll + eps) - 1

    if len(delta.shape) == 3:
        delta[:, 0, :] = 0.0
    else:
        delta[:, 0] = 0.0

    if std_norm:
        mean = torch.mean(delta, dim=1, keepdim=True)
        std = torch.std(delta, dim=1, keepdim=True)
        delta = (delta - mean) / (std + eps)
        return delta, x0, mean, std
    return delta, x0

def reverse_roll_norm(delta: torch.Tensor, x0: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, std_norm=True, eps=1e-12):
    if std_norm:
        delta = delta * (std + eps) + mean
    x = torch.zeros_like(delta)
    x[:, 0] = x0
    for i in range(1, x.shape[1]):
        x[:, i] = (x[:, i-1] + eps) * (delta[:, i]+1)
    return x
