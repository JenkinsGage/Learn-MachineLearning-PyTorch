import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from data import StockDataset, roll_norm, reverse_roll_norm
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import get_linear_schedule_with_warmup
from model import TiDE
from loss import FourierLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

DATA_ROOT = './Data/StockMarket/'
LOOKBACK_STEPS = 270
HORIZON_STEPS = 60
DYNAMIC_COVARIATES_PROJECTION_DIM = 2
HIDDEN_DIM = 512
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 4
DECODER_OUTPUT_DIM = 64
TEMPORAL_DECODER_HIDDEN_DIM = 64
DROPOUT = 0.5

DATA_COLUMNS = ['Close']
DATASET_ROLL_STEPS=7
TRAIN_SPLIT = 0.95
BATCH_SIZE = 64
MODEL_SAVE_PATH = './Models/'
EVAL_PLOT = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # A GPU with memory >=8GB is capable of training
NUM_WORKERS = 4
EPOCHS = 4
NUM_WARMUP_EPOCHS = 1
LEARNING_RATE = 4e-5
PENALTY_L1 = 1e-6
FFT_TOPK = 12
FFT_ALPHA = 0.2

# The following parameters are determined by the dataset
STATIC_ATTRIBUTES_DIM = 20
DYNAMIC_COVARIATES_DIM = 19
gdt = 0

def collate(batch):
    lookback_batch, horizon_batch, static_attributes_batch, dynamic_covariates_batch = [], [], [], []
    for i in range(len(batch)):
        lookback, horizon, static_attributes = batch[i]

        dates = pd.concat([lookback['Date'], horizon['Date']])
        weekday_onehot = F.one_hot(torch.tensor(dates.dt.weekday.values), num_classes=7).to(torch.float32)
        month_onehot = F.one_hot(torch.tensor(dates.dt.month.values-1), num_classes=12).to(torch.float32)
        dynamic_covariates = torch.cat([weekday_onehot, month_onehot], dim=-1)

        lookback_batch.append(torch.tensor(lookback[DATA_COLUMNS].values).squeeze())
        horizon_batch.append(torch.tensor(horizon[DATA_COLUMNS].values).squeeze())
        
        static_attributes_batch.append(static_attributes)
        dynamic_covariates_batch.append(dynamic_covariates)
    return torch.stack(lookback_batch), torch.stack(horizon_batch), torch.stack(static_attributes_batch), torch.stack(dynamic_covariates_batch)

def train_batch(model, optimizer, lookback_batch, horizon_batch, static_attributes_batch, dynamic_covariates_batch):
    global gdt
    model.train()
    optimizer.zero_grad()
    predictions = model(lookback_batch, static_attributes_batch, dynamic_covariates_batch)
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    fft_err = criterion_fft(predictions, horizon_batch)
    loss = criterion(predictions, horizon_batch) + PENALTY_L1 * l1_norm + FFT_ALPHA * fft_err
    if not torch.isnan(loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.9)
        optimizer.step()
        scheduler.step()
        gdt += 1
        with torch.no_grad():
            mae = F.l1_loss(predictions, horizon_batch)
            writer.add_scalar('Train/Loss', loss.item(), gdt)
            writer.add_scalar('Train/MAE', mae.item(), gdt)
            writer.add_scalar('Train/L1Norm', l1_norm.item(), gdt)
            writer.add_scalar('Train/FFTErr', fft_err.item(), gdt)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], gdt)
    else:
        print(f'NaN loss at step {gdt}')

def train_epochs(model, optimizer, epochs):
    global gdt
    for epoch in tqdm(range(epochs), desc='Training'):
        for lookback_batch, horizon_batch, static_attributes_batch, dynamic_covariates_batch in tqdm(ts_dataloader_train, desc=f'Epoch {epoch}'):
            lookback_batch = lookback_batch.to(DEVICE)
            lookback_batch, _, _, _ = roll_norm(lookback_batch)
            horizon_batch = horizon_batch.to(DEVICE)
            horizon_batch, _, _, _ = roll_norm(horizon_batch)
            static_attributes_batch = static_attributes_batch.to(DEVICE)
            dynamic_covariates_batch = dynamic_covariates_batch.to(DEVICE)
            train_batch(model, optimizer, lookback_batch, horizon_batch, static_attributes_batch, dynamic_covariates_batch)    
        eval_epoch(model)

def eval_epoch(model):
    global gdt
    global EVAL_PLOT
    model.eval()
    total_losses = 0.0
    total_mae = 0.0
    total_mae_reversed = 0.0
    total_fft_err = 0.0
    steps = 0
    with torch.no_grad():
        for lookback_batch, horizon_batch, static_attributes_batch, dynamic_covariates_batch in tqdm(ts_dataloader_val, desc='Run Validation'):
            lookback_batch = lookback_batch.to(DEVICE)
            lookback_batch, l0, lmean, lstd = roll_norm(lookback_batch)
            horizon_batch = horizon_batch.to(DEVICE)
            horizon_batch, h0, hmean, hstd = roll_norm(horizon_batch)
            static_attributes_batch = static_attributes_batch.to(DEVICE)
            dynamic_covariates_batch = dynamic_covariates_batch.to(DEVICE)

            predictions = model(lookback_batch, static_attributes_batch, dynamic_covariates_batch)
            loss = criterion(predictions, horizon_batch)
            mae = F.l1_loss(predictions, horizon_batch)
            fft_err = criterion_fft(predictions, horizon_batch)
            total_losses += loss.item() + FFT_ALPHA*fft_err.item()
            total_mae += mae.item()
            total_mae_reversed += F.l1_loss(reverse_roll_norm(predictions, h0, hmean, hstd), reverse_roll_norm(horizon_batch, h0, hmean, hstd)).item()
            total_fft_err += fft_err.item()
            steps += 1

            if EVAL_PLOT:
                lookbacks = reverse_roll_norm(lookback_batch, l0, lmean, lstd).cpu().numpy()
                horizons = reverse_roll_norm(horizon_batch, h0, hmean, hstd).cpu().numpy()
                predictions_ = reverse_roll_norm(predictions, h0, hmean, hstd).cpu().numpy()
                for i, (l, h, p) in enumerate(zip(lookbacks, horizons, predictions_)):
                    # x1 = np.arange(len(l))
                    x2 = np.arange(len(l), len(l)+len(h))
                    fig, ax = plt.subplots()
                    # ax.plot(x1, l, label='Lookback', color='blue')
                    ax.plot(x2, h, label='Horizon', color='green')
                    ax.plot(x2, p, label='Prediction', color='red')
                    h_ma = pd.Series(h).rolling(window=5).mean()
                    p_ma = pd.Series(p).rolling(window=5).mean()
                    ax.plot(x2, h_ma, label='Horizon-MA', color='cyan')
                    ax.plot(x2, p_ma, label='Prediction-MA', color='magenta')
                    ax.legend()
                    writer.add_figure(f'ValFig/G{gdt}-B{steps}-{i}', fig)
                EVAL_PLOT = False
        EVAL_PLOT = True
        writer.add_scalar('Val/Loss', total_losses / steps, gdt)
        writer.add_scalar('Val/MAE', total_mae / steps, gdt)
        writer.add_scalar('Val/MAE(INV)', total_mae_reversed / steps, gdt)
        writer.add_scalar('Val/FFTErr', total_fft_err / steps, gdt)
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f'TiDE_{gdt}_MAE({total_mae/steps:.4f}).pt'))


if __name__ == '__main__':
    stock_dataset = StockDataset(data_root=DATA_ROOT, lookback_steps=LOOKBACK_STEPS, horizon_steps=HORIZON_STEPS, roll_steps=DATASET_ROLL_STEPS)
    num_warmup_steps = NUM_WARMUP_EPOCHS * int(TRAIN_SPLIT * len(stock_dataset)) // BATCH_SIZE
    num_training_steps = EPOCHS * int(TRAIN_SPLIT * len(stock_dataset)) // BATCH_SIZE
    sampler_train = SubsetRandomSampler(range(int(TRAIN_SPLIT * len(stock_dataset))))
    sampler_val = SubsetRandomSampler(range(int(TRAIN_SPLIT * len(stock_dataset)), len(stock_dataset)))
    ts_dataloader_train = DataLoader(stock_dataset, batch_size=BATCH_SIZE, collate_fn=collate, sampler=sampler_train, num_workers=NUM_WORKERS)
    ts_dataloader_val = DataLoader(stock_dataset, batch_size=BATCH_SIZE, collate_fn=collate, sampler=sampler_val, num_workers=NUM_WORKERS)

    model = TiDE(lookback_steps=LOOKBACK_STEPS, horizon_steps=HORIZON_STEPS, static_attributes_dim=STATIC_ATTRIBUTES_DIM, dynamic_covariates_dim=DYNAMIC_COVARIATES_DIM, dynamic_covariates_projection_dim=DYNAMIC_COVARIATES_PROJECTION_DIM,
                hidden_dim=HIDDEN_DIM, num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS, decoder_output_dim=DECODER_OUTPUT_DIM, temporal_decoder_hidden_dim=TEMPORAL_DECODER_HIDDEN_DIM, dropout=DROPOUT)
    print(model)
    print(f'Model Num Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f} M')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'{name}: {param.numel()/1e3:.1f}K')
    model = model.to(DEVICE)

    criterion = nn.MSELoss()
    criterion_fft = FourierLoss(fft_topk=FFT_TOPK, p=2)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    writer = SummaryWriter()

    print(f'Start Training | Total Steps: {num_training_steps} | Warmup Steps: {num_warmup_steps}')
    train_epochs(model, optimizer, epochs=EPOCHS)
