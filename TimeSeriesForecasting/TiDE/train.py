import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from data import StockDataset, TimeSeriesDataset, collate_fn
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup
from model import TiDE, LogNorm
from loss import FourierLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

DATA_ROOT = './Data/StockMarket/'
LOOKBACK_STEPS = 360
HORIZON_STEPS = 60
DYNAMIC_COVARIATES_PROJECTION_DIM = 3
HIDDEN_DIM = 512
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DECODER_OUTPUT_DIM = 64
TEMPORAL_DECODER_HIDDEN_DIM = 64
DROPOUT = 0.5

STOCK_SUBSET = 200
TRAINING_DATASET_GAP = 30 # Two time series for training will have a gap of TRAINING_DATASET_GAP steps
VALIDATION_DATASET_GAP = 30
TRAIN_SPLIT = 0.95
BATCH_SIZE = 64
MODEL_SAVE_PATH = './Models/'
EVAL_PLOT = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # A GPU with memory >=8GB is capable of training
NUM_WORKERS = 8
EPOCHS = 20
NUM_WARMUP_EPOCHS = 2
LEARNING_RATE = 4e-5
PENALTY_L1 = 1e-8
FFT_TOPK = 8
FFT_ALPHA = 0.25

# The following parameters are determined by the dataset
STATIC_ATTRIBUTES_DIM = 20
DYNAMIC_COVARIATES_DIM = 19
gdt = 0

def train_batch(model, optimizer, lookback_batch, horizon_batch, static_attributes_batch, dynamic_covariates_batch):
    global gdt
    model.train()
    optimizer.zero_grad()
    predictions = model(lookback_batch, static_attributes_batch, dynamic_covariates_batch)
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    fft_err = criterion_fft(predictions, horizon_batch)
    loss = criterion(predictions, horizon_batch) + PENALTY_L1 * l1_norm + FFT_ALPHA * fft_err
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

def train_epochs(model, optimizer, epochs):
    global gdt
    for epoch in tqdm(range(epochs), desc='Training'):
        for lookback_batch, horizon_batch, static_attributes_batch, dynamic_covariates_batch in tqdm(timeseries_dataloader_train, desc=f'Epoch {epoch}'):
            lookback_batch = lookback_batch.to(DEVICE)
            lookback_batch = log_norm(lookback_batch)
            horizon_batch = horizon_batch.to(DEVICE)
            horizon_batch = log_norm.normalize(horizon_batch)
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
    total_mae_inversed = 0.0
    total_fft_err = 0.0
    steps = 0
    with torch.no_grad():
        for lookback_batch, horizon_batch, static_attributes_batch, dynamic_covariates_batch in tqdm(timeseries_dataloader_val, desc='Run Validation'):
            lookback_batch = lookback_batch.to(DEVICE)
            lookback_batch = log_norm(lookback_batch)
            horizon_batch = horizon_batch.to(DEVICE)
            horizon_batch = log_norm.normalize(horizon_batch)
            static_attributes_batch = static_attributes_batch.to(DEVICE)
            dynamic_covariates_batch = dynamic_covariates_batch.to(DEVICE)
            predictions = model(lookback_batch, static_attributes_batch, dynamic_covariates_batch)
            loss = criterion(predictions, horizon_batch)
            mae = F.l1_loss(predictions, horizon_batch)
            fft_err = criterion_fft(predictions, horizon_batch)
            total_losses += loss.item()
            total_mae += mae.item()
            total_mae_inversed += F.l1_loss(log_norm.inverse(predictions), log_norm.inverse(horizon_batch)).item()
            total_fft_err += fft_err.item()
            steps += 1

            if EVAL_PLOT:
                lookbacks = log_norm.inverse(lookback_batch).cpu().numpy()
                horizons = log_norm.inverse(horizon_batch).cpu().numpy()
                predictions_ = log_norm.inverse(predictions).cpu().numpy()
                for i, (l, h, p) in enumerate(zip(lookbacks, horizons, predictions_)):
                    # x1 = np.arange(len(l))
                    x2 = np.arange(len(l), len(l)+len(h))
                    fig, ax = plt.subplots()
                    # ax.plot(x1, l, label='Lookback', color='blue')
                    ax.plot(x2, h, label='Horizon', color='green')
                    ax.plot(x2, p, label='Prediction', color='red')
                    h_ma = pd.Series(h).rolling(window=5).mean()
                    p_ma = pd.Series(p).rolling(window=5).mean()
                    ax.plot(x2, h_ma, label='Horizon_MA', color='cyan')
                    ax.plot(x2, p_ma, label='Prediction_MA', color='magenta')
                    ax.legend()
                    writer.add_figure(f'ValFig/G{gdt}-B{steps}-{i}', fig)
                EVAL_PLOT = False
        EVAL_PLOT = True
        writer.add_scalar('Val/Loss', total_losses / steps, gdt)
        writer.add_scalar('Val/MAE', total_mae / steps, gdt)
        writer.add_scalar('Val/MAE(INV)', total_mae_inversed / steps, gdt)
        writer.add_scalar('Val/FFTErr', total_fft_err / steps, gdt)
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f'TiDE_{gdt}_MAE({total_mae/steps:.4f}).pt'))


if __name__ == '__main__':
    stock_dataset = StockDataset(data_root=DATA_ROOT, subset=STOCK_SUBSET)
    stock_dataset_train, stock_dataset_val = random_split(stock_dataset, [TRAIN_SPLIT, 1-TRAIN_SPLIT])
    timeseries_dataset_train = TimeSeriesDataset(stock_dataset_train, lookback_steps=LOOKBACK_STEPS, horizon_steps=HORIZON_STEPS, gap=TRAINING_DATASET_GAP)
    timeseries_dataset_val = TimeSeriesDataset(stock_dataset_val, lookback_steps=LOOKBACK_STEPS, horizon_steps=HORIZON_STEPS, gap=VALIDATION_DATASET_GAP)
    num_warmup_steps = NUM_WARMUP_EPOCHS * len(timeseries_dataset_train) // BATCH_SIZE
    num_training_steps = EPOCHS * len(timeseries_dataset_train) // BATCH_SIZE

    timeseries_dataloader_train = DataLoader(timeseries_dataset_train, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)
    timeseries_dataloader_val = DataLoader(timeseries_dataset_val, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=True)

    model = TiDE(lookback_steps=LOOKBACK_STEPS, horizon_steps=HORIZON_STEPS, static_attributes_dim=STATIC_ATTRIBUTES_DIM, dynamic_covariates_dim=DYNAMIC_COVARIATES_DIM, dynamic_covariates_projection_dim=DYNAMIC_COVARIATES_PROJECTION_DIM,
                hidden_dim=HIDDEN_DIM, num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS, decoder_output_dim=DECODER_OUTPUT_DIM, temporal_decoder_hidden_dim=TEMPORAL_DECODER_HIDDEN_DIM, dropout=DROPOUT)
    print(model)
    print(f'Model Num Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f} M')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'{name}: {param.numel()/1e3:.1f}K')
    model = model.to(DEVICE)

    log_norm = LogNorm()
    criterion = nn.MSELoss()
    criterion_fft = FourierLoss(fft_topk=FFT_TOPK, p=2)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    writer = SummaryWriter()

    print(f'Start Training | Total Steps: {num_training_steps} | Warmup Steps: {num_warmup_steps}')
    train_epochs(model, optimizer, epochs=EPOCHS)
