import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
import numpy as np
from data import fmri_preview, eeg_preview
import matplotlib.pyplot as plt


class ConvolutionalModel(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.AvgPool2d((1, 4)),
            nn.Conv2d(1, 32, (1, 99), padding=(0, 49)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 64)),
            nn.Flatten()
        )

        self.transcoder = nn.Sequential(
            nn.Linear(127296, 100),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.Linear(100, 1000),
            nn.BatchNorm1d(1000),
            nn.ELU(),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ELU(),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ELU(),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ELU(),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ELU(),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ELU(),
            nn.Linear(1000, 100),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.Linear(100, 173628),
            nn.ELU()       
        )

        self.decoder = nn.Sequential(
            nn.Conv3d(1, 16, (5, 5, 5), padding=2),
            nn.BatchNorm3d(16),
            nn.ELU(),
            nn.Conv3d(16, 16, (5, 5, 5), padding=2),
            nn.BatchNorm3d(16),
            nn.ELU(),
            nn.Conv3d(16, 16, (5, 5, 5), padding=2),
            nn.BatchNorm3d(16),
            nn.ELU(),
            nn.Conv3d(16, 16, (5, 5, 5), padding=2),
            nn.BatchNorm3d(16),
            nn.ELU(),
            nn.Conv3d(16, 1, (5, 5, 5), padding=2),
            nn.ELU()
        )

    def forward(self, x_full):
        x = x_full[0].view(x.size(0), 1, 34, 30000)
        x = self.encoder(x).view(x_full[0].size(0), 127296)
        x = self.transcoder(x).view(x.size(0), 1, 53, 63, 52)
        x = self.decoder(x).view(x.size(0), 53, 63, 52)
        return x        

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        y_pred = torch.mul(y_pred, y[1])
        y_gt = torch.mul(y[0], y[1])

        loss = F.mse_loss(y_gt, y_pred, reduction='sum') / y[1].sum()

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        y_pred = torch.mul(y_pred, y[1])
        y_gt = torch.mul(y[0], y[1])

        loss = F.mse_loss(y_gt, y_pred, reduction='sum') / y[1].sum()

        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5),
            'monitor': 'train_loss'
        }
