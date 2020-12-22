import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
import numpy as np

class ConvolutionalModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.eeg_frequency = 1000  # Hz
        self.sample_length = 30  # seconds
        self.eeg_channels = 34
        self.eeg_sample_length = self.eeg_frequency * self.sample_length
        self.fmri_dimensions = [64, 64, 32]

        self.model = nn.Sequential(
            nn.Conv2d(1, 4, (1, 5), stride=2),
            nn.AvgPool2d((1, 2)),
	    nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, (1, 5), stride=2),
            nn.AvgPool2d((1, 2)),
	    nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, (1, 5), stride=2),
            nn.AvgPool2d((1, 2)),
	    nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, (1, 5), stride=2),
            nn.AvgPool2d((1, 2)),
	    nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (1, 5), stride=2),
            nn.AvgPool2d((1, 2)),
	    nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, (1, 5), stride=2),
            nn.AvgPool2d((1, 2)),
	    nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
	    nn.Linear(768, 1000),
	    nn.ReLU(),
            nn.Linear(1000, np.prod(self.fmri_dimensions))
        ).double()

    def forward(self, x):
        x = x.view(x.size(0), 1, self.eeg_channels, self.eeg_sample_length)
        y_pred = self.model(x)
        y_pred = y_pred.view(y_pred.size(0), *self.fmri_dimensions)
        return y_pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.mse_loss(y, self.forward(x))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = F.mse_loss(y, self.forward(x))
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.2)
