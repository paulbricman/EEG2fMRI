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
            nn.AvgPool2d((1, 4)),
            nn.Conv2d(1, 32, (1, 25)),
            nn.AvgPool2d((1, 2)),
	        nn.BatchNorm2d(32),
            nn.ReLU(),
		nn.Conv2d(32, 32, (1, 25)),
            nn.AvgPool2d((1, 2)),
	        nn.BatchNorm2d(32),
            nn.ReLU(),
		nn.Conv2d(32, 32, (1, 25)),
            nn.AvgPool2d((1, 2)),
	        nn.BatchNorm2d(32),
            nn.ReLU(),
		nn.Conv2d(32, 32, (1, 25)),
            nn.AvgPool2d((1, 2)),
	        nn.BatchNorm2d(32),
            nn.ReLU(),
		nn.Conv2d(32, 32, (1, 25)),
            nn.AvgPool2d((1, 2)),
	        nn.BatchNorm2d(32),
            nn.ReLU(),
		nn.Conv2d(32, 32, (1, 25)),
            nn.AvgPool2d((1, 2)),
	        nn.BatchNorm2d(32),
            nn.ReLU(),
		nn.Conv2d(32, 32, (1, 25)),
            nn.AvgPool2d((1, 2)),
	        nn.BatchNorm2d(32),
            nn.ReLU(),
		nn.Conv2d(32, 32, (1, 25)),
	        nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
		nn.Linear(10880, 50),
		nn.ReLU(),
            nn.Linear(50, np.prod(self.fmri_dimensions), False),
		nn.Tanh()
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
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=0.01)


class TransformerModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.eeg_frequency = 1000  # Hz
        self.sample_length = 30  # seconds
        self.eeg_channels = 34
        self.eeg_sample_length = self.eeg_frequency * self.sample_length
        self.fmri_dimensions = [64, 64, 32]

        self.encoder = nn.Sequential(
            nn.AvgPool2d((1, 4)),
            nn.Conv2d(1, 32, (1, 25)),
            nn.AvgPool2d((1, 2)),
	        nn.BatchNorm2d(32),
            nn.ReLU(),
		nn.Conv2d(32, 32, (1, 25)),
            nn.AvgPool2d((1, 2)),
	        nn.BatchNorm2d(32),
            nn.ReLU(),
		nn.Conv2d(32, 32, (1, 25)),
            nn.AvgPool2d((1, 2)),
	        nn.BatchNorm2d(32),
            nn.ReLU(),
		nn.Conv2d(32, 32, (1, 25)),
            nn.AvgPool2d((1, 2)),
	        nn.BatchNorm2d(32),
            nn.ReLU(),
		nn.Conv2d(32, 32, (1, 25)),
            nn.AvgPool2d((1, 2)),
	        nn.BatchNorm2d(32),
            nn.ReLU(),
		nn.Conv2d(32, 32, (1, 25)),
            nn.AvgPool2d((1, 2)),
	        nn.BatchNorm2d(32),
            nn.ReLU(),
		nn.Conv2d(32, 32, (1, 25)),
            nn.AvgPool2d((1, 2)),
	        nn.BatchNorm2d(32),
            nn.ReLU(),
		nn.Conv2d(32, 32, (1, 25)),
	        nn.BatchNorm2d(32),
            nn.ReLU()
        ).double()

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(32 * 10, 8).double()
        self.transformer = nn.TransformerEncoder(self.transformer_encoder_layer, 6).double()

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(34 * 32 * 10, 100),
		nn.ReLU(),
            nn.Linear(100, np.prod(self.fmri_dimensions), False),
		nn.Tanh()
        ).double()

    def forward(self, x):
        x = x.view(x.size(0), 1, self.eeg_channels, self.eeg_sample_length)
        encoded = self.encoder(x)
        encoded = torch.transpose(encoded, 1, 2).reshape(encoded.size(0), self.eeg_channels, -1)
        transformed = self.transformer(encoded)
        decoded = self.decoder(transformed)
        return decoded.view(decoded.size(0), *self.fmri_dimensions)

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
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=0.01)


