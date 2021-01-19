import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
import numpy as np


class TransformerModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.eeg_frequency = 1000  # Hz
        self.sample_length = 30  # seconds
        self.eeg_channels = 34
        self.eeg_sample_length = self.eeg_frequency * self.sample_length
        self.fmri_dimensions = [53, 63, 52]

        # Encoder for generating channel embeddings from channel recordings
        self.encoder = nn.Linear(30000, 512).double()

        # Transformer for generating slice embeddings from channel embeddings
        self.transformer = nn.Transformer().double()

        # Decoder for generating slices from slice embeddings
        self.decoder = nn.Linear(512, 63 * 52)

    def forward(self, x):
        channel_recordings = x.view(x.size(0), 1, self.eeg_channels, self.eeg_sample_length)

        # Generate channel embeddings from channel recordings
        # Generate slice embeddings from channel embeddings
        # Generate slices from slice embeddings
        

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


