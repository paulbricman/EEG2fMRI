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
        self.flattened_slice_length = 63 * 52

        # Start of sequence embedding
        self.sos = nn.Embedding(1, self.flattened_slice_length).double()

        # Encoder for generating channel embeddings from channel recordings
        self.encoder = nn.Linear(30000, self.flattened_slice_length).double()

        # Transformer for generating slices from channel embeddings
        self.transformer = nn.Transformer(self.flattened_slice_length).double()

    def forward(self, x, y):
        batch_size = x.size(0)
        channel_recordings = x.view(batch_size * self.eeg_channels, self.eeg_sample_length)

        # Generate channel embeddings from channel recordings
        channel_embeddings = self.encoder(channel_recordings)
        channel_embeddings = channel_embeddings.view(self.eeg_channels, batch_size, self.flattened_slice_length)

        # TODO apply positional embeddings
        # Generate slices from channel embeddings
        tgt = y.view(53, batch_size, self.flattened_slice_length)
        tgt = torch.cat((self.sos, tgt[:-1]))
        mask = (torch.triu(torch.ones(53, 53)) == 1).transpose(0, 1)
        slices = self.transformer(channel_embeddings, tgt, tgt_mask = mask)
        

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


