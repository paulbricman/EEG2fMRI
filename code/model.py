import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
import numpy as np


class TransformerModel(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.sos_emb = nn.Embedding(1, 63 * 52).double()
        self.channel_pos_emb = nn.Embedding(34, 63 * 52).double()
        self.slice_pos_emb = nn.Embedding(53, 63 * 52).double()

        self.encoder = nn.Linear(7500, 63 * 52).double()
        self.transformer = nn.Transformer(63 * 52).double()

    def forward(self, x, y):
        # Downsample to 250 Hz
        x = nn.AvgPool2d(x, (1, 4))

        # Generate channel embeddings from channel recordings
        channel_emb = torch.cat([self.encoder(sample) for sample in x]) # N, S, E

        # Apply channel positional embeddings
        channel_emb = torch.cat([sample + self.channel_pos_emb for sample in channel_emb]) # N, S, E

        # Shift slices in target
        slices = y.view(y.size(0), 53, 63 * 52)
        slices = torch.cat([torch.cat((self.sos_emb, sample[:-1])) for sample in slices])

        # Apply slice positional embeddings
        slices = torch.cat([sample + self.slice_pos_emb for sample in slices]) # N, T, E

        # Apply slice positional embeddings
        channel_emb = torch.moveaxis(channel_emb, 1, 0)
        slices = torch.moveaxis(slices, 1, 0)

        mask = self.transformer.generate_square_subsequent_mask(53)
        output = self.transformer(channel_emb, slices, tgt_mask = mask)
        

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.mse_loss(y, self.forward(x, y))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = F.mse_loss(y, self.forward(x, y))
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=0.01)


