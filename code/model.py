import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
import numpy as np


class TransformerModel(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Linear(7500, 63 * 52).double()
        self.transformer = nn.Transformer(63 * 52, nhead = 12, \
		num_encoder_layers = 2, num_decoder_layers = 2).double()

        self.sos_emb = nn.Embedding(1, 63 * 52).double()
        self.channel_pos_emb = nn.Embedding(34, 63 * 52).double()
        self.slice_pos_emb = nn.Embedding(53, 63 * 52).double()

        self.register_buffer('sos_emb_idx', torch.LongTensor([0]))
        self.register_buffer('channel_pos_emb_idx', torch.arange(0, 34, dtype=torch.long))
        self.register_buffer('slice_pos_emb_idx', torch.arange(0, 53, dtype=torch.long))
        self.register_buffer('tgt_mask', self.transformer.generate_square_subsequent_mask(53))

    def forward(self, x, y):
        # Downsample to 250 Hz
        x = nn.AvgPool2d((1, 4))(x) # N, 34, 7500

        # Generate channel embeddings from channel recordings
        channel_emb = torch.cat([self.encoder(sample) for sample in x]).view(x.size(0), 34, 63 * 52)

        # Apply channel positional embeddings
        channel_emb = torch.cat([sample + self.channel_pos_emb(self.channel_pos_emb_idx) for sample in channel_emb]).view(x.size(0), 34, 63 * 52)

        # Shift slices in target
        slices = torch.cat([torch.cat((self.sos_emb(self.sos_emb_idx), sample[:-1])) for sample in y]).view(y.size(0), 53, 63 * 52)

        # Apply slice positional embeddings
        slices = torch.cat([sample + self.slice_pos_emb(self.slice_pos_emb_idx) for sample in slices]).view(y.size(0), 53, 63 * 52)

        # Apply slice positional embeddings
        channel_emb = torch.movedim(channel_emb, 1, 0)
        slices = torch.movedim(slices, 1, 0)

        # Generate target mask and pipe through transformer
        mask = self.transformer.generate_square_subsequent_mask(53)
        output = self.transformer(channel_emb, slices, tgt_mask = self.tgt_mask)

        return output        

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_tgt = y.view(y.size(0), 53, 63 * 52)
        y_output = torch.movedim(y_tgt, 1, 0)

        loss = F.mse_loss(y_output, self.forward(x, y_tgt))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_tgt = y.view(y.size(0), 53, 63 * 52)
        y_output = torch.movedim(y_tgt, 1, 0)

        loss = F.mse_loss(y_output, self.forward(x, y_tgt))
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


