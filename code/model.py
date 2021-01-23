import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
import numpy as np
from data import fmri_preview, eeg_preview
import matplotlib.pyplot as plt


class TransformerModel(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.channel_encoder = nn.Linear(7500, 512).double()
        self.slice_encoder = nn.Linear(63 * 52, 512).double()
        self.output_decoder = nn.Linear(512, 63 * 52, bias=False).double()
        self.transformer = nn.Transformer().double()

        self.sos_emb = nn.Embedding(1, 512).double()
        self.channel_pos_emb = nn.Embedding(34, 512).double()
        self.slice_pos_emb = nn.Embedding(53, 512).double()

        self.register_buffer('channel_pos_emb_idx', torch.arange(0, 34, dtype=torch.long))
        self.register_buffer('slice_pos_emb_idx', torch.arange(0, 53, dtype=torch.long))
        self.register_buffer('sos_emb_idx', torch.LongTensor([0]))
        self.register_buffer('tgt_mask', self.transformer.generate_square_subsequent_mask(53))

    def forward(self, x, y):
        # Downsample to 250 Hz
        x = nn.AvgPool2d((1, 4))(x) # N, 34, 7500

        # Generate channel embeddings from channel recordings
        channel_emb = torch.cat([self.channel_encoder(sample) for sample in x]).view(x.size(0), 34, 512)
        channel_emb = torch.cat([sample + self.channel_pos_emb(self.channel_pos_emb_idx) for sample in channel_emb]).view(y.size(0), 34, 512)

        # Shift slices in target
        slice_emb = torch.cat([self.slice_encoder(sample) for sample in y]).view(y.size(0), 53, 512)
        slice_emb = torch.cat([torch.cat((sample[:-2], sample[-2:])) for sample in slice_emb]).view(y.size(0), 53, 512)
        slice_emb = torch.cat([sample + self.slice_pos_emb(self.slice_pos_emb_idx) for sample in slice_emb]).view(y.size(0), 53, 512)

        # Reshape for transformer
        channel_emb = torch.movedim(channel_emb, 1, 0)
        slice_emb = torch.movedim(slice_emb, 1, 0)

        # Generate target mask and pipe through transformer
        output = self.transformer(channel_emb, slice_emb, tgt_mask = self.tgt_mask)
        output = self.output_decoder(output)
        output = torch.sigmoid(output)

        return output        

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_tgt = y.view(y.size(0), 53, 63 * 52)
        y_output = torch.movedim(y_tgt, 1, 0)
        y_pred = self.forward(x, y_tgt)

        #fmri_preview(torch.cat([slice[0] for slice in y_output]).view(53, 63, 52).cpu().numpy())
        #fmri_preview(torch.cat([slice[0] for slice in y_pred]).view(53, 63, 52).cpu().detach().numpy())

        loss = F.mse_loss(y_output, y_pred)
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
        return torch.optim.Adam(self.parameters(), lr=0.1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * 0.1
        pe[:, 1::2] = torch.cos(position * div_term) * 0.1
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
