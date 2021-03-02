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
            nn.Conv2d(1, 512, (34, 49), padding=(0, 24), dilation=(1, 2), stride=(1, 4)),
            nn.ELU(),
            nn.BatchNorm2d(512),
            nn.AvgPool2d((1, 16)),
            nn.Flatten()
        )

        self.transcoder = nn.Sequential(
            nn.Linear(59392, 100),
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
            nn.Linear(1000, 100),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.Linear(100, 173628),
            nn.ELU()       
        )

        self.decoder = nn.Sequential(
            nn.Conv3d(1, 16, (5, 5, 5), padding=2),
	    nn.ELU(),
            nn.Conv3d(16, 16, (5, 5, 5), padding=2),
	    nn.ELU(),
            nn.Conv3d(16, 16, (5, 5, 5), padding=2),
	    nn.ELU(),
            nn.Conv3d(16, 1, (5, 5, 5), padding=2),
	    nn.ELU()
        )

    def forward(self, x_full):
        x = x_full[0].view(x_full[0].size(0), 1, 34, 30000)
        x = self.encoder(x).view(x.size(0), 59392)
        x = self.transcoder(x).view(x.size(0), 1, 53, 63, 52)
        x = self.decoder(x).view(x.size(0), 53, 63, 52)
        return x        

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        y_pred = torch.mul(y_pred, y[1])
        y_gt = torch.mul(y[0], y[1])

        loss = F.mse_loss(y_pred, y_gt, reduction='none')
        loss = torch.sum(loss, (1, 2, 3))
        loss = torch.div(loss, torch.sum(y[1], (1, 2, 3)))
        loss = torch.mean(loss)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        y_pred = torch.mul(y_pred, y[1])
        y_gt = torch.mul(y[0], y[1])

        loss = F.mse_loss(y_pred, y_gt, reduction='none')
        loss = torch.sum(loss, (1, 2, 3))
        loss = torch.div(loss, torch.sum(y[1], (1, 2, 3)))
        loss = torch.mean(loss)

        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10),
            'monitor': 'train_loss'
        }


class TransformerModel(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.channel_encoder = nn.Linear(7500, 512)
        self.slice_encoder = nn.Linear(63 * 52, 512)
        self.output_decoder = nn.Linear(512, 63 * 52)
        self.transformer = nn.Transformer(num_encoder_layers=1, num_decoder_layers=1)

        self.sos_emb = nn.Embedding(1, 512)
        self.channel_pos_emb = nn.Embedding(34, 512)
        self.slice_pos_emb = nn.Embedding(53, 512)

        self.register_buffer('channel_pos_emb_idx', torch.arange(0, 34, dtype=torch.long))
        self.register_buffer('slice_pos_emb_idx', torch.arange(0, 53, dtype=torch.long))
        self.register_buffer('sos_emb_idx', torch.LongTensor([0]))
        self.register_buffer('tgt_mask', self.transformer.generate_square_subsequent_mask(53))

    def forward(self, x, y=None, mode='train'):
        # Downsample to 250 Hz
        x = nn.AvgPool2d((1, 4))(x) # N, 34, 7500

        # Generate channel embeddings from channel recordings
        channel_emb = torch.cat([self.channel_encoder(sample) for sample in x]).view(x.size(0), 34, 512)
        channel_emb = torch.cat([sample + self.channel_pos_emb(self.channel_pos_emb_idx) for sample in channel_emb]).view(x.size(0), 34, 512)

        if mode == 'train':
            # Shift slices in target
            slice_emb = torch.cat([self.slice_encoder(sample) for sample in y]).view(y.size(0), 53, 512)
            slice_emb = torch.cat([torch.cat((self.sos_emb(self.sos_emb_idx), sample[:-1])) for sample in slice_emb]).view(y.size(0), 53, 512)
            slice_emb = torch.cat([sample + self.slice_pos_emb(self.slice_pos_emb_idx) for sample in slice_emb]).view(y.size(0), 53, 512)

            # Reshape for transformer
            channel_emb = torch.movedim(channel_emb, 1, 0)
            slice_emb = torch.movedim(slice_emb, 1, 0)

            # Generate target mask and pipe through transformer
            output = self.transformer(channel_emb, slice_emb, tgt_mask = self.tgt_mask)
            output = self.output_decoder(output)

            return output

        elif mode == 'test':
            # Start with sos embedding and for each batch
            slice_emb = torch.cat([self.sos_emb(self.sos_emb_idx) for sample in x]).view(x.size(0), 1, 512)
            slice_emb = torch.cat([sample + self.slice_pos_emb(self.sos_emb_idx) for sample in slice_emb]).view(x.size(0), 1, 512)
            
            channel_emb = torch.movedim(channel_emb, 1, 0)
            slice_emb = torch.movedim(slice_emb, 1, 0)
            output_volume = []

            for slice_idx in range(53):
                tgt_mask = self.transformer.generate_square_subsequent_mask(slice_idx + 1)
                output = self.transformer(channel_emb, slice_emb, tgt_mask = tgt_mask)

                new_slice = self.output_decoder(output)[-1].view(1, x.size(0), 63 * 52)
                output_volume += [new_slice]
                new_slice = torch.movedim(new_slice, 1, 0)
                new_emb = torch.cat([self.slice_encoder(sample) for sample in new_slice]).view(new_slice.size(0), 1, 512)
                new_emb = torch.cat([sample + self.slice_pos_emb(torch.LongTensor([slice_idx])) for sample in new_emb]).view(new_emb.size(0), 1, 512)
                new_emb = torch.movedim(new_emb, 1, 0)
                slice_emb = torch.cat([slice_emb, new_emb])
                #print(slice_idx, torch.std(new_slice, dim=(0, 1))[:10])
        
            output_volume = torch.cat(output_volume)
            output_volume = torch.movedim(output_volume, 1, 0).view(x.size(0), 53, 63, 52)
            return output_volume

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_tgt = y[0].view(y[0].size(0), 53, 63 * 52)
        y_gt = torch.movedim(y_tgt, 1, 0)
        y_pred = self.forward(x[0], y_tgt)
        y_pred = torch.movedim(y_pred, 1, 0).view(y[0].size(0), 53, 63, 52)
        y_pred = torch.mul(y_pred, y[1])
        y_gt = torch.mul(y[0], y[1])

        loss = F.mse_loss(y_pred, y_gt, reduction='none')
        loss = torch.sum(loss, (1, 2, 3))
        loss = torch.div(loss, torch.sum(y[1], (1, 2, 3)))
        loss = torch.mean(loss)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_tgt = y[0].view(y[0].size(0), 53, 63 * 52)
        y_gt = torch.movedim(y_tgt, 1, 0)
        y_pred = self.forward(x[0], y_tgt)
        y_pred = torch.movedim(y_pred, 1, 0).view(y[0].size(0), 53, 63, 52)
        y_pred = torch.mul(y_pred, y[1])
        y_gt = torch.mul(y[0], y[1])

        loss = F.mse_loss(y_pred, y_gt, reduction='none')
        loss = torch.sum(loss, (1, 2, 3))
        loss = torch.div(loss, torch.sum(y[1], (1, 2, 3)))
        loss = torch.mean(loss)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10),
            'monitor': 'train_loss'
        }


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
