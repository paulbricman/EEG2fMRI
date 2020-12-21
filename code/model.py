import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F


class ConvolutionalModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, (1, 5), stride=16),
            nn.AvgPool2d((1, 64)),
            nn.Flatten(),
            nn.Linear(696, 32 * 64 * 64)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), 1, 34, 30000)
        y_pred = self.model(x)
        y_pred = y_pred.view(y_pred.size(0), 64, 64, 32)
        loss = F.mse_loss(y, y_pred)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)