import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from torch.utils.data import random_split, ConcatDataset
from data import OddballDataset
from model import DenseModel, ConvolutionalModel, TransformerModel, TransformerEncoderModel

train_subj_sel = [e for e in range(1, 18) if e not in [2, 4]]
val_subj_sel = [2]

train_dataset = OddballDataset('../../OddballData', train_subj_sel)
val_dataset = OddballDataset('../../OddballData', val_subj_sel)
train_dataset = ConcatDataset([train_dataset, [val_dataset[0]]])
batch_size = 32
train_loader = DataLoader(train_dataset, num_workers=32, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, num_workers=32, batch_size=batch_size, drop_last=True)

model = TransformerModel()
trainer = pl.Trainer(gpus=[3], accelerator='dp', max_epochs=200, track_grad_norm=2)
trainer.fit(model, train_loader, val_loader)
