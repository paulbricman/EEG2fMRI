import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from data import OddballDataset
from model import ConvolutionalModel

train_subj_sel = [e for e in range(1, 18) if e not in [4, 16, 17]]
val_subj_sel = [16, 17]

train_dataset = OddballDataset('../../OddballData', train_subj_sel)
val_dataset = OddballDataset('../../OddballData', val_subj_sel)

train_loader = DataLoader(train_dataset, num_workers=32, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, num_workers=32, batch_size=64)

model = ConvolutionalModel()
trainer = pl.Trainer(gpus=[2, 3], accelerator='dp', max_epochs=1000, track_grad_norm=2)
trainer.fit(model, train_loader, val_loader)
