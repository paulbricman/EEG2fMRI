import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from data import OddballDataset
from model import ConvolutionalModel, TransformerModel

train_subj_sel = [e for e in range(1, 18) if e not in [2, 4]]
val_subj_sel = [2]

train_dataset = OddballDataset('../../OddballData', train_subj_sel)
val_dataset = OddballDataset('../../OddballData', val_subj_sel)

train_loader = DataLoader(train_dataset, num_workers=32, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, num_workers=32, batch_size=32)

model = TransformerModel()
trainer = pl.Trainer(gpus=[3], accelerator='dp', max_epochs=1000, track_grad_norm=2, gradient_clip_val=0.1)
trainer.fit(model, train_loader, val_loader)
