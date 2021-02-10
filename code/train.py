import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from data import OddballDataset
from model import ConvolutionalModel

dataset = OddballDataset('../../OddballData')

train_prop, test_prop = 0.8, 0.1
train_size = int(train_prop * len(dataset))
test_size = int(test_prop * len(dataset))
val_size = len(dataset) - train_size - test_size

batch_size = 128
train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(0))
train_loader = DataLoader(train_dataset, num_workers=32, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, num_workers=32, batch_size=batch_size, drop_last=True)

model = ConvolutionalModel()
trainer = pl.Trainer(gpus=4, accelerator='dp', max_epochs=1, track_grad_norm=2, check_val_every_n_epoch=5, gradient_clip_val=0.1)
trainer.fit(model, train_loader, val_loader)
