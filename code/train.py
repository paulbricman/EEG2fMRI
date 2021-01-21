import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from data import OddballDataset
from model import TransformerModel

dataset = OddballDataset('../../OddballData')

train_prop, test_prop = 0.8, 0.1
train_size = int(train_prop * len(dataset))
test_size = int(test_prop * len(dataset))
val_size = len(dataset) - train_size - test_size

batch_size = 4
train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])
train_loader = DataLoader(train_dataset, num_workers=32, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, num_workers=32, batch_size=batch_size)

model = TransformerModel()
trainer = pl.Trainer(gpus=[2], max_epochs=50)
trainer.fit(model, train_loader, val_loader)
