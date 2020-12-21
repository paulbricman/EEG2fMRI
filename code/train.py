import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data import OddballDataset
from model import ConvolutionalModel

dataset = OddballDataset('../../OddballData')

train_loader = DataLoader(dataset)

model = ConvolutionalModel()

trainer = pl.Trainer()
trainer.fit(model, train_loader)
