import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from data import OddballDataset, fmri_preview
from model import ConvolutionalModel
import numpy as np

trained_model = ConvolutionalModel.load_from_checkpoint('./trained.ckpt')
trained_model.eval()

dataset = OddballDataset('../../OddballData')

train_prop, test_prop = 0.8, 0.1
train_size = int(train_prop * len(dataset))
test_size = int(test_prop * len(dataset))
val_size = len(dataset) - train_size - test_size

batch_size = 8
train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(0))
train_loader = DataLoader(train_dataset, num_workers=8, batch_size=batch_size, drop_last=True)
val_loader = DataLoader(val_dataset, num_workers=8, batch_size=batch_size, drop_last=True)

for i_batch, sample_batched in enumerate(val_loader):
	if i_batch == 0:
		pred = trained_model(sample_batched[0])

		for i in range(batch_size):
			fmri_preview(torch.mul(sample_batched[1][0][i], sample_batched[1][1][i]).detach().numpy())
			fmri_preview(torch.mul(pred[i], sample_batched[1][1][i]).detach().numpy())

	break

