import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from data import OddballDataset, fmri_preview
from model import DenseModel
import numpy as np

trained_model = DenseModel.load_from_checkpoint('./trained.ckpt')
trained_model.eval()

train_subj_sel = [e for e in range(1, 18) if e not in [2, 4]]
val_subj_sel = [2]

train_dataset = OddballDataset('../../OddballData', train_subj_sel)
val_dataset = OddballDataset('../../OddballData', val_subj_sel)

train_loader = DataLoader(train_dataset, num_workers=32, batch_size=8, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, num_workers=32, batch_size=8, shuffle=True, drop_last=True)

for i_batch, sample_batched in enumerate(val_loader):
	if i_batch == 0:
		pred = trained_model.forward(sample_batched[0])

		for i in range(1, 8):
			#fmri_preview(sample_batched[1][0][i].detach().numpy())			
			#fmri_preview(sample_batched[1][1][i].detach().numpy())
			fmri_preview((sample_batched[1][0][i] - sample_batched[1][0][0]).detach().numpy())
			fmri_preview(torch.mul(pred[i] - pred[0], sample_batched[1][1][i]).detach().numpy())
	break


