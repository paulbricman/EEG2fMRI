import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from data import OddballDataset, fmri_preview
from model import TransformerModel
import numpy as np

trained_model = TransformerModel.load_from_checkpoint('./transformer.ckpt')
trained_model.eval()

train_subj_sel = [e for e in range(1, 18) if e not in [2, 4]]
val_subj_sel = [2]

train_dataset = OddballDataset('../../OddballData', train_subj_sel)
val_dataset = OddballDataset('../../OddballData', val_subj_sel)

merged_mask = torch.max(torch.cat([train_dataset[e][1][1] for e in range(0, 155 * 15, 155)]).view(15, 53, 63, 52), dim=0)[0]

train_loader = DataLoader(train_dataset, num_workers=32, batch_size=8, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, num_workers=32, batch_size=8, shuffle=True, drop_last=True)

for i_batch, sample_batched in enumerate(train_loader):
	if i_batch == 0:
		pred = trained_model.forward(sample_batched[0][0], sample_batched[1][0], mode='test')

		for i in range(1, 8):
			#fmri_preview(sample_batched[1][0][i].detach().numpy())			
			#fmri_preview(sample_batched[1][1][i].detach().numpy())
			fmri_preview((sample_batched[1][0][i]).detach().numpy(), '/home/s3908194/Pictures/' + str(i) + 'gt.png')
			fmri_preview(torch.mul(pred[i], sample_batched[1][1][i]).detach().numpy(), '/home/s3908194/Pictures/' + str(i) + 'pred.png')
	break


