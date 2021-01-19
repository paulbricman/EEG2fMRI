import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from data import OddballDataset, fmri_preview
from model import ConvolutionalModel, TransformerModel

def reconstruct(volume):
	volume = (volume + 1) / 2 * 3.76 +1.4
	volume = 10 ** volume
	return volume

trained_model = TransformerModel.load_from_checkpoint('./trained.ckpt')

def compare(idx):
	sample = OddballDataset('../../OddballData')[600]
	eeg_sample = sample[0].view(1, 1, 34, 30000)
	pred = trained_model(eeg_sample).detach().numpy()
	pred = reconstruct(pred)

	gt = reconstruct(sample[1])

	fmri_preview(gt)
	fmri_preview(pred[0])

compare(0)
compare(600)
