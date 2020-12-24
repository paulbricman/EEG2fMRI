import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from data import OddballDataset, fmri_preview
from model import ConvolutionalModel

sample = OddballDataset('../../OddballData')[0]
eeg_sample = sample[0].view(1, 1, 34, 30000)

trained_model = ConvolutionalModel()
y = trained_model(eeg_sample).detach().numpy()

print(y.shape)