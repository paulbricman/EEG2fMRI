from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from torch.utils.data import random_split, ConcatDataset
from data import OddballDataset
from model import DenseModel, ConvolutionalModel, TransformerModel, TransformerEncoderModel

train_subj_sel = [e for e in range(1, 18) if e not in [2, 4]]
val_subj_sel = [2]

train_dataset = OddballDataset('../../OddballData', train_subj_sel)
val_dataset = OddballDataset('../../OddballData', val_subj_sel)

X_train = [e[0][0].flatten() for e in train_dataset]
y_train = [e[1][0].flatten() for e in train_dataset]
X_test = [e[0][0].flatten() for e in test_dataset]
y_test = [e[1][0].flatten() for e in test_dataset]

print('loaded data')

model = linear_model.Ridge()
model.fit(X_train, y_train)
print('trained model')
y_pred = model.predict(X_test)

print('Mean squared error:', mean_squared_error(y_test, y_pred))