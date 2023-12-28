from Transfer_Learning import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from gluonts.mx import DeepAREstimator, Trainer

import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.window_size = 358 * 5

    def __len__(self):
        return len(self.data) - self.window_size - 358

    def __getitem__(self, index):
        start_index = index
        end_index = index + self.window_size

        if end_index + 358 > len(self.data):
            raise IndexError("Index out of bounds. Reached the end of the dataset.")

        X_train = self.data.iloc[start_index:end_index]
        y_train = self.data.iloc[end_index:end_index + 358, 0]

        X_train_tensor = torch.Tensor(X_train.values)
        y_train_tensor = torch.Tensor(y_train.values)

        return X_train_tensor, y_train_tensor


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()

        self.additional_layer = nn.Sequential(
            nn.Linear(358, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 358),
        )

    def forward(self, train, gt=None):
        output = self.backbone(train)
        # output = self.additional_layer(output)
        loss = F.smooth_l1_loss(output, gt) + F.mse_loss(output, gt)

        if gt != None:
            return output, loss

        return output


if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    # input_dir = '../Database/'
    # file = 'train.csv'
    # data = pd.read_csv(os.path.join(input_dir, file), index_col=0)
    # data.index = pd.DatetimeIndex(data.index)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    data = pd.read_csv('../Database/PCA_data.csv')
    data.pop('일시')
    data.astype(float)

    dataload = True
    if dataload:
        print('Loading data...')
        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1

        train_data, val_test_data = train_test_split(data, test_size=(val_ratio + test_ratio), shuffle=False)
        val_data, test_data = train_test_split(val_test_data, test_size=test_ratio / (val_ratio + test_ratio),
                                               shuffle=False)

        # train_data.reset_index(drop=True, inplace=True)
        # val_test_data.reset_index(drop=True, inplace=True)
        # test_data.reset_index(drop=True, inplace=True)

        train_dataset = CustomDataset(train_data)
        val_dataset = CustomDataset(val_test_data)
        test_dataset = CustomDataset(test_data)

        dataloaders = {'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
                       'val': DataLoader(val_dataset, batch_size=32, shuffle=True),
                       'test': DataLoader(test_dataset, batch_size=32, shuffle=False),
                       }

        model = RegressionModel()
        model.to(device)
        print('Finished loading data!')

    DeepAR = True
    if DeepAR:
        print('Loading')

        estimator = DeepAREstimator(fred='D', prediction_length=358, num_layers=3,
                                    trainer_kwargs={'accelerator': 'cuda', 'max_epochs': 30})

        predictor = estimator.train(dataloaders['train'], num_workers=2)

        pred = list(predictor.predict(dataloaders['train']))
