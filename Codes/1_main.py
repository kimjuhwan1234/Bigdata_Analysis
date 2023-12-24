from Transfer_Learning import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.nn as nn
import tensorflow as tf
import torch.optim as optim
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.window_size = 358 * 3

    def __len__(self):
        return len(self.data) - self.window_size - 357

    def __getitem__(self, index):
        start_index = index
        end_index = index + self.window_size

        X_train = self.data.iloc[start_index:end_index]
        y_train = self.data.iloc[end_index:end_index + 358, 0]

        X_train_tensor = torch.Tensor(X_train.values)
        y_train_tensor = torch.Tensor(y_train.values)

        return X_train_tensor, y_train_tensor


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()

        self.backbone = nn.Sequential(
            nn.LSTM(input_size=10, hidden_size=128, num_layers=1, batch_first=True, return_sequences=True),
            nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True, return_sequences=True),
            nn.LSTM(input_size=128, hidden_size=10, num_layers=1, batch_first=True, return_sequences=True)
        )

        in_features = self.backbone.hidden_size * self.backbone.num_layers

        self.additional_layer = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1, 358),
        )

    def forward(self, train, gt=None):

        backbone_output = self.backbone(train)
        final_layer_output = backbone_output[-1]
        output = self.additional_layer(final_layer_output.unsqueeze(1))

        if gt != None:
            return output, nn.SmoothL1Loss(output, gt) + nn.MSELoss(output, gt)

        return output


if __name__ == "__main__":
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    data = pd.read_csv('../Database/PCA_data.csv')
    data.pop('일시')
    data.astype(float)


    dataload = True
    if dataload:
        print('Loading data...')
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1

        train_data, val_test_data = train_test_split(data, test_size=(val_ratio + test_ratio), shuffle=True)
        val_test_size = len(val_test_data)
        val_data, test_data = train_test_split(val_test_data, test_size=test_ratio/(val_ratio+test_ratio), shuffle=True)

        # train_data.reset_index(drop=True, inplace=True)
        # val_test_data.reset_index(drop=True, inplace=True)
        # test_data.reset_index(drop=True, inplace=True)

        train_dataset = CustomDataset(train_data)
        val_dataset = CustomDataset(val_test_data)
        test_dataset = CustomDataset(test_data)

        dataloaders = {'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
                       'val': DataLoader(val_dataset, batch_size=32, shuffle=False),
                       'test': DataLoader(test_dataset, batch_size=32, shuffle=False),
                       }

        model = RegressionModel()
        model.to(device)
        print('Finished loading data!')


    train = True
    if train:
        print('Training model...')
        TL = Transfer_Learning(device)

        num_epochs = 5
        weight_path = '../Weight/LSTM.pth'
        opt = optim.Adam(model.parameters(), lr=0.01)
        lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5)

        parameters = {
            'num_epochs': num_epochs,
            'weight_path': weight_path,

            'train_dl': dataloaders['train'],
            'val_dl': dataloaders['val'],

            'optimizer': opt,
            'lr_scheduler': lr_scheduler,
        }

        model, loss_hist, metric_hist = TL.train_and_eval(model, parameters)
        print('Finished training model!')

        accuracy_check = True
        if accuracy_check:
            print('Check stats...')
            correct = 0
            total = 0

            with torch.no_grad():
                for data in dataloaders['val']:
                    images, mask = data[0].to(device), data[1].to(device)
                    output = model(images)

                    pred_mask = (output > 0.5).float()
                    accuracy = (pred_mask == mask).sum().item()
                    total_pixels = mask.numel()
                    accuracy = accuracy / total_pixels

            print(f'Accuracy {100 * accuracy:.2f} %')

            loss_hist_numpy = loss_hist.applymap(
                lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x)
            metric_hist_numpy = metric_hist.applymap(
                lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x)

            # plot loss progress
            plt.title("Train-Val Loss")
            plt.plot(range(1, num_epochs + 1), loss_hist_numpy.iloc[:, 0], label="train")
            plt.plot(range(1, num_epochs + 1), loss_hist_numpy.iloc[:, 1], label="val")
            plt.ylabel("Loss")
            plt.xlabel("Training Epochs")
            plt.legend()
            plt.show()

            # plot accuracy progress
            plt.title("Train-Val Accuracy")
            plt.plot(range(1, num_epochs + 1), metric_hist_numpy.iloc[:, 0], label="train")
            plt.plot(range(1, num_epochs + 1), metric_hist_numpy.iloc[:, 1], label="val")
            plt.ylabel("Accuracy")
            plt.xlabel("Training Epochs")
            plt.legend()
            plt.show()
            print('Finished stats check!')





