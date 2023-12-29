from Transfer_Learning import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

        X_train = self.data.iloc[start_index:end_index, :]
        y_train = self.data.iloc[end_index:end_index + 358, 0]

        X_train_tensor = torch.Tensor(X_train.values)
        y_train_tensor = torch.Tensor(y_train.values)

        return X_train_tensor, y_train_tensor


class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StackedLSTM, self).__init__()

        # 첫 번째 LSTM 층
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=0.4, batch_first=True)

        # 두 번째 LSTM 층
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, dropout=0.2, batch_first=True)

        # 세 번째 LSTM 층
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, dropout=0.1, batch_first=True)

        # 출력을 위한 선형 레이어
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 첫 번째 LSTM 층
        out, _ = self.lstm1(x)

        # 두 번째 LSTM 층
        out, _ = self.lstm2(out)

        # 세 번째 LSTM 층
        out, _ = self.lstm3(out)

        # 출력을 위한 선형 레이어
        out = self.fc(out)

        out = out[:, :358, :]

        return out


class StackedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StackedGRU, self).__init__()

        # 첫 번째 GRU 층
        self.GRU1 = nn.GRU(input_size, hidden_size, num_layers=num_layers, dropout=0.4, batch_first=True)

        # 두 번째 GRU 층
        self.GRU2 = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, dropout=0.2, batch_first=True)

        # 세 번째 GRU 층
        self.GRU3 = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, dropout=0.1, batch_first=True)

        # 출력을 위한 선형 레이어
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 첫 번째 GRU 층
        out, _ = self.GRU1(x)

        # 두 번째 GRU 층
        out, _ = self.GRU2(out)

        # 세 번째 GRU 층
        out, _ = self.GRU3(out)

        # 출력을 위한 선형 레이어
        out = self.fc(out)

        return out


class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RegressionModel, self).__init__()

        self.backbone = StackedLSTM(input_size=input_size, hidden_size=hidden_size,
                                    num_layers=num_layers, output_size=output_size)

        # self.additional_layer = StackedGRU(input_size=output_size, hidden_size=hidden_size,
        #                                    num_layers=num_layers, output_size=output_size)

        # self.additional_layer = nn.Sequential(
        #     nn.Linear(358, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(512, 358),
        # )

    def forward(self, train, gt=None):
        output = self.backbone(train)
        # output = self.additional_layer(output)
        output = output.squeeze()

        if gt != None:
            loss = F.l1_loss(output, gt)
            return output, loss

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

        dataloaders = {'train': DataLoader(train_dataset, batch_size=32, shuffle=False),
                       'val': DataLoader(val_dataset, batch_size=32, shuffle=False),
                       'test': DataLoader(test_dataset, batch_size=32, shuffle=False),
                       }

        model = RegressionModel(10, 128, 2, 1)
        model.to(device)

        # weight_path = '../Weight/LSTM_GRU.pth'
        # model.load_state_dict(torch.load(weight_path))

        print('Weights are loaded!')
        print('Finished loading data!')

    train = True
    if train:
        print('Training model...')
        TL = Transfer_Learning(device)
        num_epochs = 10
        weight_path = '../Weight/LSTM.pth'
        opt = optim.Adam(model.parameters(), lr=0.00002)
        lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.2, patience=3)

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
            print('Check loss and MAE...')
            error = 0
            i = 0

            model.eval()
            with ((torch.no_grad())):
                for data, gt in dataloaders['val']:
                    i += 1
                    TL.plot_bar('Val', i, len(dataloaders['val']))
                    data = data.to(device)
                    gt = gt.to(device)
                    output = model(data)

                    MAE = TL.calculate_accuracy(output, gt)
                    error += MAE

                error = error / len(dataloaders['val'])

            print(f'Total MAE about Valset {error:.4f}')

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
            plt.title("Train-Val MAE")
            plt.plot(range(1, num_epochs + 1), metric_hist_numpy.iloc[:, 0], label="train")
            plt.plot(range(1, num_epochs + 1), metric_hist_numpy.iloc[:, 1], label="val")
            plt.ylabel("MAE")
            plt.xlabel("Training Epochs")
            plt.legend()
            plt.show()
            print('Finished checking loss and MAE!')

    if not train:
        print('Skipped train...')
        model.load_state_dict(torch.load(weight_path))
        print('Weights are loaded!')

    Prediction = False
    if Prediction:
        print('Try prediction...')

        model.to(device)
        to_predict = data.iloc[-1790:, :]
        to_predict = torch.Tensor(to_predict.values)

        model.eval()
        with torch.no_grad():
            predicted = model(to_predict)

        pred = pd.read_csv('../Database/sample_submission.csv')
        pred['평균기온'] = predicted

        pred.to_csv('../Files/LSTM.csv', index=False)
        print("Finished saving Prediction!")

    if not Prediction:
        print('Evaluation in progress for testset...')

        all_predictions = []
        all_gt = []
        i = 0

        model.eval()
        with ((torch.no_grad())):
            for data, gt in dataloaders['test']:
                i += 1
                TL.plot_bar('Test', i, len(dataloaders['test']))
                data = data.to(device)
                gt = gt.to(device)
                output = model(data)

                all_predictions.extend(output.tolist())
                all_gt.extend(gt.tolist())

        all_mse = []
        all_mae = []
        all_r2 = []

        for i in range(len(all_predictions)):
            predictions = all_predictions[i]
            ground_truth = all_gt[i]

            mse = mean_squared_error(ground_truth, predictions)
            mae = mean_absolute_error(ground_truth, predictions)
            r2 = r2_score(ground_truth, predictions)
            all_mse.append(mse)
            all_mae.append(mae)
            all_r2.append(r2)

        print(f'Mean Squared Error (MSE): {np.mean(all_mse):.4f}')
        print(f'Mean Absolute Error (MAE): {np.mean(all_mae):.4f}')
        print(f'R-squared (R2): {np.mean(all_r2):.4f}')
        print("Finished evaluation!")
