from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Transfer_Learning import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import arch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import statsmodels.api as sm
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
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional):
        super(StackedLSTM, self).__init__()

        if bidirectional:
            hidden_size2 = hidden_size * 2

        if not bidirectional:
            hidden_size2 = hidden_size

        # 첫 번째 LSTM 층
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                             bidirectional=bidirectional, dropout=0.2, batch_first=True)

        # 두 번째 LSTM 층
        self.lstm2 = nn.LSTM(hidden_size2, hidden_size, num_layers=num_layers,
                             bidirectional=bidirectional, dropout=0.2, batch_first=True)

        # 세 번째 LSTM 층
        self.lstm3 = nn.LSTM(hidden_size2, hidden_size, num_layers=num_layers,
                             bidirectional=bidirectional, dropout=0.1, batch_first=True)

        # 출력을 위한 선형 레이어
        self.fc = nn.Linear(hidden_size2, output_size)

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
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional):
        super(StackedGRU, self).__init__()

        if bidirectional:
            hidden_size2 = hidden_size * 2

        if not bidirectional:
            hidden_size2 = hidden_size

        # 첫 번째 GRU 층
        self.GRU1 = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                           bidirectional=bidirectional, dropout=0.4, batch_first=True)

        # 두 번째 GRU 층
        self.GRU2 = nn.GRU(hidden_size2, hidden_size, num_layers=num_layers,
                           bidirectional=bidirectional, dropout=0.3, batch_first=True)

        # 세 번째 GRU 층
        self.GRU3 = nn.GRU(hidden_size2, hidden_size, num_layers=num_layers,
                           bidirectional=bidirectional, dropout=0.1, batch_first=True)

        # 출력을 위한 선형 레이어
        self.fc = nn.Linear(hidden_size2, output_size)

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
    def __init__(self, input_size, hidden_size, num_layers, output_size, additional, bidirectional):
        super(RegressionModel, self).__init__()

        self.additional = additional

        self.backbone = StackedLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                    output_size=output_size, bidirectional=bidirectional)

        if additional:
            self.additional_layer = StackedGRU(input_size=output_size, hidden_size=hidden_size, num_layers=num_layers,
                                               output_size=output_size, bidirectional=bidirectional)

    def forward(self, train, gt=None):
        output = self.backbone(train)

        if self.additional:
            output = self.additional_layer(output)

        output = output.squeeze()

        if gt != None:
            loss = F.l1_loss(output, gt)
            return output, loss

        return output

class Adjustment:
    def __init__(self, test_input, pred):
        self.test_input = test_input
        self.pred = pred

    def generalized_wiener_process(self, a, b):
        ad = 1 / 4
        forecast = np.random.normal(a, (b ** 2) * ad, 30)
        return forecast

    def adjust_GARCH(self):
        SARIMA_model = sm.tsa.statespace.SARIMAX(endog=self.test_input, order=(1, 0, 1), trend='n').fit()
        GARCH = arch.arch_model(SARIMA_model.resid, vol='Garch', p=1, q=1)
        GARCH_model = GARCH.fit(disp='off', show_warning=False)
        conditional_var = GARCH_model.conditional_volatility[:-2]
        error = np.random.randn(358) * np.sqrt(conditional_var)
        self.pred['GARCH_Temperature'] = self.pred['avg_Temperature'].values + error.values

    def adjust_Wiener_Process(self, a):
        monthly_temperature = np.split(self.test_input.values, 12)
        forecasted_var = []
        for i in range(len(monthly_temperature)):
            b = np.std(monthly_temperature[i])
            forecasted_var.append(self.generalized_wiener_process(a, b))
        forecasted_var = pd.DataFrame(np.abs(np.concatenate(forecasted_var))).iloc[:358]
        error = np.random.randn(358) * np.sqrt(forecasted_var[0])
        self.pred['Wiener_Temperature'] = self.pred['avg_Temperature'].values + error.values


if __name__ == "__main__":
    device = torch.device('CUDA') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    train = pd.read_csv('../Database/PCA_data.csv')
    test_input = train.iloc[-365:-5, 1]
    test_input.index = train['일시'].iloc[-365:-5]
    train.pop('일시')

    scaler= StandardScaler()
    train['평균기온'] = scaler.fit_transform(train['평균기온'].values.reshape(-1, 1))

    dataload = True
    if dataload:
        print('Loading data...')
        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1

        train_data, val_test_data = train_test_split(train, test_size=(val_ratio + test_ratio), shuffle=False)
        val_data, test_data = train_test_split(val_test_data, test_size=test_ratio / (val_ratio + test_ratio),
                                               shuffle=False)

        train_dataset = CustomDataset(train_data)
        val_dataset = CustomDataset(val_test_data)
        test_dataset = CustomDataset(test_data)

        dataloaders = {'train': DataLoader(train_dataset, batch_size=32, shuffle=False),
                       'val': DataLoader(val_dataset, batch_size=32, shuffle=True),
                       'test': DataLoader(test_dataset, batch_size=32, shuffle=True),
                       }
        TL = Transfer_Learning(device)
        print('Finished loading data!')

    bidirectional = True
    if bidirectional:
        print('Building model...')
        additional = False
        model = RegressionModel(10, 64, 2, 1, additional, bidirectional)
        model.to(device)

        if additional:
            weight_path = '../Weight/bi_LSTM_GRU.pth'

        if not additional:
            weight_path = '../Weight/bi_LSTM.pth'

        model.load_state_dict(torch.load(weight_path))

        print('Weights are loaded!')

    if not bidirectional:
        print('Building model...')
        additional = False
        model = RegressionModel(9, 128, 2, 1, additional, bidirectional)
        model.to(device)

        if additional:
            weight_path = '../Weight/LSTM_GRU.pth'

        if not additional:
            weight_path = '../Weight/LSTM.pth'

        model.load_state_dict(torch.load(weight_path))

        print('Weights are loaded!')

    train_eval = False
    if train_eval:
        print('Training model...')

        num_epochs = 50
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
            print('Check loss and adjusted_R_square...')
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

                    R_square = TL.calculate_adjusted_r2_score(output, gt, data.shape[1], data.shape[2])
                    error += R_square

                error = error / len(dataloaders['val'])

            print(' ')
            print(f'Total adjusted_R_square about Valset {error:.4f}')

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
            plt.title("Train-Val adjusted_R_square")
            plt.plot(range(1, num_epochs + 1), metric_hist_numpy.iloc[:, 0], label="train")
            plt.plot(range(1, num_epochs + 1), metric_hist_numpy.iloc[:, 1], label="val")
            plt.ylabel("adjusted_R_square")
            plt.xlabel("Training Epochs")
            plt.legend()
            plt.show()
            print('Finished checking loss and adjusted R_square!')

    Evaluation = False
    if Evaluation:
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

        print(' ')
        print(f'Model: {weight_path}')
        print(f'Mean Squared Error (MSE): {np.mean(all_mse):.4f}')
        print(f'Mean Absolute Error (MAE): {np.mean(all_mae):.4f}')
        print(f'R-squared (R2): {np.mean(all_r2):.4f}')
        print("Finished evaluation!")

    if not Evaluation:
        print('Try prediction...')

        model.to(device)
        to_predict = train.iloc[-1790:, :]
        to_predict = torch.Tensor(to_predict.values).to(device)
        to_predict = to_predict.unsqueeze(0)

        model.eval()
        with torch.no_grad():
            predicted = model(to_predict)

        predicted = predicted.cpu().detach().numpy()
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1))


        pred = pd.read_csv('../Database/sample_submission.csv', encoding='ANSI')
        pred['평균기온'] = predicted
        pred.columns = ['Date', 'avg_Temperature']

        print('Finished prediction!')

        adjusted_temp = True
        if adjusted_temp:
            print('Adjusting prediction...')
            Ad = Adjustment(test_input, pred)
            Ad.adjust_GARCH()
            Ad.adjust_Wiener_Process(0)

            Ad.pred.to_csv(f'../Files/{weight_path[10:-4]}.csv', encoding='ANSI', index=False)
            print("Finished saving adjusted_Prediction!")
