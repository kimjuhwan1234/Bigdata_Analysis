from Transfer_Learning import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.window_size = 358 * 10

    def __len__(self):
        return len(self.data) - self.window_size - 357

    def __getitem__(self, index):
        start_index = index
        end_index = index + self.window_size

        # 만약 end_index가 데이터셋의 길이보다 크거나 같으면 데이터셋 생성 종료
        if end_index >= len(self.data) - self.window_size - 357:
            raise IndexError("Index out of range, dataset creation complete.")

        X_train = self.data.iloc[start_index:end_index]
        y_train = self.data.iloc[end_index:end_index + 358, 0]

        X_train_tensor = torch.Tensor(X_train.values)
        y_train_tensor = torch.Tensor(y_train.values)

        return X_train_tensor, y_train_tensor


# LSTM을 여러 번 쌓아서 사용할 모델 클래스 정의
class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StackedLSTM, self).__init__()

        # 첫 번째 LSTM 층
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)

        # 두 번째 LSTM 층
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)

        # 출력을 위한 선형 레이어
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 첫 번째 LSTM 층
        out, _ = self.lstm1(x)

        # 두 번째 LSTM 층
        out, _ = self.lstm2(out)

        # 마지막 시간 단계의 출력을 사용하여 선형 레이어 통과
        out = self.fc(out[:, -1, :])

        return out


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()

        self.backbone = StackedLSTM(input_size=10, hidden_size=128, num_layers=2, output_size=358)

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

        train_data, val_test_data = train_test_split(data, test_size=(val_ratio + test_ratio), shuffle=False)
        val_test_size = len(val_test_data)
        val_data, test_data = train_test_split(val_test_data, test_size=test_ratio / (val_ratio + test_ratio),
                                               shuffle=False)

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

        num_epochs = 100
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
            print('Check loss and accuracy...')
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
            print('Finished check loss and accuracy!')

    if not train:
        print('Skipped train...')
        model.load_state_dict(torch.load(weight_path))
        print('Weights are loaded!')

    # Prediction = True
    # if Prediction:
    #     print('Try prediction...')
    #     model.to(device)
    #     to_predict =
    #
    #     eval_transform = get_transform()
    #     test_image = eval_transform(test_image).unsqueeze(0)
    #
    #     model.eval()
    #     with torch.no_grad():
    #         predicted_mask = model(test_image)
    #         predicted_mask = torch.sigmoid(predicted_mask)
    #
    #
    #     output_image.save("../Database/output.png")
    #     print("Masking and saving complete!")
    #
    # if not Prediction:
    #     print('Evaluation in progress for testset...')
    #     image, mask = test_dataset[9]
    #     logits_mask = model(image.to(device).unsqueeze(0))  # (c,h,w) -> (b,c,h,w)
    #     logits_mask = torch.sigmoid(logits_mask)
    #     pred_mask = (logits_mask > 0.5) * 1.0
    #     pred = pred_mask.squeeze(0)
    #
    #     image_pil = ToPILImage()(image.cpu())
    #     true_mask_pil = ToPILImage()(mask.cpu())
    #     predicted_mask_pil = ToPILImage()(pred.cpu())
    #
    #     fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    #
    #     axs[0].imshow(image_pil)
    #     axs[0].set_title('Image')
    #
    #     axs[1].imshow(true_mask_pil)
    #     axs[1].set_title('True Mask')
    #
    #     axs[2].imshow(predicted_mask_pil)
    #     axs[2].set_title('Predicted Mask')
    #
    #     plt.show()