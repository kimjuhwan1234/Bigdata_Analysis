from tqdm.notebook import tqdm

import time
import torch
import pandas as pd


class Transfer_Learning:
    def __init__(self, device):
        self.device = device

    def get_lr(self, opt):
        for param_group in opt.param_groups:
            return param_group['lr']

    def calculate_accuracy(self, output, gt):
        # 정확도 계산 (MAE)
        accuracy = 1.0 - torch.abs(output - gt).mean().item()
        return accuracy

    def eval_fn(self, model, dataset_dl):
        total_loss = 0.0
        total_accuracy = 0.0
        len_data = len(dataset_dl.dataset)

        model.eval()

        with torch.no_grad():
            for data, gt in tqdm(dataset_dl):
                data = data.to(self.device)
                gt = gt.to(self.device)

                output, loss = model(data, gt)
                total_loss += loss

                accuracy = self.calculate_accuracy(output, gt)
                total_accuracy += accuracy

            total_loss = total_loss / len_data
            total_accuracy = total_accuracy / len_data

        return total_loss, total_accuracy

    def train_fn(self, model, dataset_dl, opt):
        total_loss = 0.0
        total_accuracy = 0.0
        len_data = len(dataset_dl.dataset)

        model.train()
        for data, gt in tqdm(dataset_dl):
            data = data.to(self.device)
            gt = gt.to(self.device)

            opt.zero_grad()
            output, loss = model(data, gt)
            loss.backward()
            opt.step()

            total_loss += loss

            accuracy = self.calculate_accuracy(output, gt)
            total_accuracy += accuracy

        total_loss = total_loss / len_data
        total_accuracy = total_accuracy / len_data

        return total_loss, total_accuracy

    def train_and_eval(self, model, params):
        num_epochs = params['num_epochs']
        weight_path = params["weight_path"]

        train_dl = params["train_dl"]
        val_dl = params["val_dl"]

        opt = params["optimizer"]
        lr_scheduler = params["lr_scheduler"]

        loss_history = pd.DataFrame(columns=['train', 'val'])
        accuracy_history = pd.DataFrame(columns=['train', 'val'])

        best_loss = float('inf')
        start_time = time.time()

        for epoch in range(num_epochs):
            current_lr = self.get_lr(opt)
            print(f'Epoch {epoch + 1}/{num_epochs}, current lr={current_lr}')

            train_loss, train_accuracy = self.train_fn(model, train_dl, opt)

            loss_history.loc[epoch, 'train'] = train_loss
            accuracy_history.loc[epoch, 'train'] = train_accuracy

            val_loss, val_accuracy = self.eval_fn(model, val_dl)
            loss_history.loc[epoch, 'val'] = val_loss
            accuracy_history.loc[epoch, 'val'] = val_accuracy

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), weight_path)
                print('Saved model Weight!')

            lr_scheduler.step(val_loss)

            print(f'train loss: {train_loss:.2f}, val loss: {val_loss:.2f}')
            print(f'accuracy: {100 * val_accuracy:.2f} %, time: {(time.time() - start_time) / 60:.2f}')

            print('-' * 10)

        return model, loss_history, accuracy_history
