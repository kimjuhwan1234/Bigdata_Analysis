from gluonts.torch.model.deepar import DeepARModel

import os
import pandas as pd
import torch.nn as nn

import torch.nn.functional as F


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()

        self.backbone = DeepARModel()

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

    input_dir = '../Database/'
    file = 'train.csv'
    data = pd.read_csv(os.path.join(input_dir, file), index_col=0)
    data.index = pd.DatetimeIndex(data.index)


