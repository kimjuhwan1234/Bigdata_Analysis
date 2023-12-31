from Transfer_Learning import *

import os

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    input_dir = '../Database/'
    file = 'sample_submission.csv'
    data = pd.read_csv(os.path.join(input_dir, file), encoding='ANSI', index_col=0)
    data.index = pd.DatetimeIndex(data.index)

    input_dir2 = '../Files/'
    file2 = 'bi_LSTM.csv'
    sol = pd.read_csv(os.path.join(input_dir2, file2), encoding='ANSI', index_col=0)
    sol.index = pd.DatetimeIndex(sol.index)

    data['평균기온'] = sol['avg_Temperature']

    data.to_csv(os.path.join(input_dir2, file), encoding='utf-8')
