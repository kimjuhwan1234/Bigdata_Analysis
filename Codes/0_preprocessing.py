from sklearn.decomposition import PCA
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import os
import warnings
import numpy as np
import pandas as pd


def generate_PCA_data(data: pd.DataFrame):
    gt = pd.DataFrame(data['평균기온'], columns=['평균기온'])
    X = data.drop('평균기온', axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.99)
    pca.fit(X_scaled)
    print(pca.explained_variance_ratio_)
    X_pca = pca.transform(X_scaled)

    columns_pca = [f'PCA_{i}' for i in range(1, X_pca.shape[1] + 1)]
    X_pca_df = pd.DataFrame(X_pca, columns=columns_pca)
    X_pca_df.index = data.index

    df_combined = pd.concat([gt, X_pca_df], axis=1)
    return df_combined


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    input_dir = '../Database/'
    file = 'train.csv'
    data = pd.read_csv(os.path.join(input_dir, file), index_col=0)
    data.index = pd.DatetimeIndex(data.index)
    data['최고기온'] = data['최고기온'].interpolate(method='nearest')
    data['최저기온'] = data['최저기온'].interpolate(method='nearest')
    data['일교차'] = data['최고기온'] - data['최저기온']
    data['평균풍속'] = data['평균풍속'].interpolate(method='linear')
    data['일조합'] = data['일조합'].interpolate(method='linear')

    value1 = True
    if value1:
        train_linear = data[['일교차', '일사합']]
        train_linear.dropna(axis=0, inplace=True)

        X_train_linear = train_linear[['일교차']]
        y_train_linear = train_linear['일사합']

        model = LinearRegression()
        model.fit(X_train_linear, y_train_linear)

        missing1 = data[data['일사합'].isnull()][['일교차']]
        predicted_values1 = model.predict(missing1)
        data.loc[data['일사합'].isnull(), '일사합'] = predicted_values1

    value2 = True
    if value2:
        train_linear = data[['일조합', '일조율']]
        train_linear.dropna(axis=0, inplace=True)

        X_train_linear = train_linear['일조합'].values.reshape(-1, 1)
        y_train_linear = train_linear['일조율']

        model = LinearRegression()
        model.fit(X_train_linear, y_train_linear)

        missing2 = data[data['일조율'].isnull()]['일조합'].values.reshape(-1, 1)
        predicted_values2 = model.predict(missing2)
        data.loc[data['일조율'].isnull(), '일조율'] = predicted_values2

    value3 = True
    if value3:
        train_rainfall = data.dropna(axis=0)
        test_rainfall = data[data['강수량'].isnull()].drop('강수량', axis=1)

        X_train_rainfall = train_rainfall.drop(['강수량'], axis=1)
        y_train_rainfall = train_rainfall['강수량']

        cat_model = CatBoostRegressor(silent=True, iterations=300, depth=8, l2_leaf_reg=0.001)
        cat_model.fit(X_train_rainfall, y_train_rainfall)

        predicted_values_rainfall = cat_model.predict(test_rainfall)
        data.loc[data['강수량'].isnull(), '강수량'] = predicted_values_rainfall
        data['강수량'][data['강수량'] < 0] = 0

    pca = True
    if pca:
        PCA_data = generate_PCA_data(data)

        date_time = pd.DatetimeIndex(data.index)
        day_of_year = date_time.dayofyear

        PCA_data['Day sin'] = np.sin(2 * np.pi * day_of_year / 365)
        PCA_data['Day cos'] = np.cos(2 * np.pi * day_of_year / 365)
        PCA_data.to_csv('../Database/PCA_data.csv')

    if not pca:
        date_time = pd.DatetimeIndex(data.index)
        day_of_year = date_time.dayofyear

        data['Day sin'] = np.sin(2 * np.pi * day_of_year / 365)
        data['Day cos'] = np.cos(2 * np.pi * day_of_year / 365)
        data.dropna()
        data.to_csv('../Database/PCA_data.csv')
