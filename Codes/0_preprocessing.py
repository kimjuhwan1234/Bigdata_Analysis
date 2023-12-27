import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression


def generate_PCA_data(data: pd.DataFrame):
    """
    :param data: momentum_data
    :return: Mom1+PCA_Data
    """
    gt = data.astype(float).loc[:, '평균기온']
    data_normalized = (data - data.mean()) / data.std()
    mat = data_normalized.astype(float)

    # 1. Searching optimal n_components
    n_components = min(len(data), 10)

    pca = PCA(n_components)
    pca.fit(mat)
    total_variance = np.sum(pca.explained_variance_ratio_)

    while total_variance > 0.99:
        n_components -= 1
        pca = PCA(n_components)
        pca.fit(mat)
        total_variance = np.sum(pca.explained_variance_ratio_)

    while total_variance < 0.99:
        n_components += 1
        pca = PCA(n_components)
        pca.fit(mat)
        total_variance = np.sum(pca.explained_variance_ratio_)

    # 2. PCA
    pca_mat = PCA(n_components=n_components).fit(data).transform(data)
    cols = [f'pca_component_{i + 1}' for i in range(pca_mat.shape[1])]
    mat_pd_pca = pd.DataFrame(pca_mat, columns=cols)
    mat_pd_pca_matrix = mat_pd_pca.values

    # 3. combined mom1 and PCA data
    first_column_matrix = np.array(gt).reshape(-1, 1)
    combined_matrix = np.hstack((first_column_matrix, mat_pd_pca_matrix))
    df_combined = pd.DataFrame(combined_matrix)
    df_combined.columns = df_combined.columns.astype(str)
    df_combined.index = data.index

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
    data['일교차'] = data['최고기온']-data['최저기온']
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

    value3 = False
    if value3:
        train_rainfall = data.dropna(axis=0)
        test_rainfall = data[data['강수량'].isnull()].drop('강수량', axis=1)

        X_train_rainfall = train_rainfall.drop(['강수량'], axis=1)
        y_train_rainfall = train_rainfall['강수량']

        cat_model = CatBoostRegressor(silent=True, iterations=300, depth=8, l2_leaf_reg=0.001)
        cat_model.fit(X_train_rainfall, y_train_rainfall)

        predicted_values_rainfall = cat_model.predict(test_rainfall)
        data.loc[data['강수량'].isnull(), '강수량'] = predicted_values_rainfall

    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['강수량'].values, marker='o', linestyle='-', color='b')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    PCA_data = generate_PCA_data(data)


    date_time = pd.DatetimeIndex(PCA_data.index)
    day_of_year = date_time.dayofyear

    PCA_data['Day sin'] = np.sin(2 * np.pi * day_of_year / 365)
    PCA_data['Day cos'] = np.cos(2 * np.pi * day_of_year / 365)

    PCA_data.to_csv('../Database/PCA_data.csv')
