import pandas as pd

class DataIOSteam:

    def get_data(self, path, f_name, flag=False):
        return pd.read_csv(f'{path}/{f_name}.csv')

    def get_X_y(self, data, x_list, y_value):
        X = data[data.columns[1:]]
        X = X[x_list]
        y = data[y_value]

        return X, y