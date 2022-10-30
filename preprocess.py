import numpy as np
import pandas as pd
from copy import copy
from numpy import nan

class TitanicPreprocess:
    def __init__(self):
        pass

    def run_preprocessing(self, data1, data2):
        data = self._merge_data(data1,data2)
        data = self._set_fill_na(data)
        return data

    def assign_1(self,data):
        c_list= self.delete_column_assign1()
        removed_column = self.remove_columns(data,c_list)
        return removed_column;

    def assign_2(self, data):
        c_list = self.delete_column_assign2()
        removed_column = self.remove_columns(data, c_list)
        df=self.set_missing_value(removed_column)
        df=self.set_initial(df)
        return df;

    def assign_3(self,data):
        c_list = self.delete_column_assign3()
        removed_column = self.remove_columns(data, c_list)
        df = self.set_missing_value(removed_column)
        df = self.set_initial(df)
        return df;

    def _merge_data(self, data1, data2):
        data = pd.merge(data1, data2, on = 'user_uuid', how='outer',indicator = True)
        #print(data.isnull().sum())
        return data

    def _set_fill_na(self, data):
        data['visits'] = data['visits'].fillna(0)
        data['revenue'] = data['revenue'].fillna(0)
        return data

    def delete_column_assign1(self):
        columns_list = ['visits']
        return columns_list

    def delete_column_assign2(self):
        columns_list = ['date_joined','visits','revenue','user_uuid','_merge']
        return columns_list

    def delete_column_assign3(self):
        columns_list = ['sex','age_group','os','revenue']
        return columns_list

    def delete_column_assign3(self):
        columns_list = ['date_joined','visits','revenue','user_uuid','_merge']
        return columns_list

    def remove_columns(self, DF, remove_list):
        result = copy(DF)
        for column in remove_list:
            del(result[column])
        return result

    def set_missing_value(self,df):
        values = ["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"]
        df = self.check_column(df, "age_group", values)
        df = df.dropna(axis=0)
        print(df.isnull().sum())
        return df

    #age값이 -99있음,,
    def check_column(self, DF, column, values):
        result = copy(DF)
        temp = []
        cnt = 0
        for x in result[column]:

            if x not in values:
                temp.append(nan)
                cnt = cnt + 1
            else:
                temp.append(x)
        result[column] = temp
        return result

    #문자데이터 변환 및 원핫인코딩
    def set_initial(self, df):
        # y값 처리
        df['marketing_channel'] = np.where(df['marketing_channel'].to_numpy() == "channel_A", 0, 1)
        # x값 처리
        df['sex'] = np.where(df['sex'].to_numpy() == "male", 0, 1)
        df['os'] = np.where(df['os'].to_numpy() == "android", 0, 1)
        df['age_group'] = df['age_group'].apply(self.age_code)
        return df

    def age_code(self,x):
        if x == "10-19":
            return 0
        elif x == "20-29":
            return 1
        elif x == "30-39":
            return 2
        elif x == "40-49":
            return 3
        elif x == "50-59":
            return 4
        elif x == "60-69":
            return 5
        else:
            return 6

