import pandas as pd

from preprocess import TitanicPreprocess
from config import PathConfig
from dataio import DataIOSteam
from model import MarketingDataModeling
from visualize_graph import VisualizeProcess

class MarketingcMain(TitanicPreprocess,PathConfig,DataIOSteam,MarketingDataModeling, VisualizeProcess):

    def __init__(self):
        TitanicPreprocess.__init__(self)
        PathConfig.__init__(self)
        MarketingDataModeling.__init__(self)
        DataIOSteam.__init__(self)

    def prepro_data1(self, f_name1, f_name2, **kwargs):

        #dataIO로 data1과 data2를 받아옴,,
        data1 = self.get_data(self.data_path, f_name1)
        data2 = self.get_data(self.data_path, f_name2)
        #preprocess화
        data = self.run_preprocessing(data1, data2)
        data1 = self.assign_1(data)
        return data1
        #data = self.run_preprocessing(data)

    def check_revuenue_by_channel(self, data):
        result = self.group_by_channel(data)
        return result

    def prepro_data2(self, f_name1, f_name2, **kwargs):

        #dataIO로 data1과 data2를 받아옴,,
        data1 = self.get_data(self.data_path, f_name1)
        data2 = self.get_data(self.data_path, f_name2)
        #preprocess화
        data = self.run_preprocessing(data1, data2)
        data2 = self.assign_2(data)
        return data2


    def model1(self,df):
        y_train, y_test, x_train, x_test, x_val, partial_x_train, y_val, partial_y_train = self.train_data_define(df)

        history, model = self.assign_model1(df,y_train, y_test, x_train, x_test, x_val, partial_x_train, y_val, partial_y_train )
        #data = self.run_preprocessing(data)
        return history,model

    def model2(self, df):
        y_train, y_test, x_train, x_test, x_val, partial_x_train, y_val, partial_y_train = self.train_data_define(df)

        history, model = self.assign_simpple_kerasModel(df, y_train, y_test, x_train, x_test, x_val, partial_x_train, y_val,
                                            partial_y_train)
        # data = self.run_preprocessing(data)
        return history, model

    def check_model(self, history, model):
        self.training_val_loss(history)
        self.training_acc(history)
        self.train_acc_loss(history)

    def prepro_data3(self,f_name1, f_name2, **kwargs):

        #dataIO로 data1과 data2를 받아옴,,
        data1 = self.get_data(self.data_path, f_name1)
        data2 = self.get_data(self.data_path, f_name2)
        #preprocess화
        data = self.run_preprocessing(data1, data2)
        data3 = self.assign_3(data)
        return data3