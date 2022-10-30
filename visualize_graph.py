import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class VisualizeProcess:
    def __init__(self):
        pass

    def show_mean_gap(self, data1):
        print(data1.mean())

    def training_val_loss(self,history):
        history_dict = history.history
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        epochs = range(1, len(loss) + 1)

        plt.plot(epochs, loss, 'go', label='Training loss')  # go : green dot
        plt.plot(epochs, val_loss, 'g', label='Validation loss')  # g : green line
        plt.title('Training & Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def training_acc(self, history):
        plt.clf()  # 그래프 초기화하기
        history_dict = history.history
        acc = history_dict['accuracy']
        val_acc = history_dict['val_accuracy']
        loss = history_dict['loss']

        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, acc, 'go', label='Training acc')  # go : green dot
        plt.plot(epochs, val_acc, 'g', label='Validation acc')  # g : green line
        plt.title('Training & Validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()

    def train_acc_loss(self, history):
        fig, loss_ax = plt.subplots()

        acc_ax = loss_ax.twinx()

        loss_ax.set_ylim([0.0, 1.0])
        acc_ax.set_ylim([0.0, 1.0])

        loss_ax.plot(history.history['loss'], 'y', label='train loss')
        acc_ax.plot(history.history['accuracy'], 'b', label='train acc')

        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuray')

        loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='lower left')

        plt.show()