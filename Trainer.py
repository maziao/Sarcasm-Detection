import time
import datetime
import numpy as np
from DataSet import DataSet
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, config, network, datamanager):
        self.start_time = config.start_time
        self.end_time = None
        self.network = network.to(config.device)
        self.train_dataset = DataSet(datamanager, mode='train')
        self.test_dataset = DataSet(datamanager, mode='test')
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2
        )
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2
        )
        self.config = config
        self.train_acc_list = []
        self.test_acc_list = []
        self.loss_list = []

    def train(self):
        print('Epoc\tTrain\tTest\tLoss')

        for i in range(self.config.epoch_num):
            loss = 0
            for j, data in enumerate(self.train_dataloader):
                data[0] = data[0].to(self.config.device)
                data[2] = data[2].to(self.config.device)
                loss += self.network.step(data)
                print('\rEpoch', i + 1, 'completed:', round(j * 100 / len(self.train_dataloader)), '%', end='')
            loss = loss / len(self.train_dataloader)
            self.loss_list.append(loss)

            print('\r{:<8d}'.format(i + 1), end='')
            self.test(mode='train')
            self.test(mode='test')
            print('{:<8.3f}'.format(loss), '                                       ')
        self.end_time = time.time()
        print('==================================================')
        print('Time consumed:', round(self.end_time - self.start_time, 1), 's')

    def test(self, mode):
        self.network.eval()
        if mode == 'train':
            dataloader = self.train_dataloader
        else:
            dataloader = self.test_dataloader

        correct_num = 0
        predicted_num = 0
        for i, data in enumerate(dataloader):
            data[0] = data[0].to(self.config.device)
            data[2] = data[2].to(self.config.device)
            c, p = self.network.evaluate(data)
            correct_num += c
            predicted_num += p
        acc = correct_num / predicted_num

        if mode == 'train':
            self.train_acc_list.append(acc)
        else:
            self.test_acc_list.append(acc)
        print('{:<8.1f}'.format(acc * 100), end='')

        self.network.train()

    def save_training_log(self):
        model = self.network.__class__.__name__
        t = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = model + '_' + t + '_.txt'
        path = '../TrainingLog/' + filename
        file = open(path, 'w')
        training_info = self.config.get_training_info()
        for info in training_info:
            file.write(info + '\n')
        file.write('==============================\n')
        file.write('Epoc    Train   Test    Loss\n')
        for i in range(self.config.epoch_num):
            line = '{:<8d}'.format(i + 1) + '{:<8.1f}'.format(self.train_acc_list[i] * 100) + \
                   '{:<8.1f}'.format(self.test_acc_list[i] * 100) + '{:<8.1f}'.format(self.loss_list[i] * 100) + '\n'
            file.write(line)
        file.write('==============================\n')
        best_acc = np.max(self.test_acc_list)
        line = 'Best accuracy: ' + str(round(best_acc * 100, 1)) + '%\n'
        file.write(line)
        line = 'Time consumed: ' + str(round(self.end_time - self.start_time, 1)) + 's\n'
        file.write(line)
        file.close()

    def visualize(self):
        model = self.network.__class__.__name__
        t = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        acc_path = '../Figure/' + model + '_' + t + '_acc.jpg'
        loss_path = '../Figure/' + model + '_' + t + '_loss.jpg'

        x = np.array(range(1, self.config.epoch_num + 1))
        y1 = np.array(self.train_acc_list)
        y2 = np.array(self.test_acc_list)
        y3 = np.array(self.loss_list)
        y1 = y1 * 100
        y2 = y2 * 100

        plt.figure()
        plt.xlabel('Epoch', fontdict={'family': 'Times New Roman', 'size': 14})
        plt.ylabel('Accuracy/%', fontdict={'family': 'Times New Roman', 'size': 14})
        plt.plot(x, y1, 'dodgerblue', linewidth=1.0, label='Train')
        plt.plot(x, y2, 'red', linewidth=1.0, label='Test')
        plt.xticks(fontproperties='Times New Roman', size=10)
        plt.yticks(fontproperties='Times New Roman', size=10)
        plt.legend(prop={'family': 'Times New Roman', 'size': 10})
        plt.savefig(acc_path)
        plt.show()

        plt.figure()
        plt.xlabel('Epoch', fontdict={'family': 'Times New Roman', 'size': 14})
        plt.ylabel('Loss', fontdict={'family': 'Times New Roman', 'size': 14})
        plt.plot(x, y3, 'red', linewidth=1.0)
        plt.xticks(fontproperties='Times New Roman', size=10)
        plt.yticks(fontproperties='Times New Roman', size=10)
        plt.legend(prop={'family': 'Times New Roman', 'size': 10})
        plt.savefig(loss_path)
        plt.show()
