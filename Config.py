import time
import torch
import argparse


class Config:
    def __init__(self):
        self.start_time = time.time()
        self.word_vec_dim = None
        self.word_embedding_path = '../GloVe/glove.6B.50d.word2vec.txt'
        self.training_data_root = '../News-Headlines-Dataset-For-Sarcasm-Detection-master/Sarcasm_Headlines_Dataset.json'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        args = self._get_config()
        for key in args.__dict__:
            setattr(self, key, args.__dict__[key])

        for key in self.__dict__:
            if key == 'start_time':
                continue
            print(key, ':', str(self.__dict__[key]))
        print('==================================================')

    def _get_config(self):
        parser = argparse.ArgumentParser()
        parser.description = 'configuration for the model'

        parser.add_argument('--epoch_num', type=int,
                            default=50,
                            help='number of epochs for training')
        parser.add_argument('--lr', type=float,
                            default=1e-2,
                            help='learning rate')
        parser.add_argument('--momentum', type=float,
                            default=0.9,
                            help='momentum parameter in optimizer SGD')
        parser.add_argument('--weight_decay', type=float,
                            default=1e-4,
                            help='L2 weight decay')
        parser.add_argument('--batch_size', type=int,
                            default=64,
                            help='batch size')
        parser.add_argument('--kernel_size_list', type=list,
                            default=[2, 3, 4, 5],
                            help='list of kernel sizes in conv layer')
        parser.add_argument('--filter_num', type=int,
                            default=50,
                            help='number of filters of each kernel size')
        parser.add_argument('--num_layer', type=int,
                            default=1,
                            help='number of hidden layers in lstm')
        args = parser.parse_args()
        return args

    def get_training_info(self):
        training_info = []
        for key in self.__dict__:
            if key == 'start_time':
                continue
            info = key + ' : ' + str(self.__dict__[key])
            training_info.append(info)
        return training_info
