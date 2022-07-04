from Config import Config
from DataManager import DataManager
from Trainer import Trainer
from Model import Model

if __name__ == '__main__':
    config = Config()
    datamanager = DataManager(config)
    model = Model(config, datamanager)
    trainer = Trainer(config, model, datamanager)
    trainer.train()
    trainer.save_training_log()
    trainer.visualize()
