from DataManager import DataManager


class Statistics:
    def __init__(self):
        self.cat_num = 19
        self.remove_entity_mark = True
        self.wordnet_hypernym = False
        self.training_set_path = '../SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
        self.testing_set_path = '../SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'
        self.datamanager = DataManager(self)


if __name__ == '__main__':
    statistics = Statistics()
