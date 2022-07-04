from torch.utils.data import Dataset


class DataSet(Dataset):
    def __init__(self, datamanager, mode):
        sentences = datamanager.get_sentences()
        lengths = datamanager.get_lengths()
        labels = datamanager.get_labels()
        if mode == 'train':
            self.sentences = sentences[:20000]
            self.lengths = lengths[:20000]
            self.labels = labels[:20000]
        else:
            self.sentences = sentences[20000:]
            self.lengths = lengths[20000:]
            self.labels = labels[20000:]

    def __getitem__(self, item):
        return self.sentences[item], self.lengths[item], self.labels[item]

    def __len__(self):
        return len(self.sentences)
