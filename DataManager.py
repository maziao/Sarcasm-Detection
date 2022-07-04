import re
import json
import time
import nltk
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter
from torch.nn.utils.rnn import pad_sequence


class DataManager:
    """
        A class which prepares data for all the six models in this project.
        The output data can be further processed in 'class ***_DataLoader' to form batches.
    """
    def __init__(self, config):
        """
            Get the configuration object which contains all the information of the model to be trained.
            Get the pretrained model and the dataset loaded.
        """
        self.sentences = None
        self.lengths = None
        self.labels = None

        self.embedding_model = None
        self.word_vec_dim = None
        self.dictionary = None
        self.word_embedding = None

        self.config = config

        self.load_word_embedding_model()
        self.load_training_data()
        self.build_dict()
        self.build_word_embedding()
        print('Data loading completed.')
        print('==================================================')

        del self.embedding_model

    def get_sentences(self):
        """
            Returns a list of training sentences in the form of embedding indices of each word.
            If a word in the sentence is not in the dictionary, it will be replaced by the most frequently occurred word
                in the training set(index '1').
            If 'trimming_and_padding' parameter in the config file is True, all the sentences will be transformed into
                the same length(according to the parameter 'max_sen_len' in the config file).
        """
        sentence_list = []
        for i, sentence in enumerate(self.sentences):
            embedded_sentence = []
            for word in sentence:
                if word in self.dictionary.keys():
                    embedded_sentence.append(self.dictionary[word])
                else:
                    embedded_sentence.append(1)
            sentence_list.append(torch.tensor(embedded_sentence))
        sentence_list = pad_sequence(sentence_list, batch_first=True)
        return sentence_list

    def get_lengths(self):
        """
            Returns a list of integer, representing the lengths of the sentences in the training set.
            This can be used to mask the padded information in LSTM, etc.
            Note that this info is only valid when 'max_sen_len' is greater than the actual maximum length of sentences.
        """
        return torch.tensor(self.lengths)

    def get_labels(self):
        """
            Returns a list of one-hot vectors for each label of the training sentences.
        """
        return torch.tensor(self.labels)

    def get_length_masks(self):
        """
            Returns a list of length mask for every sentence in the training set.
            In every mask, '1' represents a word exists in that position, while '0' represents padding.
        """
        mask_list = []
        for i in range(len(self.sentences)):
            mask = []
            for j in range(self.lengths[i]):
                mask.append(1)
            for j in range(self.config.max_sen_len - self.lengths[i]):
                mask.append(0)
            mask_list.append(mask)
        mask_list = np.array(mask_list, dtype=np.bool)
        return mask_list

    def get_word_embedding(self):
        """
            Returns a numpy matrix, containing the embedding vectors of the most frequently occurred words
                in the training set.
        """
        return self.word_embedding

    @staticmethod
    def clean_str(text):
        """
            Normalize the input sentences.
        """
        text = text.lower()
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"that's", "that is ", text)
        text = re.sub(r"there's", "there is ", text)
        text = re.sub(r"it's", "it is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "can not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)

        return text.strip()

    def load_word_embedding_model(self):
        """
            Load pretrained word embedding model from the appointed path.
            Get the dimension of word embedding simultaneously.
        """
        start = time.time()
        vocab = {}
        with open(self.config.word_embedding_path, 'r', encoding='utf-8') as f:
            embedding_info = f.readline().strip().split()
            embedding_info = list(map(int, embedding_info))
            vocab_size = embedding_info[0]
            word_vec_dim = embedding_info[1]
            for _ in tqdm(range(vocab_size), desc='Loading pretrained word embedding'):
                line = f.readline().strip().split()
                word = line[0].lower()
                vector = list(map(float, line[1:]))
                vocab[word] = vector
        end = time.time()
        print('Pretrained word embedding model has been loaded from:')
        print('   ', self.config.word_embedding_path)
        print('    Number of embeddings:', vocab_size)
        print('    Dimension of word vector:', word_vec_dim)
        print('    Time consumed:', round(end - start, 2), 's')
        print('==================================================')
        time.sleep(0.01)
        self.embedding_model = vocab
        self.word_vec_dim = word_vec_dim
        self.config.word_vec_dim = word_vec_dim

    def load_training_data(self):
        """
            Load training data, including tokenized headlines, labels and lengths of headlines.
        """
        lines = [line for line in open(self.config.training_data_root, 'r')]
        labels = []
        headlines = []
        for line in lines:
            anno = json.loads(line)
            labels.append(anno['is_sarcastic'])
            headlines.append(anno['headline'])

        max_sen_len = 0

        sentences = []
        lengths = []
        for headline in headlines:
            headline = self.clean_str(headline)
            tokens = nltk.word_tokenize(headline)
            sentences.append(tokens)
            lengths.append(len(tokens))

            if len(tokens) > max_sen_len:
                max_sen_len = len(tokens)

        print('Training data has been loaded. ')
        print('    Max sentence length:', max_sen_len)

        self.sentences = sentences
        self.labels = labels
        self.lengths = lengths
        self.config.max_sen_len = max_sen_len

    def build_dict(self):
        """
            Build a dictionary according to the training set.
        """
        word_counter = Counter()
        for sentence in self.sentences:
            for word in sentence:
                word_counter[word] += 1

        ls = word_counter.most_common()

        # Reserve index 0 for padding
        self.dictionary = {w[0]: index + 1 for (index, w) in enumerate(ls)}
        print('Dictionary building completed.')

    def build_word_embedding(self):
        """
            Build word embedding matrix according to
            (1) Pretrained word embedding model(self.embedding_model)
            (2) Dictionary on the training set(self.dictionary)
        """
        word_embedding = np.random.uniform(-1.0, 1.0, size=(len(self.dictionary) + 1, self.word_vec_dim))
        word_embedding[0] = np.zeros(self.word_vec_dim, dtype=float)
        for word in self.embedding_model.keys():
            if word in self.dictionary:
                word_embedding[self.dictionary[word]] = self.embedding_model[word]
        self.word_embedding = word_embedding.astype(np.float32)
        print('Word embedding building completed.')
