# WORK5 Sarcasm Detection

## Model Architecture

### Part I. Feature Extraction

Branch I. Multi-scale convolution layer(kernel_size = 2, 3, 4, 5)

Branch II. Bidirectional LSTM with self-attention

### Part II. Classifying

Linear + Activation + Linear

## Data

Training set: 0~19999
Testing set: 20000~28618

Word Embedding: GloVe6B

##Results

100% on training set, 85.1% on testing set(Default settings).

## Last modification: 2022.07.04