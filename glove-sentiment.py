from numpy.random import random, permutation, randn, normal, uniform, choice
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter   #Replace this with an efficient version
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import nltk.data
import sklearn
import pickle
import bcolz
import re
import os

import tensorflow as tf
from tensorflow.python.keras.models import save_model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras._impl.keras.optimizers import Adam
from tensorflow.python.keras._impl.keras.preprocessing import sequence
from tensorflow.python.keras.layers import  Convolution1D, Dense, Dropout, Embedding, Flatten, MaxPooling1D

glove_path = 'C:\\Users\\Karthik\\Desktop\\sentiment_analysis\\imdb\\glove\\'

#One-time run.
with open(glove_path+ 'glove.6B.50d.txt', 'r', encoding="utf8") as f:
    lines = [line.split() for line in f]
    words = [d[0] for d in lines]
    vecs = np.stack(np.array(d[1:], dtype=np.float32) for d in lines)
    wordidx = {o:i for i,o in enumerate(words)}
    c=bcolz.carray(vecs, rootdir=glove_path+ 'glove.6B.50d.dat', mode='w')
    c.flush()
    pickle.dump(words, open(glove_path+'glove.6B.50d_words.pkl','wb'))
    pickle.dump(wordidx, open(glove_path+'glove.6B.50d_idx.pkl','wb'))

#Load the vectors from GloVe
vecs = bcolz.open(glove_path+ 'glove.6B.50d.dat')[:]
words = pickle.load(open(glove_path+'glove.6B.50d_words.pkl','rb'))
wordidx = pickle.load(open(glove_path+'glove.6B.50d_idx.pkl','rb'))

#User Defined function to retrieve Word Vector
def w2v(w): return vecs[wordidx[w]]

def review_to_wordlist(review):
    review_text = BeautifulSoup(review, "lxml").get_text()
    review_text = re.sub("[^a-zA-Z0-9\-]"," ", review_text)
    words = review_text.lower().split()
    words += '.'
    return(words)

#punkt tokenizer for sentence splitting
#nltk.download()   
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_sentences( review, tokenizer):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence))
    return sentences

app_path = r"C:\Users\Karthik\Desktop\sentiment_analysis\imdb"

corpus_train = pd.read_csv(os.path.join(app_path,"labeledTrainData.tsv"), header=0,                     delimiter="\t", quoting=3)
corpus_test = pd.read_csv(os.path.join(app_path,"testData.tsv"), header=0,                     delimiter="\t", quoting=3)
unlabeled_corpus_train = pd.read_csv(os.path.join(app_path,"unlabeledTrainData.tsv"), header=0,                     delimiter="\t", quoting=3)

sentences = []  # Initialize an empty list of sentences
print ("Parsing sentences from training set")
for review in corpus_train["review"]:
    sentences += review_to_sentences(review, tokenizer)
train_sentences = list(sentences)

print ("Parsing sentences from unlabeled set")
for review in unlabeled_corpus_train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print ("Parsing sentences from test set")
for review in corpus_test["review"]:
    sentences += review_to_sentences(review, tokenizer)

#All words used in training and unlabeled datasets. ! Should test words be included here?
def accum_words(data):
    words = []
    for i in data:
        for d in i:
            words.append(d)
    return words

words_union = accum_words(sentences)
print('Total words in training and unlabeled dataset: ', len(words))

cnt = Counter(words_union)
word_freq_inv = cnt.most_common()
idx = {word_freq_inv[i][0] : i for i in range(len(cnt))}
idx2word = {v: k for k, v in idx.items()}
print('Length of word index {train + unlabeled}: ', idx.__len__())
print(vecs.shape)

vocab_size = 15000

n_fact = vecs.shape[1]
emb = np.zeros((vocab_size, n_fact))
for i in range(1,len(emb)):
    word = idx2word[i]
    if word and re.match(r"^[a-zA-Z0-9\-]*$", word):
        try:
            src_idx = wordidx[word] #GloVe
            emb[i] = vecs[src_idx]
        except KeyError:
            emb[i] = normal(scale=0.6, size=(n_fact,))
    else:
        #random initialization for missing words
        emb[i] = normal(scale=0.6, size=(n_fact,))

#random initialization for rare words
emb[-1] = normal(scale=0.6, size=(n_fact,))
emb/=3

seq_len = 1500
X = corpus_train[["id", "review"]]
y = corpus_train["sentiment"]

#Should EoS-'period' be handled here or within the rev2sentnc func?
def reformat_dataset_list_of_words(dataset):
    X_revw_indx = []
    for record in dataset:
        indices = []
        sentences = []
        sentences += review_to_sentences(record, tokenizer)
        for sentence in sentences:
            for word in sentence:
                indices.append(idx[word])
        X_revw_indx.append(indices)
    return X_revw_indx

X_revw_indx = reformat_dataset_list_of_words(X['review'])

#split the dataset:   #Replace this with native TF for efficient splitting
sss = StratifiedShuffleSplit(n_splits=3, test_size=0.1, random_state=0)

#Splitting the corpus into train and test
for train_index, test_index in sss.split(X, y):
    #print(train_index)
    X_train, X_test = [X_revw_indx[i] for i in train_index], [X_revw_indx[i] for i in test_index]
    y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

#rev_lengths = [X_reviews[i].__len__() for i in range(len(X_revw_indx))]

#print("Average sequence length: ", np.mean(np.array(rev_lengths)))
#print("Maximum sequence length: ", np.max(np.array(rev_lengths)))
#print("Minimum sequence length: ", np.min(np.array(rev_lengths)))

#Replace words with rank > vocab_size with a constant
X_train = [np.array([i if i<vocab_size-1 else vocab_size-1 for i in s]) for s in X_train]
X_test = [np.array([i if i<vocab_size-1 else vocab_size-1 for i in s]) for s in X_test]

df2 = pd.DataFrame({"review": X_test, "sentiment": y_test})
df3 = pd.DataFrame({"review": X_train, "sentiment": y_train})
df2.to_csv("ValidationData.csv")
df3.to_csv("TrainData.csv")

X_train = sequence.pad_sequences(X_train, maxlen=seq_len, value=0)
X_test = sequence.pad_sequences(X_test, maxlen=seq_len, value=0)
y_train=np.array(y_train)
y_test = np.array(y_test)

model = Sequential([
    Embedding(vocab_size, 50, input_length=seq_len, #dropout=0.2, 
              weights=[emb], trainable=False),
    Dropout(0.25),
    Convolution1D(64, 5, padding='same', activation='relu'),
    Dropout(0.25),
    MaxPooling1D(),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(0.7),
    Dense(1, activation='sigmoid')])

model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=64)
model.layers[0].trainable=True
model.optimizer.lr=1e-4

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=64)
model_path = "C:\\Users\\Karthik\\Desktop\\sentiment_analysis\\imdb\\"
model.save_weights(model_path+'glove50_08_12_2017.h5')
save_model(
    model,
    filepath = model_path+'CNN_glove_model_with_weights_08_12_2017.h5',
    overwrite=True,
    include_optimizer=True
)

