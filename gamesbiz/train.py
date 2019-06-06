import os
import json

import os
import sys
import traceback

import numpy as np
import pandas as pd
import sagemaker as sage
from time import gmtime, strftime
import numpy as np

pd.options.mode.chained_assignment = None
from copy import deepcopy
from string import punctuation
from random import shuffle

import re
from string import punctuation 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import boto3
import json
import pandas as pd
from sagemaker import get_execution_role
from io import StringIO
import datetime

import keras
import tensorflow
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Activation
from keras.layers.embeddings import Embedding
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

import emoji

from gamesbiz.resolve import paths

prefix = '/opt/ml/'

input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", 
                   "can't've": "cannot have", "'cause": "because", "could've": "could have", 
                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 
                   "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
                   "he'll've": "he will have", "he's": "he is", "how'd": "how did", 
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
                   "I'll've": "I will have","I'm": "I am", "I've": "I have", 
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
                   "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
                   "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
                   "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                   "she's": "she is", "should've": "should have", "shouldn't": "should not", 
                   "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                   "this's": "this is",
                   "that'd": "that would", "that'd've": "that would have","that's": "that is", 
                   "there'd": "there would", "there'd've": "there would have","there's": "there is", 
                       "here's": "here is",
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                   "they'll've": "they will have", "they're": "they are", "they've": "they have", 
                   "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                   "we're": "we are", "we've": "we have", "weren't": "were not", 
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
                   "what's": "what is", "what've": "what have", "when's": "when is", 
                   "when've": "when have", "where'd": "where did", "where's": "where is", 
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                   "who's": "who is", "who've": "who have", "why's": "why is", 
                   "why've": "why have", "will've": "will have", "won't": "will not", 
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                   "you'll've": "you will have", "you're": "you are", "you've": "you have" } 

smileys ={
        ":‑)":"smiley",
        ":-]":"smiley",
        ":-3":"smiley",
        ":->":"smiley",
        "8-)":"smiley",
        ":-}":"smiley",
        ":)":"smiley",
        ":]":"smiley",
        ":3":"smiley",
        ":>":"smiley",
        "8)":"smiley",
        ":}":"smiley",
        ":o)":"smiley",
        ":c)":"smiley",
        ":^)":"smiley",
        "=]":"smiley",
        "=)":"smiley",
        ":-))":"smiley",
        ":‑D":"smiley",
        "8‑D":"smiley",
        "x‑D":"smiley",
        "X‑D":"smiley",
        ":D":"smiley",
        "8D":"smiley",
        "xD":"smiley",
        "XD":"smiley",
        ":‑(":"sad",
        ":‑c":"sad",
        ":‑<":"sad",
        ":‑[":"sad",
        ":(":"sad",
        ":c":"sad",
        ":<":"sad",
        ":[":"sad",
        ":-||":"sad",
        ">:[":"sad",
        ":{":"sad",
        ":@":"sad",
        ">:(":"sad",
        ":'‑(":"sad",
        ":'(":"sad",
        ":‑P":"playful",
        "X‑P":"playful",
        "x‑p":"playful",
        ":‑p":"playful",
        ":‑Þ":"playful",
        ":‑þ":"playful",
        ":‑b":"playful",
        ":P":"playful",
        "XP":"playful",
        "xp":"playful",
        ":p":"playful",
        ":Þ":"playful",
        ":þ":"playful",
        ":b":"playful",
        "<3":"love"
        }

def clean_tokens(tweet):

    tweet = tweet.lower()
    
    tokens = tokenizer.tokenize(tweet)
    
    #remove call outs
    tokens = filter(lambda t: not t.startswith('@'), tokens)
    
    tweet = " ".join(tokens)
    
    # convert emojis to words
    tweet = emoji.demojize(tweet).replace(":"," ").replace("_"," ")
    #remove numbers
    tweet = re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)', '', tweet)
    tweet = re.sub(r'/([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)/', '', tweet)
    #clean apostrophe
    tweet = tweet.replace("’","'")
    tweet = tweet.replace('“','"')
    tweet = tweet.replace('”','"')
    tweet = tweet.replace('…','')
    tweet = tweet.replace('\n','')
    tweet = tweet.replace('...','')
    tweet = tweet.replace('..','')
    tweet = tweet.replace('�','')
    tweet = tweet.replace('£','')
    tweet = tweet.replace('·','')
    tweet = tweet.replace('–','')
    tweet = tweet.replace('🏻','')
    tweet = tweet.replace('à','')
    tweet = tweet.replace(' ‍','')
    
    tokens = tokenizer.tokenize(tweet)
    
    #remove hashtags
    tokens = filter(lambda t: not t.startswith('#'), tokens)
    #remove urls
    tokens = filter(lambda t: not t.startswith('http'), tokens)
    tokens = filter(lambda t: not t.startswith('t.co/'), tokens)
    tokens = filter(lambda t: not t.startswith('ow.ly/'), tokens)
    tokens = filter(lambda t: not t.startswith('bit.ly/'), tokens)
    tokens = filter(lambda t: not t.startswith('soundcloud.com/'), tokens)
    tokens = filter(lambda t: not t.startswith('outline.com/'), tokens)
    
    new_tokens = []
    for token in tokens:
        if len(token.strip())>0:
            new_tokens.append(token)
    
    _stopwords = set(list(punctuation) + ['AT_USER','URL'])
    #_stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])
    
    new_tokens = [smileys[word] if word in smileys else word for word in new_tokens]
    new_tokens = [contraction_mapping[word] if word in contraction_mapping else word for word in new_tokens]
    new_tokens = [word for word in new_tokens if word not in _stopwords]
    
    return new_tokens

def data_process(raw_data):
    
    raw_data=raw_data[raw_data['sentiment']!='MIXED']

    raw_data['tweet'] = raw_data['tweet'].apply(lambda x: clean_tokens(str(x)))

    x_train = raw_data['tweet']
    y_train = raw_data['sentiment']

    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_train_Y = encoder.transform(y_train)
    dummy_train_y = np_utils.to_categorical(encoded_train_Y)
    
    tokenizer_obj = Tokenizer()   

    tokenizer_obj.fit_on_texts(x_train)    
    max_length = 100 # max([len(s.split()) for s in x_train])    
    vocab_size = len(tokenizer_obj.word_index)+1  

    #Building the vectors of words
    x_train_tokens = tokenizer_obj.texts_to_sequences(x_train)

    x_train_pad = pad_sequences(x_train_tokens, maxlen=max_length, padding='post')
    
    return x_train_pad, dummy_train_y, vocab_size, max_length

def baseline_model(vocab_size, max_length):

    # create model
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, kernel_initializer="normal", activation='softmax'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def generate_model(X_train, y_train, vocab_size, max_length):

    estimator = baseline_model(vocab_size, max_length)
    estimator.fit(X_train, y_train, epochs=25, batch_size=128, verbose=2)
    return estimator

def read_config_file(config_json):
    """This function reads in a json file like hyperparameters.json or resourceconfig.json
    :param config_json: this is a string path to the location of the file (for both sagemaker or local)
    :return: a python dict is returned"""

    config_path = paths.config(config_json)
    if os.path.exists(config_path):
        json_data = open(config_path).read()
        return(json.loads(json_data))


def entry_point():
    """
    This function acts as the entry point for a docker container that an be used to train
    the model either locally or on Sagemaker depending in whichever context its called in as per resolve.paths class.
    """
    # Turn off TensorFlow warning messages in program output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    print("Print in Train.entry_point")
    print(paths.input('training', 'tweets.csv'))
    
    # load training data set from csv file
    training_data_df = pd.read_csv(paths.input('training', 'tweets.csv'))
    x_train, y_train, vocab_size, max_length = data_process(training_data_df)

    optimized_classifier = generate_model(x_train, y_train, vocab_size, max_length)
    optimized_classifier.model.save(os.path.join(model_path, 'ann-churn.h5'))

if __name__ == "__main__":
    entry_point()
