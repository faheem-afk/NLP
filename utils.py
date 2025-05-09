import re
import pickle
import os
import pandas as pd

def labelled(df):
    # label-to-numeric key value pairs
    labelled = {k:v for v, k in enumerate(sorted(find_unique_labels(df)))}
    return labelled


def inverse_labelled(df):
    # numeric-to-label key value pairs
    inverse_labelled = {k:v for k, v in enumerate(sorted(find_unique_labels(df)))}
    return inverse_labelled


def labelled_pos(df):
    # pos-to-numeric key value pairs
    labelled_pos = {k:v for v, k in enumerate(find_unique_pos(df))}
    return labelled_pos


def labeller(x, labelled_pos):
    sent = []
    
    for i in x:
        sent.append(labelled_pos[i])
    return sent


def find_unique_labels(df):
    unique_labels = []
    for i in df['ner_tags']:
        for j in i:
            if j not in unique_labels:
                unique_labels.append(j)
    return unique_labels


def find_unique_pos(df):
    unique_pos_labels = []
    for i in df['pos_tags']:
        for j in i:
            if j not in unique_pos_labels:
                unique_pos_labels.append(j)
    return unique_pos_labels


def remove_punc(text):
    pattern = re.compile("[!\"#$%&\'()*+-./:;<=>?@[\\]^_`{|}~‘]")
    return pattern.sub('', text)


def build_v(df):
    data = []
    for sent_ix, sent in enumerate(df['tokens']):
        dt = {}
        for word_ix, word in enumerate(sent):
            dt[word] = [df['ner_tags'][sent_ix][word_ix], df['pos_tags'][sent_ix][word_ix] ]
        data.append(dt)
    return data


def cleansed(vocab):
    sent_inputs = []
    sent_labels = []
    sent_pos = []
    for sent_ix, di in enumerate(vocab):
        
        input = []
        label = []
        pos = []
        for key in di:
            s = remove_punc(key)
            if len(s) > 0:
                input.append(s)
                label.append(vocab[sent_ix][key][0])
                pos.append(vocab[sent_ix][key][1])
        
        sent_inputs.append(input)
        sent_labels.append(label)
        sent_pos.append(pos)
    return sent_inputs, sent_labels, sent_pos


def load_data():
    df_train = pd.read_parquet('data/train.parquet', engine='pyarrow')
    df_test = pd.read_parquet('data/test.parquet', engine='pyarrow')
    
    return df_train, df_test
