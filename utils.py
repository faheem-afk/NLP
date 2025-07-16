import re
import pandas as pd
import json
from datetime import datetime
from google.cloud import storage
import os

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


def load_data(path):
    df_train = pd.read_parquet(f'{path}/train.parquet', engine='pyarrow')
    df_test = pd.read_parquet(f'{path}/test.parquet', engine='pyarrow')
    df_validation = pd.read_parquet(f'{path}/validation.parquet', engine='pyarrow')
    
    return df_train, df_test, df_validation


def log_interaction(user_input, model_name, prediction):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": user_input,
        "model": model_name,
        "prediction": prediction
    }

    storage_client = storage.Client()
    bucket = storage_client.get_bucket('logs_nlp')

    blob = bucket.blob('logs/log_file.jsonl')

    with blob.open('wt') as log_file:
        log_file.write(json.dumps(log_entry) + "\n")

def logger(res, model_name):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "results" : res,
        "model":model_name
    }
    os.makedirs("Logs", exist_ok=True)
    with open("Logs/logs_file.jsonl", "a") as file:
        file.write(json.dumps(log_entry) + "\n")


def download_weights(bucket_name, source_blob_name, destination_file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    