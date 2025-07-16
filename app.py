from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from Prediction import Prediction
from transformers import BertForTokenClassification
from utils import inverse_labelled, labelled, load_data, log_interaction
import torch
from transformers import BertTokenizerFast
import pandas as pd

app = Flask(__name__)
CORS(app) 

df_train = pd.read_parquet("data/train.parquet", engine='pyarrow')
df_train_pre = pd.read_parquet("data/preprocessed_data/train.parquet", engine='pyarrow')

model_bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=4, id2label=inverse_labelled(df_train_pre), label2id=labelled(df_train_pre))
tokenizer_bert = BertTokenizerFast.from_pretrained("bert-base-cased")

model_bert.load_state_dict(torch.load("artifacts/model_bert.pt", map_location=torch.device("cpu")))

@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model = data.get('model')
    user_input_1 = data.get('input')
    user_input_2 = data.get('pos')

    pred = Prediction(model, user_input_1, user_input_2, df_train)

    if model == 'rnn':
        message = pred.prediction(model_bert, tokenizer_bert)
        log_interaction((user_input_1, user_input_2), model, message)
    elif model == 'lstm':
       message = pred.prediction(model_bert, tokenizer_bert)
       log_interaction((user_input_1, user_input_2), model, message)
    elif model == 'gru':
        message = pred.prediction(model_bert, tokenizer_bert) 
        log_interaction((user_input_1, user_input_2), model, message)
    else:
        message = "Invalid model selected."

    return jsonify({'result': message})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)


