from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from Prediction import Prediction
from transformers import BertForTokenClassification
from utils import inverse_labelled, labelled, load_data, log_interaction
import torch

app = Flask(__name__)
CORS(app) 

df_train, _, _ = load_data("data/preprocessed_data")

model_bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=4, id2label=inverse_labelled(df_train), label2id=labelled(df_train))

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

    pred = Prediction(model, user_input_1, user_input_2)

    if model == 'rnn':
        message = pred.prediction(model_bert)
        log_interaction((user_input_1, user_input_2), model, message)
    elif model == 'lstm':
       message = pred.prediction(model_bert)
       log_interaction((user_input_1, user_input_2), model, message)
    elif model == 'gru':
        message = pred.prediction(model_bert) 
        log_interaction((user_input_1, user_input_2), model, message)
    else:
        message = "Invalid model selected."

    return jsonify({'result': message})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)


