from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from Prediction import Prediction
import json
from datetime import datetime
from google.cloud import storage

app = Flask(__name__)
CORS(app) 

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
        message = pred.prediction()
        log_interaction((user_input_1, user_input_2), model, message)
    elif model == 'lstm':
       message = pred.prediction()
       log_interaction((user_input_1, user_input_2), model, message)
    elif model == 'gru':
        message = pred.prediction() 
        log_interaction((user_input_1, user_input_2), model, message)
    else:
        message = "Invalid model selected."

    return jsonify({'result': message})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)


