from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from Prediction import Prediction
from data_ingestion import DataIngestion

app = Flask(__name__)
CORS(app) 

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
    elif model == 'lstm':
       message = pred.prediction()
    elif model == 'gru':
        message = pred.prediction() 
    else:
        message = "Invalid model selected."

    return jsonify({'result': message})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)


