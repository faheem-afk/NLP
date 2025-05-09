from data_ingestion import DataIngestion
from data_preprocessing import DataPreprocessing
from tokenization import Tokenization
from modelling import BertModelTraining, Model_LSTM, Model_RNN, Model_GRU
from embeddings import GenerateEmbeddings
from datasetMod import Datasett  
from torch.utils.data import DataLoader
from training_prediction import TrainingPrediction
import torch
from utils import load_data


dataIngestionObject = DataIngestion()

dataPreprocessingObject = DataPreprocessing()

tokenizationObject = Tokenization(dataPreprocessingObject)

bertModelObject = BertModelTraining(dataPreprocessingObject, tokenizationObject)

generateEmbeddingsObject = GenerateEmbeddings(bertModelObject)

final_embeddings_train, final_embeddings_test = generateEmbeddingsObject.get_embeddings()

datasetObject = Datasett(final_embeddings_train, torch.tensor(bertModelObject.tokenized_dataset_train['pos']), torch.tensor(bertModelObject.tokenized_dataset_train['labels']))

loaderObject = DataLoader(datasetObject, batch_size=32, shuffle=True, pin_memory=True)

models = {
    "model_lstm":Model_LSTM(),
    "model_rnn":Model_RNN(),
    "model_gru":Model_GRU()
}

for model in models:
   
    modelObject = models[model]
    trainingPredictionObject = TrainingPrediction(loaderObject, modelObject, bertModelObject, final_embeddings_test, dataPreprocessingObject, model)
   
    trainingPredictionObject.Training_and_prediction()

