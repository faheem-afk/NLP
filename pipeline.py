from data_ingestion import DataIngestion
from data_preprocessing import DataPreprocessing
from tokenization import Tokenization
from modelling import BertModelTraining, Model_LSTM, Model_RNN, Model_GRU
from embeddings import GenerateEmbeddings
from datasetMod import Datasett  
from torch.utils.data import DataLoader
from training_prediction import TrainingPrediction
import torch


dataIngestionObject = DataIngestion()

dataPreprocessingObject = DataPreprocessing()

dataPreprocessingObject.pre_processing()

tokenizationObject = Tokenization()

bertModelObject = BertModelTraining(tokenizationObject)

generateEmbeddingsObject = GenerateEmbeddings(bertModelObject)

final_embeddings_train, final_embeddings_test, final_embeddings_validation = generateEmbeddingsObject.get_embeddings()

train_datasetObject = Datasett(final_embeddings_train, torch.tensor(bertModelObject.tokenized_dataset_train['pos']), torch.tensor(bertModelObject.tokenized_dataset_train['labels']))
train_loaderObject = DataLoader(train_datasetObject, batch_size=32, shuffle=True, pin_memory=True)

validation_datasetObject = Datasett(final_embeddings_validation, torch.tensor(bertModelObject.tokenized_dataset_validation['pos']), torch.tensor(bertModelObject.tokenized_dataset_validation['labels']))
validation_loaderObject = DataLoader(validation_datasetObject, batch_size=32, shuffle=False, pin_memory=True)


models = {
    "model_lstm":Model_LSTM(),
    "model_rnn":Model_RNN(),
    "model_gru":Model_GRU()
}

for model in models:
   
    modelObject = models[model]
    trainingPredictionObject = TrainingPrediction(train_loaderObject, validation_loaderObject, modelObject, bertModelObject, final_embeddings_test, final_embeddings_validation, model)
   
    trainingPredictionObject.Training_and_prediction()

