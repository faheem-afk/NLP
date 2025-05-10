from utils import find_unique_labels, logger, load_data
import settings
import torch.nn as nn
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from modelling import Model_LSTM, Model_RNN, Model_GRU


class TrainingPrediction():
   
    def __init__(self, train_loaderObject, validation_loaderObject, lstmObject, BertModelObject, final_embeddings_test, final_embeddings_validation, model_name):
        self.tokenized_dataset_test = BertModelObject.tokenized_dataset_test
        self.tokenized_dataset_validation = BertModelObject.tokenized_dataset_validation
        self.device = settings.deviceOption()
        self.epochs = settings.epochs
        self.lr = settings.lr
        self.final_embeddings_test = final_embeddings_test
        self.final_embeddings_validation = final_embeddings_validation
        self.df_train, _ , _ = load_data("data/preprocessed_data")
        self.model = lstmObject
        self.train_loader = train_loaderObject
        self.validation_loader = validation_loaderObject
        self.model_name = model_name

  
    def Training_and_prediction(self):
        model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

        for epoch in range(self.epochs):
            total_loss = 0
            model.train()
            for batch_inputs, batch_pos, batch_labels in self.train_loader: 

                batch_inputs, batch_pos, batch_labels = batch_inputs.to(self.device), batch_pos.to(self.device), batch_labels.to(self.device)
        
                self.optimizer.zero_grad()
        
                y_pred = model(batch_inputs, batch_pos).view(-1, 4)

                loss = self.criterion(y_pred, batch_labels.view(-1))
        
                loss.backward()
        
                self.optimizer.step()
        
                total_loss += loss.item()
                
            print(total_loss / len(self.train_loader))
            
            vald_loss = self.compute_validation(model)

            scheduler.step(vald_loss)

        print(f"{'='*80}Training Complete using{model}{'=' * 80}")

        model.to('cpu')

        torch.save(model.state_dict(), f"artifacts/{self.model_name}.pt")

        if self.model_name == 'model_rnn':
            model = Model_RNN().to(self.device)
        elif self.model_name == 'model_lstm':
            model = Model_LSTM().to(self.device)
        else:
            model = Model_GRU().to(self.device)

        model.load_state_dict(torch.load(f"artifacts/{self.model_name}.pt", map_location=torch.device(self.device)))
       
        self.report(model, self.tokenized_dataset_test)
       
        print(f"{'='*80}Prediction Complete using{self.model_name}{'=' * 80}")

   
    def compute_validation(self, model):
        model.eval()
        
        with torch.no_grad():
            total_loss_validation = 0

            for batch_inputs, batch_pos, batch_labels in self.validation_loader:

                batch_inputs, batch_pos, batch_labels = batch_inputs.to(self.device), batch_pos.to(self.device), batch_labels.to(self.device)
        
                y_pred = model(batch_inputs, batch_pos).view(-1, 4)

                loss = self.criterion(y_pred, batch_labels.view(-1))            
        
                total_loss_validation += loss.item()
            
            avg_loss = total_loss_validation  / len(self.validation_loader)
        
            return avg_loss


    def make_pred(self, model, df):

        with torch.no_grad():
            output = model(self.final_embeddings_test.to(self.device), torch.tensor(df['pos']).to(self.device)).view(-1, 4)
        
            probs = torch.nn.functional.softmax(output, dim=1)
            value, index = torch.max(probs, dim=1)

        return index, torch.tensor(df['labels'])


   
    def report(self, model, df):
        model.eval()
        
        prediction, padded_true_labels = self.make_pred(model, df)

        prediction = prediction.cpu().numpy()
        
        padded_true_labels = padded_true_labels.cpu().numpy()
        
        pred_outer_sent = []
        actual_sent=[]
        names_pred = []
        true = []
        for ix, i in enumerate(padded_true_labels):
            pred_inner_sent = []
            actual_ = []
            for jx, j in enumerate(i):
                if j == -100:
                    pass
                else:
                    
                    pred_inner_sent.append(prediction.reshape(250, -1)[ix][jx])
                    actual_.append(j)
                    true.append(j)
                    names_pred.append(prediction.reshape(250, -1)[ix][jx])
                    
            pred_outer_sent.append(pred_inner_sent)
            actual_sent.append(actual_)
    
            
        output_0 = []
        output_1 = []
        output_2 = []
        output_3 = []

    
        
        for i in range(len(pred_outer_sent)):
            try:
                
                actual = actual_sent[i]
                pred = pred_outer_sent[i]
            
                output_0.append(classification_report(actual, pred, output_dict=True, zero_division=0)['0']['f1-score'])
                output_1.append(classification_report(actual, pred, output_dict=True, zero_division=0)['1']['f1-score'])
                output_2.append(classification_report(actual, pred, output_dict=True, zero_division=0)['2']['f1-score'])
                output_3.append(classification_report(actual, pred, output_dict=True, zero_division=0)['3']['f1-score'])
            
            except KeyError:
                continue

        res = {

            "B-AC": np.array(output_0).mean(),
            "B-LF": np.array(output_1).mean(),
            "I-LF": np.array(output_2).mean(),
            "O": np.array(output_3).mean(),
        }

        logger(res, self.model_name)

        print(f"B-AC: {np.array(output_0).mean()}")
        print(f"B-LF: {np.array(output_1).mean()}")
        print(f"I-LF: {np.array(output_2).mean()}")
        print(f"O: {np.array(output_3).mean()}")

        tick_labels = sorted(find_unique_labels(self.df_train), key=lambda x: x)
        
        cm = confusion_matrix(true, names_pred)
        
        plt.figure(figsize=(6, 5))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=tick_labels, 
                yticklabels=tick_labels)
        
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

        return res
