from utils import inverse_labelled, labelled
from datasets import Dataset
from transformers import BertForTokenClassification, TrainingArguments, Trainer
import torch.nn as nn
import torch
import settings
import os 


class BertModelTraining():
    
    def __init__(self, DataPreprocessingObject, TokenizationObject):
        self.df_train, self.df_test = DataPreprocessingObject.pre_processing()
        self.tokenizer = TokenizationObject
        self.device = settings.deviceOption()
   
    def bert_model(self):

        dataset_train = Dataset.from_pandas(self.df_train)
        dataset_test = Dataset.from_pandas(self.df_test)
        self.tokenized_dataset_train = dataset_train.map(self.tokenizer.tokenize,  batched=True)
        self.tokenized_dataset_test = dataset_test.map(self.tokenizer.tokenize, batched=True)
        

        # bert_model_initialization
        model_bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=4, id2label=inverse_labelled(self.df_train), label2id=labelled(self.df_train)).to(self.device)

        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=8,
            num_train_epochs=3,
            logging_dir="./logs",
            
        )

        trainer = Trainer(
            model=model_bert,
            args=training_args,
            train_dataset=self.tokenized_dataset_train,
        )

        trainer.train()
        
        model_bert.to('cpu')

        os.makedirs("artifacts", exist_ok=True)

        torch.save(model_bert.state_dict(), "artifacts/model_bert.pt")

        model_bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=4, id2label=inverse_labelled(self.df_train), label2id=labelled(self.df_train)).to(self.device)

        model_bert.load_state_dict(torch.load("artifacts/model_bert.pt", map_location=torch.device(self.device)))
        
        return model_bert


class Model_RNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(512, 100) 
        
        self.rnn = nn.RNN(868, 64, batch_first=True, bidirectional=True)

        hidden_size = 64 * 2 if self.rnn.bidirectional else 64

        self.hidden1 = nn.Linear(hidden_size, 32)
        self.norm = nn.LayerNorm(32)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        self.output_layer = nn.Linear(32, 4)

    def forward(self, inputs, pos):
        pos = pos.long()

        pos_embedding = self.embedding(pos)
        
        combined_input = torch.cat((inputs, pos_embedding), dim=-1)
        
        output, _ = self.rnn(combined_input) 
       
        x = output 
        
        B, T, H = x.shape
        x = x.contiguous().view(B * T, H)

        x = self.hidden1(x)
        x = self.norm(x)
        
        x = self.relu1(x)
        x = self.dropout(x)
        
        x = self.output_layer(x)

        x = x.view(B, T, -1)

        return x


class Model_GRU(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.embedding = nn.Embedding(512, 100) 
        
        self.gru = nn.GRU(868, 64, batch_first=True, bidirectional=True)

        hidden_size = 64 * 2 if self.gru.bidirectional else 64

        self.hidden1 = nn.Linear(hidden_size, 32)
        self.norm = nn.LayerNorm(32)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        self.output_layer = nn.Linear(32, 4)

    def forward(self, inputs, pos):
        
        pos = pos.long()

        pos_embedding = self.embedding(pos)
        
        combined_input = torch.cat((inputs, pos_embedding), dim=-1)
        
        output, _ = self.gru(combined_input)  
       
        x = output 
        
        B, T, H = x.shape
        x = x.contiguous().view(B * T, H)

        x = self.hidden1(x)
        x = self.norm(x)
        x = self.relu1(x)
        
        x = self.dropout(x)
        
        x = self.output_layer(x)

        x = x.view(B, T, -1)

        return x



class Model_LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(512, 100) 
        
        self.lstm = nn.LSTM(868, 64, batch_first=True, bidirectional=True)

        hidden_size = 64 * 2 if self.lstm.bidirectional else 64

        self.hidden1 = nn.Linear(hidden_size, 32)
        self.norm = nn.LayerNorm(32)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        self.output_layer = nn.Linear(32, 4)

    def forward(self, inputs, pos):
        pos = pos.long()

        pos_embedding = self.embedding(pos)

        combined_input = torch.cat((inputs, pos_embedding), dim=-1)
        
        output, _ = self.lstm(combined_input) 
       
        x = output 
        B, T, H = x.shape
        x = x.contiguous().view(B * T, H)

        x = self.hidden1(x)
        x = self.norm(x)
        x = self.relu1(x)
        x = self.dropout(x)
    
    
        x = self.output_layer(x)

        x = x.view(B, T, -1)

        return x