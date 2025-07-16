import settings
from utils import labelled_pos, inverse_labelled
import torch
from modelling import Model_LSTM, Model_RNN, Model_GRU


class Prediction():
    def __init__(self, model, input_, pos, df_train):
        self.device = settings.deviceOption()
        self.input = input_
        self.pos = pos
        self.model = model
        self.df_train = df_train
        self.labelled_pos = labelled_pos(self.df_train)
        

    def prediction(self, model_bert, tokenizer_bert):
        token_ids = tokenizer_bert([self.input], 
                        is_split_into_words=True, 
                        truncation=True, 
                        padding='max_length', 
                        return_offsets_mapping=True,
                      )
        pos = []

        word_ids = token_ids.word_ids(batch_index=0)
        pos_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                pos_ids.append(0)
            elif word_idx != previous_word_idx:
                pos_ids.append(self.labelled_pos[self.pos])
            else:
                pos_ids.append(0)  
            previous_word_idx = word_idx
        pos.append(pos_ids)

        token_ids['pos'] = pos

        output = model_bert(input_ids=torch.tensor([token_ids['input_ids']]), 
                        attention_mask= torch.tensor([token_ids['attention_mask']]), 
                        token_type_ids= torch.tensor([token_ids['token_type_ids']]),
                        output_hidden_states=True)
      
        all_hidden_states = output.hidden_states 

        embeddings = torch.mean(torch.stack(all_hidden_states[-4:]), dim=0)
        padded_pos_torch = torch.tensor(token_ids['pos'])

        with torch.no_grad():
            
            if self.model == 'rnn':
                modelObject = Model_RNN()
                modelObject.load_state_dict(torch.load("model_rnn.pt", map_location=torch.device("cpu")))
            elif self.model == 'lstm':
                modelObject = Model_LSTM()
                modelObject.load_state_dict(torch.load("model_lstm.pt", map_location=torch.device("cpu")))
            else:
                modelObject = Model_GRU()
                modelObject.load_state_dict(torch.load("model_gru.pt", map_location=torch.device("cpu")))

            output = modelObject(embeddings,padded_pos_torch).view(-1, 4)
            probs = torch.nn.functional.softmax(output, dim=1)
            
            value, index = torch.max(probs, dim=1)

        return inverse_labelled(self.df_train)[index[1].item()]
           
