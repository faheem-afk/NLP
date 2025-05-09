import settings
import torch
from torch.utils.data import DataLoader, TensorDataset



class GenerateEmbeddings():
    def __init__(self, BertModelTrainingObject):
        self.Bert = BertModelTrainingObject
        self.model = BertModelTrainingObject.bert_model()
        self.device = settings.deviceOption()
    


    def get_embeddings(self):

        input_ids = torch.tensor(self.Bert.tokenized_dataset_train['input_ids'])
        attention_mask = torch.tensor(self.Bert.tokenized_dataset_train['attention_mask'])
        token_type_ids = torch.tensor(self.Bert.tokenized_dataset_train['token_type_ids'])

        dataset_train = TensorDataset(input_ids, attention_mask, token_type_ids)
        dataloader_train = DataLoader(dataset_train, batch_size=16)  
        
        input_ids = torch.tensor(self.Bert.tokenized_dataset_test['input_ids'])
        attention_mask = torch.tensor(self.Bert.tokenized_dataset_test['attention_mask'])
        token_type_ids = torch.tensor(self.Bert.tokenized_dataset_test['token_type_ids'])

        dataset_test = TensorDataset(input_ids, attention_mask, token_type_ids)
        dataloader_test = DataLoader(dataset_test, batch_size=16)  

        self.model.eval()
        self.model.to(self.device)

        all_embeddings_train = []
        all_embeddings_test = []

        with torch.no_grad():
            for batch_train, batch_test in zip(dataloader_train, dataloader_test):
                
                b_input_ids_train, b_attn_mask_train, b_token_type_ids_train = [x.to(self.device) for x in batch_train]
                b_input_ids_test, b_attn_mask_test, b_token_type_ids_test = [x.to(self.device) for x in batch_test]

                outputs_train = self.model(
                    input_ids=b_input_ids_train,
                    attention_mask=b_attn_mask_train,
                    token_type_ids=b_token_type_ids_train,
                    output_hidden_states=True
                )

                outputs_test = self.model(
                    input_ids=b_input_ids_test,
                    attention_mask=b_attn_mask_test,
                    token_type_ids=b_token_type_ids_test,
                    output_hidden_states=True
                )

                all_hidden_states_train = outputs_train.hidden_states
                all_hidden_states_test = outputs_test.hidden_states
                
                embeddings_train = torch.mean(torch.stack(all_hidden_states_train[-4:]), dim=0)
                embeddings_test = torch.mean(torch.stack(all_hidden_states_test[-4:]), dim=0)

                all_embeddings_train.append(embeddings_train)
                all_embeddings_test.append(embeddings_test)

        
        final_embeddings_train = torch.cat(all_embeddings_train, dim=0)
        final_embeddings_test = torch.cat(all_embeddings_test, dim=0)
        
        return final_embeddings_train, final_embeddings_test





