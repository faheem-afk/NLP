import settings
import torch
from torch.utils.data import DataLoader, TensorDataset



class GenerateEmbeddings():
    def __init__(self, BertModelTrainingObject):
        self.Bert = BertModelTrainingObject
        self.model = self.Bert.bert_model()
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
       
        input_ids = torch.tensor(self.Bert.tokenized_dataset_validation['input_ids'])
        attention_mask = torch.tensor(self.Bert.tokenized_dataset_validation['attention_mask'])
        token_type_ids = torch.tensor(self.Bert.tokenized_dataset_validation['token_type_ids'])
        dataset_validation = TensorDataset(input_ids, attention_mask, token_type_ids)
        dataloader_validation = DataLoader(dataset_validation, batch_size=16)  

        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            all_embeddings_train = []
            for batch in dataloader_train:
                b_input_ids, b_attn_mask, b_token_type_ids = [x.to(self.device) for x in batch]
                outputs = self.model(input_ids=b_input_ids, attention_mask=b_attn_mask, token_type_ids=b_token_type_ids, output_hidden_states=True)
                embeddings = torch.mean(torch.stack(outputs.hidden_states[-4:]), dim=0)
                all_embeddings_train.append(embeddings)

            all_embeddings_test = []
            for batch in dataloader_test:
                b_input_ids, b_attn_mask, b_token_type_ids = [x.to(self.device) for x in batch]
                outputs = self.model(input_ids=b_input_ids, attention_mask=b_attn_mask, token_type_ids=b_token_type_ids, output_hidden_states=True)
                embeddings = torch.mean(torch.stack(outputs.hidden_states[-4:]), dim=0)
                all_embeddings_test.append(embeddings)

            all_embeddings_validation = []
            for batch in dataloader_validation:
                b_input_ids, b_attn_mask, b_token_type_ids = [x.to(self.device) for x in batch]
                outputs = self.model(input_ids=b_input_ids, attention_mask=b_attn_mask, token_type_ids=b_token_type_ids, output_hidden_states=True)
                embeddings = torch.mean(torch.stack(outputs.hidden_states[-4:]), dim=0)
                all_embeddings_validation.append(embeddings)

        
        final_embeddings_train = torch.cat(all_embeddings_train, dim=0)
        final_embeddings_test = torch.cat(all_embeddings_test, dim=0)
        final_embeddings_validation = torch.cat(all_embeddings_validation, dim=0)

        return final_embeddings_train, final_embeddings_test, final_embeddings_validation





