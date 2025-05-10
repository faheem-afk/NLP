from utils import labelled, load_data
from transformers import BertTokenizerFast

class Tokenization():
    
    def __init__(self):
        self.df_train, _, _ = load_data("data")
        

    def tokenize(self, batch):
    
        # bert_tokenizer initialization
        tokenizer_bert = BertTokenizerFast.from_pretrained("bert-base-cased")
        
        data_labelled = labelled(self.df_train)
        
        tokenized_inputs = tokenizer_bert(batch["tokens"], 
                                    is_split_into_words=True, 
                                    truncation=True, 
                                    padding='max_length', 
                                    return_offsets_mapping=True,
                                    )  

        all_labels = []
        pos = []
        for i, labels in enumerate(batch["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            pos_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                    pos_ids.append(0)
                elif word_idx != previous_word_idx:
                    label_ids.append(data_labelled[labels[word_idx]])
                    pos_ids.append(batch['pos_tags'][i][word_idx])
                else:
                    label_ids.append(-100)
                    pos_ids.append(0)  
                previous_word_idx = word_idx
            all_labels.append(label_ids)
            pos.append(pos_ids)

        tokenized_inputs["labels"] = all_labels
        tokenized_inputs['pos'] = pos
        
        return tokenized_inputs


    
