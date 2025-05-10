import pandas as pd
import os

class DataIngestion():
    
    def __init__(self):
        self.splits = {'train': 'PLOD-CW-25-Train.parquet', 'test': 'PLOD-CW-25-Test.parquet', 'validation': 'PLOD-CW-25-Val.parquet'}
        self.df_train = pd.read_parquet("hf://datasets/surrey-nlp/PLOD-CW-25/" + self.splits["train"])
        self.df_test = pd.read_parquet("hf://datasets/surrey-nlp/PLOD-CW-25/" + self.splits["test"])
        self.df_validation = pd.read_parquet("hf://datasets/surrey-nlp/PLOD-CW-25/" + self.splits["validation"])
        
        os.makedirs('data', exist_ok=True)
        
        self.df_train.to_parquet(f'data/train.parquet', engine='pyarrow')
        self.df_test.to_parquet(f'data/test.parquet', engine='pyarrow')
        self.df_validation.to_parquet(f'data/validation.parquet', engine='pyarrow')
        

   


