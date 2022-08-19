import pandas as pd
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
import torch

categories = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
n_categories = len(categories)

class sentencesDataset(Dataset):

    def __init__(self, data: pd.DataFrame, tokenizer: RobertaTokenizer, max_token_len: int = 30):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        sentence = str(row.Sentence)
        labels = torch.FloatTensor(row[categories])
        encoding = self.tokenizer.encode_plus(sentence, 
                                            add_special_tokens=True, 
                                            return_tensors='pt',
                                            truncation=True,
                                            max_length=self.max_token_len, 
                                            padding='max_length',
                                            return_attention_mask=True)
        
        return {'input_ids': encoding.input_ids.flatten(), 
                'attention_mask': encoding.attention_mask.flatten(), 
                'labels': labels}


class sentencesDatasetPredict(Dataset):

    def __init__(self, data: pd.DataFrame, tokenizer: RobertaTokenizer, max_token_len: int = 30):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        encoding = self.tokenizer.encode_plus(str(row.Sentence),
                                              add_special_tokens=True,
                                              return_tensors='pt',
                                              truncation=True,
                                              max_length=self.max_token_len,
                                              padding='max_length',
                                              return_attention_mask=True)

        return {'input_ids': encoding.input_ids.flatten(),
                'attention_mask': encoding.attention_mask.flatten()}


