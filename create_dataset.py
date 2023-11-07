import torch
from torch.utils.data import Dataset
from mongo_processor import Mongo


class Create_Dataset(Dataset):
    def __init__(self, datas, labels, tokenizer, max_len):
      self.datas = datas
      self.tokenizer = tokenizer
      self.max_len = max_len
      self.labels = labels
    
    def __len__(self):
      return len(self.datas)

    def __getitem__(self, idx):
        inputs = self.tokenizer.encode_plus(text=self.datas[idx],
                                           text_pair=None,
                                           add_special_tokens = True,
                                           max_length=self.max_len,
                                           padding='max_length',
                                           return_token_type_ids=False,
                                           return_attention_mask=True,
                                           truncation=True,
                                           return_tensors='pt' 
                                           )

        input_ids = inputs['input_ids'].flatten()
        attn_mask = inputs['attention_mask'].flatten()
        dataset = {'sentences': self.datas[idx],
                    'input_ids': input_ids,
                    'attention_mask': attn_mask,
                    'labels': torch.tensor(self.labels[idx], dtype=torch.float32)}
        

        return dataset



  