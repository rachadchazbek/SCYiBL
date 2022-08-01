import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sentencesDataset import sentencesDataset

class DataModule(pl.LightningDataModule):

        def __init__(self, data_train, data_test, data_val, tokenizer, batch_size = 32, max_token_len = 128):
            super().__init__()
            self.data_train = data_train
            self.data_test = data_test
            self.data_val = data_val
            self.batch_size = batch_size
            self.max_token_len = max_token_len
            self.tokenizer = tokenizer

        def setup(self, stage=None):
            self.train_dataset = sentencesDataset(self.data_train, self.tokenizer)
            self.test_dataset = sentencesDataset(self.data_test, self.tokenizer)
            self.val_dataset = sentencesDataset(self.data_val, self.tokenizer)
            print('train_dataset')
            print(self.train_dataset)
            print('test dataset')
            print(self.test_dataset)
            print('val dataset')
            print(self.val_dataset) 

        def train_dataloader(self):
            print('using train dataloader')
            print(self.train_dataset.__len__())
            return DataLoader(self.train_dataset,batch_size=self.batch_size,num_workers=2,shuffle=False)
        def val_dataloader(self):
            print('using val dataloader')
            print(self.val_dataset.__len__())
            return DataLoader(self.val_dataset,batch_size=self.batch_size,num_workers=1)
        def predict_dataloader(self):
            print('using test dataloader')
            print(self.test_dataset.__len__())
            return DataLoader(self.test_dataset,batch_size=self.batch_size,num_workers=1, shuffle=False)
        def test_dataloader(self):
            print('using test dataloader')
            print(self.test_dataset.__len__())
            return DataLoader(self.test_dataset,batch_size=self.batch_size,num_workers=2, shuffle=False)

