import torch
import numpy as np

class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, max_len, tweet_df):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tweets_dataset = tweet_df
        self.label2id = {
            "no influencer": 0,
            "nano": 1,
            "micro": 2,
            "macro": 3,
            "mega": 4
        }
  
    def __len__(self):
        return len(self.tweets_dataset)
  
    def __getitem__(self, idx):
        tweet = self.tweets_dataset.iloc[idx]['texts']
        label = self.tweets_dataset.iloc[idx]['class']
        user_id = self.tweets_dataset.iloc[idx]['twitter user id']
        label = self.label2id[label]
        labels_matrix = np.zeros(5)
        labels_matrix[label] = 1
   
        encoding = self.tokenizer(
            text = tweet,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True
        )

        return {
            'tweet': tweet,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(labels_matrix, dtype=torch.float),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'user_id': user_id
        }