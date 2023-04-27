import torch
import random
from torch.utils.data import Dataset, DataLoader

class InfluencerProfilerDataset(Dataset):

    def __init__(self, tokenizer, max_len, tweet, label):
        self.tokenizer = tokenizer
        self.max_len = max_len

        # tweet 
        self.tweet = tweet
        self.label = label
  
    def __len__(self):
        return len(self.tweet)
  
    def __getitem__(self, idx):
        tweet = self.tweet[idx]
        label = self.label[idx]
   
        encoding = self.tokenizer(
            tweet,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'tweet': tweet,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def concatenate_texts(group, max_len, tokenizer):
    texts = list(group['texts'])
    random.shuffle(texts)
    concatenated_text = ''
    for text in texts:
        concatenated_text += ' ' + text
        if len(tokenizer.encode(concatenated_text, truncation=True)) > max_len:
            concatenated_text = concatenated_text[:-len(text)]
            break
    return concatenated_text.strip()

def create_data_loader(df, tokenizer, max_len, batch_size=1):
    grouped_df = df.groupby('twitter user id').apply(lambda group: concatenate_texts(group, max_len, tokenizer)).reset_index()
    grouped_df.columns = ['twitter user id', 'concatenated_texts']

    modified_df = grouped_df.merge(df[['twitter user id', 'class']].drop_duplicates(), on='twitter user id')

    ds = InfluencerProfilerDataset(
        tokenizer=tokenizer,
        max_len=max_len,
        tweet=modified_df['concatenated_texts'].to_numpy(),
        label=modified_df['class'].to_numpy().astype(int),
    )

    return DataLoader(
        ds,
        batch_size=batch_size
    )
