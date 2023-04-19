import torch
from torch.utils.data import Dataset, DataLoader

class InfluencerProfilerDataset(Dataset):

  def __init__(self,tokenizer, max_len, tweet, label):
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
      max_length = self.max_len,
      padding ='max_length',
      truncation = True,
      return_tensors='pt',
    )

    return {
      'tweet': tweet,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'label' : torch.tensor(label, dtype=torch.long)
    }


def create_data_loader(df, tokenizer, max_len, batch_size=1):
  ds = InfluencerProfilerDataset(
    tokenizer=tokenizer,
    max_len=max_len,
    tweet=df['texts'].to_numpy(),
    label=df['class'].to_numpy().astype(int),
    # features= df.iloc[:,2:].to_numpy().astype(float)
  )

  return DataLoader(
    ds,
    batch_size=batch_size
  )