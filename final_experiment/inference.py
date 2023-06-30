
# load python main function and pytorch modules
import os
import sys
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import json

model_name_1 = "cardiffnlp/twitter-roberta-large-2022-154m"
model_name_2 = 'Twitter/twhin-bert-base'
max_tweet_len = 512
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
label2id = {'nano': 0, 'no influencer': 1, 'macro': 2, 'mega': 3, 'micro': 4}
id2label = {0: 'nano', 1: 'no influencer', 2: 'macro', 3: 'mega', 4: 'micro'}


def load_model(model_class, model_1, model_2, num_labels, file_path):
    model = model_class(model_1, model_2, num_labels)
    model.load_state_dict(torch.load(file_path))
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {file_path}")
    return model


def preprocess_tweet(tweet):
    # Preprocess text (username placeholders)
    new_text = []
    for t in tweet.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    tweet = " ".join(new_text)

    # remove spaces
    tweet = tweet.strip()

    # remove new line character
    tweet = re.sub(r'\n', '', tweet)

    # return mention count
    mentions = re.findall(r'@\w+', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    count_mentions = len(mentions)

    # return link count
    links = re.findall(r'http\w+', tweet)
    tweet = re.sub(r'http\w+', '', tweet)
    count_links = len(links)

    return tweet, count_mentions, count_links


def load_test_dataset():
    text_df = pd.read_json(
        "/home/ubuntu/Profiling-Cryptocurrency-Influencers-with-FSL/submission/test_text.json",
        lines=True)
    text_df = text_df.drop(columns=['tweet ids'])
    expanded_rows = []
    for _, row in text_df.iterrows():
        for text_dict in row['texts']:
            new_row = row.copy()
            new_row['texts'] = text_dict['text']
            expanded_rows.append(new_row)

    df = pd.DataFrame(expanded_rows)
    # Apply the preprocessing function to the 'tweet' column
    df['texts'], df['count_mention'], df['count_link'] = zip(
        *df['texts'].apply(preprocess_tweet))
    
    df = df.groupby('twitter user id').agg(
        {'texts': ' '.join, 'count_mention': sum}).reset_index()

    return df


class TwoBodyModel(nn.Module):
    def __init__(self, model_1, model_2, num_labels):
        super(TwoBodyModel, self).__init__()
        self.num_labels = num_labels
        self.model_1 = AutoModel.from_pretrained(model_1)
        self.model_2 = AutoModel.from_pretrained(model_2)

        self.dropout_1 = nn.Dropout(0.2)
        self.dropout_2 = nn.Dropout(0.2)
        self.pre_classifier_1 = nn.Linear(
            self.model_1.config.hidden_size, self.model_1.config.hidden_size)
        self.pre_classifier_2 = nn.Linear(
            self.model_2.config.hidden_size, self.model_2.config.hidden_size)
        self.dropout_1_2 = nn.Dropout(0.1)
        self.dropout_2_2 = nn.Dropout(0.1)
        self.classifier_1 = nn.Linear(self.model_1.config.hidden_size+self.model_2.config.hidden_size,
                                      self.model_1.config.hidden_size+self.model_2.config.hidden_size)
        self.dropout_3 = nn.Dropout(0.2)
        self.classifier_2 = nn.Linear(
            self.model_1.config.hidden_size+self.model_2.config.hidden_size, num_labels)

    def forward(self, input_ids_1, input_ids_2, attention_mask_1, attention_mask_2):

        output_1 = self.model_1(input_ids=input_ids_1,
                                attention_mask=attention_mask_1)
        output_2 = self.model_2(input_ids=input_ids_2,
                                attention_mask=attention_mask_2)

        pre_output_1 = self.dropout_1(F.gelu(output_1[0][:, 0, :]))
        pre_output_2 = self.dropout_2(F.gelu(output_2[0][:, 0, :]))

        pre_output_1 = self.pre_classifier_1(pre_output_1)
        pre_output_1 = self.dropout_1_2(F.gelu(pre_output_1))
        pre_output_2 = self.pre_classifier_2(pre_output_2)
        pre_output_2 = self.dropout_2_2(F.gelu(pre_output_2))
        output = self.classifier_1(torch.cat((pre_output_1, pre_output_2), 1))

        output = self.dropout_3(F.gelu(output))
        logits = self.classifier_2(output)

        return logits


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, max_len, tweet_df):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tweets_dataset = tweet_df

    def __len__(self):
        return len(self.tweets_dataset)

    def __getitem__(self, idx):
        tweet = self.tweets_dataset.iloc[idx]['texts']
        user_id = self.tweets_dataset.iloc[idx]['twitter user id']
        encoding = self.tokenizer(
            text=tweet,
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
            # 'label': torch.tensor(labels_matrix, dtype=torch.float),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'user_id': user_id
        }


def eval_model(model, data_loader_1, data_loader_2, device):
    result = []
    with torch.no_grad():
        softmax = nn.Softmax(dim=1)
        for i, d in enumerate(zip(data_loader_1, data_loader_2)):
            d_1, d_2 = d[0], d[1]
            input_ids_1 = d_1["input_ids"].reshape(
                d_1["input_ids"].shape[0], max_tweet_len).to(device)
            attention_mask_1 = d_1["attention_mask"].to(device)
            users = d_1["user_id"]

            input_ids_2 = d_2["input_ids"].reshape(
                d_2["input_ids"].shape[0], max_tweet_len).to(device)
            attention_mask_2 = d_2["attention_mask"].to(device)

            outputs = model(input_ids_1=input_ids_1, input_ids_2=input_ids_2,
                            attention_mask_1=attention_mask_1, attention_mask_2=attention_mask_2)
            
            _, prediction = torch.max(outputs, dim=1)
            probability,_ = torch.max(softmax(outputs), dim=1)
            prediction = prediction.flatten()
            probability = probability.flatten()
            prediction_label = [id2label[p.item()] for p in prediction]
            for user, pred, prob in zip(users, prediction_label, probability.tolist()):
                result.append((user, pred, prob))
    return result


def main():
    tokenizer_1 = AutoTokenizer.from_pretrained(model_name_1)
    tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2)

    # load model
    model = load_model(TwoBodyModel, "cardiffnlp/twitter-roberta-large-2022-154m", "Twitter/twhin-bert-base", 5,
                       "/home/ubuntu/Profiling-Cryptocurrency-Influencers-with-FSL/abhinav/submission/model_epoch_20.pth")

    # load data
    test_df = load_test_dataset()
    test_dataset_1 = TweetDataset(tokenizer_1, max_tweet_len, test_df)
    test_dataset_2 = TweetDataset(tokenizer_2, max_tweet_len, test_df)
    test_data_loader_1 = DataLoader(
        test_dataset_1, batch_size=16, shuffle=True, num_workers=4)
    test_data_loader_2 = DataLoader(
        test_dataset_2, batch_size=16, shuffle=True, num_workers=4)

    # predict
    result = eval_model(model, test_data_loader_1, test_data_loader_2, device)
    result_df = pd.DataFrame(result, columns=['twitter user id', 'class', 'probability'])

    with open('predictions.json', 'w') as f:
        for record in result_df.to_dict(orient='records'):
            f.write(json.dumps(record))
            f.write('\n')


if __name__ == '__main__':
    main()
