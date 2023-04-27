import pandas as pd
import numpy as np

def split_rows(dataframe):
  for index, row in dataframe.iterrows():
    if row['no.of_words']>1000:
      dataframe.loc[len(dataframe)] = {'twitter user id': row['twitter user id'],
                                       'texts': row['texts'][:5], 'class': row['class'],
                                       'no.of_tweets': 5, 'no.of_words': np.sum([len(y) for y in row['texts'][:5]])}
      dataframe.loc[len(dataframe)] = {'twitter user id': row['twitter user id'],
                                       'texts': row['texts'][5:], 'class': row['class'],
                                       'no.of_tweets': row['no.of_tweets']-5, 
                                       'no.of_words': np.sum([len(y) for y in row['texts'][5:]])}
  dataframe = dataframe[dataframe['no.of_words']<=1000]
  return dataframe

def preprocess_for_prompt(dataframe):
  dataframe = dataframe.drop(dataframe.columns[[3]], axis=1)
  dataframe = dataframe.groupby("twitter user id").agg(lambda x: list(x))
  dataframe['class'] = dataframe['class'].apply(lambda x: x[0])
  dataframe['no.of_tweets'] = dataframe['texts'].apply(lambda x: len(x))
  dataframe['no.of_words'] = dataframe['texts'].apply(lambda x: np.sum([len(y) for y in x]))
  dataframe.reset_index(inplace=True)
  dataframe = split_rows(dataframe)
  dataframe = dataframe[['twitter user id','texts','class']]
  return dataframe