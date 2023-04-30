import pandas as pd

# for prompt template 1- pass only the tweet row
def prompt_template_1_train(tweet_data, label):
  '''
  Template: Given a set of user tweets [x1],[x2],[x3],[x4],[x5],
  the user profile is labelled as [z].
  '''
  text = "Given a set of user tweets-"
  for tweets in tweet_data:
    no = 1
    for tweet in tweets:
      text+=", "+str(tweet)+" "
      no+=1
  text+=",-the user profile is labelled as "+label
  return text

# Masked labels for testing
def prompt_template_1_test(tweet_data):
  '''
  Template: Given a set of user tweets [x1],[x2],[x3],[x4],[x5],
  the user profile is labelled as [z].
  '''
  text = "Given a set of user tweets-"
  no = 1
  for tweet in tweet_data:
    text+=", "+str(tweet)+" "
    no+=1
  text+=",-the user profile is labelled as [MASK]"
  return text

# for prompt template 2- pass the whole dataframe (Binary Prompt)
def prompt_template_2(text_dataframe, labels):
  '''
  Q&A form of template
  Template: Given a set of user tweets [x1],[x2],[x3],[x4],[x5],
  the user profile is labelled as [z].
  Label: True/False
  * Binary Classification Prompt
  '''
  label_list = ['micro', 'no influencer', 'mega', 'macro', 'nano']
  text = []
  binary_labels = []
  for tweets, label in zip(text_dataframe, labels):
    tweet_text = "Given a set of user tweets-"
    no = 1
    for tweet in tweets:
      tweet_text+=", "+str(tweet)+" "
      no+=1
    positive_text = tweet_text+",-the user profile is labelled as "+label
    positive_label = True
    text.append(positive_text)
    binary_labels.append(positive_label)

    for neg_label in label_list:
      if neg_label!=label:
        negative_text = tweet_text+",-the user profile is labelled as "+neg_label
        text.append(negative_text)
        binary_labels.append(False)
  
  return text, binary_labels