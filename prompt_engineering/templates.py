import pandas as pd

# for prompt template 1- pass the entire dataframe
def prompt_template_1_train(tweet_data, labels):
  '''
  Template: Given a set of user tweets [x1],[x2],[x3],[x4],[x5],
  the user profile is labelled as [z].
  '''
  prompt_true = []
  prompt_mask = []
  for tweets, label in zip(tweet_data,labels):
    text = "Given a set of user tweets-"
    for tweet in tweets:
      text+=", "+str(tweet)+" "
    prompt_true_sub = text+",-the user profile is labelled as "+label
    prompt_mask_sub = text+",-the user profile is labelled as [MASK]"
    prompt_true.append(prompt_true_sub)
    prompt_mask.append(prompt_mask_sub)
  return prompt_true, prompt_mask

# Masked labels for testing
def prompt_template_1_test(tweet_data):
  '''
  Template: Given a set of user tweets [x1],[x2],[x3],[x4],[x5],
  the user profile is labelled as [z].
  '''
  for tweet in tweet_data:
    text = "Given a set of user tweets-"
    text+=", "+str(tweet)+" "
  text+=",-the user profile is labelled as [MASK]."
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
    for tweet in tweets:
      tweet_text+=", "+str(tweet)+" "
    positive_text = tweet_text+",-the user profile is labelled as "+label
    positive_label = 0
    text.append(positive_text)
    binary_labels.append(positive_label)

    for neg_label in label_list:
      if neg_label!=label:
        negative_text = tweet_text+",-the user profile is labelled as "+neg_label
        text.append(negative_text)
        binary_labels.append(1)
  
  return text, binary_labels