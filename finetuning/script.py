# %%
from influencer_profiler import InfluencerProfiler, save_model
from dataloader import create_data_loader
from train_loop import train_loop
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import optim
import torch.nn as nn
import torch

RANDOM_SEED = 20

MODELS = ['charlieoneill/distilbert-base-uncased-finetuned-tweet_eval-offensive', 
         'pig4431/TweetEval_roBERTa_5E', 
         'tner/twitter-roberta-base-dec2021-tweetner7-random',
         'cardiffnlp/tweet-topic-21-multi',
         'cardiffnlp/twitter-roberta-base-2021-124m'
         ]

# too large 'tner/bertweet-large-tweetner7-all',

EPOCHS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
def translate_class_column(df):
    class_map = {
        "no influencer": 0,
        "nano": 1,
        "micro": 2,
        "macro": 3,
        "mega": 4
    }
    
    if "class" not in df.columns:
        print("Error: DataFrame does not contain a column called 'class'")
        return
    
    df["class"] = df["class"].apply(lambda x: class_map.get(x, x))
    
    return df

df = pd.read_csv('../data/preprocessed_data.csv')
df = translate_class_column(df)

df_train_val, df_test = train_test_split(df, test_size=100, random_state=RANDOM_SEED)
df_train, df_val = train_test_split(df_train_val, test_size=100, random_state=RANDOM_SEED)

# %%
for model_name in MODELS:    
    print(f'Training {model_name}')
    # Retrieve the tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create data loader
    train_data_loader = create_data_loader(df_train, tokenizer, 300, 8)
    val_data_loader = create_data_loader(df_val, tokenizer, 300, 8)
    test_data_loader = create_data_loader(df_test, tokenizer, 300, 8)

    # Create model
    model = InfluencerProfiler(model= model_name, n_classes=5).to(device)
    model.requires_grad_embeddings(True)

    # Set training parameters
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=0,
                    num_training_steps=total_steps
                )

    # Set loss function
    loss_fn = nn.CrossEntropyLoss().to(device)

    # Train model
    model = train_loop(EPOCHS, train_data_loader, df_train, val_data_loader, df_val, model, loss_fn, optimizer, device, scheduler)

    # Export model
    save_as = model_name.split('/', 1)[1]
    save_model(model, f'{save_as}.pth')

    # Free GPU mem from model
    del(model)
    torch.cuda.empty_cache()
    print()