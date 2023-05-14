
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from compute_metrics import compute_metrics
from TweetDataset import TweetDataset
from transformers import EarlyStoppingCallback

MODELS = [  'tner/roberta-large-tweetner7-all',
            'tner/bertweet-large-tweetner7-all',
            'cardiffnlp/twitter-roberta-large-2022-154m',
            'roberta-large',
            'google/electra-large-discriminator',
]

BATCH_SIZE = 4
EPOCHS = 50
TOKENS = 512

train_df = pd.read_csv('../data/finetune_train_val_test/train.csv')
validate_df = pd.read_csv('../data/finetune_train_val_test/validate.csv')


for model_name in MODELS:
    print(f"Training {model_name}:")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                        num_labels=5,
                                                        problem_type="multi_label_classification",
                                                        ignore_mismatched_sizes=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    train_dataset = TweetDataset(tokenizer, TOKENS, train_df, strategy='concat')
    val_dataset = TweetDataset(tokenizer, TOKENS, validate_df, strategy='concat')

    train_steps_per_epoch = len(train_dataset) // BATCH_SIZE

    training_arguments = TrainingArguments(
        output_dir=f'./results/{model_name}',
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model= "eval_f1",
        save_total_limit = 3,
        warmup_steps=100,
        logging_dir='./logs',
        logging_steps = train_steps_per_epoch,
        disable_tqdm=True
    )

    trainer = Trainer(
        model,
        training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer= tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()

    trainer.save_model(model_name)
    print()
    del(model)



