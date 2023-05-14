import pandas as pd
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

def predict(tweet, model, tokenizer):
    model.eval()
    inputs = tokenizer.encode_plus(
        tweet,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, predicted = torch.max(logits, 1)
        return predicted.item()

def convert_id2label(label):
    label2id = {
        "no influencer": 0,
        "nano": 1,
        "micro": 2,
        "macro": 3,
        "mega": 4
    }
    return label2id[label]

def plot_confusion_matrix(conf_matrix, labels, acc, f1, title='Confusion Matrix'):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    ax.set_title(title, y=1.05, fontsize=14)
    subtitle = f'Accuracy: {acc:.3f} | F1 Score: {f1:.3f}'
    ax.text(0.5, 1.01, subtitle, fontsize=12, ha='center', transform=ax.transAxes)

    plt.show()

def print_result(df, model, model_name, tokenizer):
    df['predictions']=df['texts'].apply(lambda x: predict(x,model,tokenizer))
    df['actual'] = df['class'].apply(lambda x: convert_id2label(x))
    y_pred = df['predictions']
    y_true = df['actual']
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    plot_confusion_matrix(cm, ["no influencer", "nano", "micro", "macro", "mega"], acc, f1, model_name)
