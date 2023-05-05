from transformers import EvalPrediction
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import numpy as np
from transformers import TrainerCallback



def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result

# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

class PrintMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and state.is_local_process_zero and 'epoch' in logs:
            print(f"Epoch: {logs['epoch']:.2f}")
            if 'loss' in logs:
                print(f"Loss: {logs['loss']:.4f}")
            if 'eval_accuracy' in logs:
                print(f"Accuracy: {logs['eval_accuracy']:.4f}")
            if 'eval_f1' in logs:
                print(f"F1 Score: {logs['eval_f1']:.4f}")
            print("\n")