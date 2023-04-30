import torch.nn as nn
import torch
import numpy as np


def train_loop(epochs,train_data_loader, val_data_loader, model, loss_fn, optimizer, device, scheduler):
  for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    print('-' * 10)

    train_acc, train_loss, val_acc, val_loss = train_epoch(
        model,
        train_data_loader,
        len(train_data_loader.dataset),
        val_data_loader,
        len(val_data_loader.dataset),
        loss_fn, 
        optimizer, 
        device, 
        scheduler, 
    )
    print(f'Train loss {train_loss} accuracy {train_acc}')
    print(f'Val loss   {val_loss} accuracy {val_acc}')
    print()
  
  return model

def train_epoch(model, train_data_loader, n_train_examples, val_data_loader, n_val_examples, loss_fn, optimizer, device, scheduler):
  model = model.train()

  train_loss = []
  train_correct_predictions = 0
  
  for i,d in enumerate(train_data_loader):
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["label"].to(device)

    outputs = model(
      input_ids = input_ids,
      attention_mask = attention_mask,
    )

    loss = loss_fn(outputs, targets)
    train_loss.append(loss.item())
    loss.backward()

    predictions = outputs.argmax(dim=1)
    train_correct_predictions += torch.sum(predictions == targets)

    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
  val_loss = []
  val_correct_predictions = 0

  with torch.no_grad():
    model.eval()
    for i, d in enumerate(val_data_loader):
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["label"].to(device)

      outputs = model(
        input_ids = input_ids,
        attention_mask = attention_mask
      )

      loss = loss_fn(outputs, targets)
      val_loss.append(loss.item())

      predictions = outputs.argmax(dim=1)
      val_correct_predictions += torch.sum(predictions == targets)


  return (train_correct_predictions.double() / n_train_examples), np.mean(train_loss), (val_correct_predictions.double()/n_val_examples), np.mean(val_loss)