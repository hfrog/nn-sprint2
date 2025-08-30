#!/usr/bin/env python3

from tqdm import tqdm
from eval_lstm import MyEvaluate

N_EPOCHS = 1

def train(model, train_dataloader, val_dataloader, tokenizer, optimizer, criterion, rouge):
    for epoch in range(N_EPOCHS):
        model.train()
        train_loss = 0.
        for batch in tqdm(train_dataloader):
            inputs = batch['input_ids']
            lengths = batch['lengths']
            labels = batch['labels']

            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        val_loss, val_rouge1, val_rouge2 = MyEvaluate(model, val_dataloader, tokenizer, criterion, rouge)
        print(f'Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Rouge1: {val_rouge1:.2%} | Val Rouge2: {val_rouge2:.2%}')
