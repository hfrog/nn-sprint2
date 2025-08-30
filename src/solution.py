#!/usr/bin/env python3

import sys
import os
import re
import random
import evaluate
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from lstm_model import LSTM
from lstm_train import train
from data_utils import TextDataset, Tokenizer, read_datafile, save_datafile, clean_string
from eval_transformer_pipeline import eval_transformer

basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datadir = os.path.join(basedir, 'data')

#DATAFILE = os.path.join(datadir, 'raw_dataset.csv')
DATAFILE = os.path.join(datadir, 'raw_dataset-500k.csv')
#DATAFILE = os.path.join(datadir, 'raw_dataset-100k.csv')
#DATAFILE = os.path.join(datadir, 'raw_dataset-10k.csv')


TRAIN_PART = 0.8
VAL_PART   = 0.1
TEST_PART  = 0.1


def collate_fn(batch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    texts = [item[0] for item in batch]
    labels = torch.tensor([item[1][0] for item in batch]).to(device)
    lengths = torch.tensor([len(seq) for seq in texts])
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=tokenizer.pad())

    return {
        'input_ids': padded_texts,
        'lengths': lengths,
        'labels': labels
    }


clean_datafile = list(filter(lambda f: len(f.split()) > 3, [clean_string(line) for line in read_datafile(DATAFILE)]))

train_len = int(TRAIN_PART * len(clean_datafile))
val_len = int(VAL_PART * len(clean_datafile))
test_len = int(TEST_PART * len(clean_datafile))

save_datafile(re.sub('.csv', '-full.csv', DATAFILE), clean_datafile)
save_datafile(re.sub('.csv', '-train.csv', DATAFILE), clean_datafile[:train_len])
save_datafile(re.sub('.csv', '-val.csv', DATAFILE), clean_datafile[train_len:train_len+val_len])
save_datafile(re.sub('.csv', '-test.csv', DATAFILE), clean_datafile[train_len+val_len:])

tokenizer = Tokenizer(clean_datafile)
tokenized = [tokenizer.encode(line) for line in clean_datafile]

train_dataset = TextDataset(tokenized[:train_len])
val_dataset = TextDataset(tokenized[train_len:train_len+val_len])
test_dataset = TextDataset(tokenized[train_len+val_len:])

#BATCH_SIZE = 64
BATCH_SIZE = 256

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

HIDDEN_DIM = 128
vocab_size = tokenizer.vocab_size()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device {device}')
model = LSTM(vocab_size, HIDDEN_DIM, padding_idx=tokenizer.pad()).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
criterion = nn.CrossEntropyLoss()
rouge = evaluate.load('rouge')

train(model, train_dataloader, val_dataloader, tokenizer, optimizer, criterion, rouge)

with torch.no_grad():
    for batch in val_dataloader:
        inputs = batch['input_ids']
        lengths = batch['lengths']
        labels = batch['labels']

        logits = model(inputs, lengths)
        preds = torch.argmax(logits, dim=1)
        for i in range(len(labels)):
            input = ' '.join(tokenizer.decode(filter(lambda f: f != tokenizer.pad(), inputs[i].tolist())))
            true_output = tokenizer.decode([labels[i].item()])[0]
            pred_output = tokenizer.decode([preds[i].item()])[0]

print('Samples')
for i in range(5):
    rnd = random.randrange(len(test_dataset))
    input = test_dataset[rnd][0].tolist()
    input_str = ' '.join(tokenizer.decode(input))
    output = ' '.join(tokenizer.decode(model.gen_next(input, 16)))
    reference = ' '.join(tokenizer.decode(test_dataset[rnd][1]))
    print(f'{input_str} => {output} ({reference})')

val_datafile = read_datafile(re.sub('.csv', '-val.csv', DATAFILE))
eval_transformer(val_datafile, rouge)
