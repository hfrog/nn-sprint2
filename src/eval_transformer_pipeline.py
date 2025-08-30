#!/usr/bin/env python3

import random
import torch
from transformers import pipeline, AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
from tqdm import tqdm


MAX_LEN = 64
BATCH_SIZE = 128


def eval_transformer(datafile, rouge):
    val_x = []
    val_labels = []
    for line in datafile:
        splitted = line.split()
        x_len = int(3*len(splitted)/4)
        val_x.append({'text': ' '.join(splitted[:x_len])})
        val_labels.append(' '.join(splitted[x_len:]))
    kds1 = KeyDataset(Dataset.from_list(val_x), 'text')

    model_name = 'distilgpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding_side='left', padding='max_length', max_length=MAX_LEN)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using device {device}')
    generator = pipeline('text-generation', model=model_name, tokenizer=tokenizer, device=device)

    results = []
    i = 0
    for out in tqdm(generator(kds1, batch_size=BATCH_SIZE, max_new_tokens=16, return_full_text=False, pad_token_id=tokenizer.eos_token_id), total=len(kds1)):
        results.append(out[0]['generated_text'].strip())
        i += 1

    metrics = rouge.compute(predictions=results, references=val_labels)
    print(metrics)

    print('Samples:')
    for i in range(5):
        rnd = random.randrange(len(kds1))
        print(f'{val_x[rnd]["text"]} => {results[rnd]} ({val_labels[rnd]})')
