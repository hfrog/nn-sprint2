#!/usr/bin/env python3

import re
import torch
from torch.utils.data import Dataset, DataLoader


def read_datafile(datafile):
  with open(datafile) as f:
    lines = [line.strip() for line in f]
  return lines


def save_datafile(datafile, texts):
  with open(datafile, 'w') as f:
    for line in texts:
      f.write(line + "\n")


def clean_string(text):
    text = re.sub(r'https?[:/]+\S+', '', text)  # Remove URLs http/https
    text = re.sub(r' www\.\S+', '', text)       # Remove URLs www
    text = re.sub(r'@\w+', '', text)            # Remove user mentions
    text = re.sub(r'#', '', text)               # Remove hashtag symbol but keep the word
    text = re.sub(r'[^\w\s]', '', text)         # Remove punctuation
    text = text.strip()                         # Strip spaces from both ends
    text = re.sub(r'\s\s*', ' ', text)          # Replace multiple spaces with one
    text = text.lower()                         # Convert text to lowercase
    return text


PAD = '<PAD>'
UNK = '<UNK>'

class Tokenizer():
    def __init__(self, texts):
        # build_vocabulary
        all_words = ' '.join(texts).split()
        self.words = [PAD, UNK] + list(set(all_words))  # Список уникальных слов
        self.word_to_idx = {w:i for i,w in enumerate(self.words)}  # Словарь перевода слова в индекс
        self.idx_to_word = {i:w for w,i in self.word_to_idx.items()}  # Обратное отображение индекса в слово

    def encode(self, line):
        return [self.word_to_idx.get(word, self.word_to_idx[UNK]) for word in line.split()]

    def decode(self, a):
        return [self.idx_to_word.get(i, UNK) for i in a]

    def vocab_size(self):
        return len(self.words)

    def pad(self):
        return self.word_to_idx[PAD]


class TextDataset(Dataset):
    def __init__(self, tokenized):
        self.inputs, self.targets = [], []
        for a in tokenized:
            input_len = int(3*len(a)/4)
            self.inputs.append(a[:input_len])
            self.targets.append(a[input_len:])

    def __getitem__(self, idx):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.tensor(self.inputs[idx]).to(device), self.targets[idx]

    def __len__(self):
        return len(self.inputs)
