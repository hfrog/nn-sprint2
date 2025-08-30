#!/usr/bin/env python3

import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, **kwargs):
        super().__init__()
        self.padding_idx = kwargs.get('padding_idx')
        self.embedding = nn.Embedding(vocab_size, hidden_dim, self.padding_idx)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, lengths):
        emb = self.embedding(input_ids)
        packed_in = torch.nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed_in)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        hidden_out = out[:, -1, :]
        linear_out = self.fc(hidden_out)
        return linear_out

    def gen_next(self, ids, n=1):
        input_len = len(ids)
        for _ in range(n):
            out = self.forward(torch.tensor([ids]), torch.tensor([len(ids)]))
            next_id = torch.argmax(out).tolist()
            ids.append(next_id)
        return ids[input_len:]
