#!/usr/bin/env python3

import torch


def MyEvaluate(model, loader, tokenizer, criterion, rouge):
    model.eval()
    rouge1, rouge2, total = 0, 0, 0
    sum_loss = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch['input_ids']
            lengths = batch['lengths']
            labels = batch['labels']

            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            results = rouge.compute(predictions=tokenizer.decode(preds.tolist()), references=tokenizer.decode(labels.tolist()))
            rouge1 += results['rouge1']
            rouge2 += results['rouge2']
            total += len(labels)
            sum_loss += loss.item()

    avg_loss = sum_loss / len(loader)
    return avg_loss, rouge1/total, rouge2/total
