from transformers import AutoTokenizer
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import json
from collections import defaultdict
import datasets

np.random.seed(7)

if __name__ == '__main__':
    # tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    label_dict = {}
    hiera = defaultdict(set)
    data = datasets.load_dataset('json', data_files='wos_total.json')['train']
    for l in data['doc_label']:
        if l[0] not in label_dict:
            label_dict[l[0]] = len(label_dict)
    for l in data['doc_label']:
        assert len(l) == 2
        if l[1] not in label_dict:
            label_dict[l[1]] = len(label_dict)
        hiera[label_dict[l[0]]].add(label_dict[l[1]])
    value_dict = {i: v for v, i in label_dict.items()}
    torch.save(value_dict, 'value_dict.pt')
    torch.save(hiera, 'slot.pt')

    id = [i for i in range(len(data))]
    np_data = np.array(id)
    np.random.shuffle(id)
    np_data = np_data[id]
    train, test = train_test_split(np_data, test_size=0.2, random_state=0)
    train, val = train_test_split(train, test_size=0.2, random_state=0)
    train = train.tolist()
    val = val.tolist()
    test = test.tolist()
    with open('WebOfScience_train.json', 'w') as f:
        for i in train:
            line = json.dumps({'token': data[i]['doc_token'], 'label': [label_dict[i] for i in data[i]['doc_label']]})
            f.write(line + '\n')
    with open('WebOfScience_dev.json', 'w') as f:
        for i in val:
            line = json.dumps({'token': data[i]['doc_token'], 'label': [label_dict[i] for i in data[i]['doc_label']]})
            f.write(line + '\n')
    with open('WebOfScience_test.json', 'w') as f:
        for i in test:
            line = json.dumps({'token': data[i]['doc_token'], 'label': [label_dict[i] for i in data[i]['doc_label']]})
            f.write(line + '\n')
