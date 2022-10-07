from transformers import AutoTokenizer
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import xml.dom.minidom
import pandas as pd
import json
from collections import defaultdict

np.random.seed(7)

if __name__ == '__main__':
    source = []
    labels = []
    label_dict = {}
    hiera = defaultdict(set)
    with open('rcv1.taxonomy', 'r') as f:
        label_dict['Root'] = -1
        for line in f.readlines():
            line = line.strip().split('\t')
            for i in line[1:]:
                if i not in label_dict:
                    label_dict[i] = len(label_dict) - 1
                hiera[label_dict[line[0]]].add(label_dict[i])
        label_dict.pop('Root')
        hiera.pop(-1)
    value_dict = {i: v for v, i in label_dict.items()}
    torch.save(value_dict, 'value_dict.pt')
    torch.save(hiera, 'slot.pt')
    data = pd.read_csv('rcv1_v2.csv')
    for i, line in tqdm(data.iterrows()):
        dom = xml.dom.minidom.parseString(line['text'])
        root = dom.documentElement
        tags = root.getElementsByTagName('p')
        text = ''
        for tag in tags:
            text += tag.firstChild.data
        if text == '':
            continue
        # text = tokenizer.encode(text.lower(), truncation=True)
        source.append(text)
        l = line['topics'].split('\'')
        labels.append([label_dict[i] for i in l[1::2]])
    print(len(labels))

    data = pd.read_csv('rcv1_v2.csv')
    ids = []
    for i, line in tqdm(data.iterrows()):
        dom = xml.dom.minidom.parseString(line['text'])
        root = dom.documentElement
        tags = root.getElementsByTagName('p')
        cont = False
        for tag in tags:
            if tag.firstChild.data != '':
                cont = True
                break
        if cont:
            ids.append(line['id'])
    train_ids = []
    with open('lyrl2004_tokens_train.dat', 'r') as f:
        for line in f.readlines():
            if line.startswith('.I'):
                train_ids.append(int(line[3:-1]))

    train_ids = set(train_ids)
    train = []
    test = []
    for i in range(len(ids)):
        if ids[i] in train_ids:
            train.append(i)
        else:
            test.append(i)
    id = [i for i in range(len(train))]
    np_data = np.array(train)
    np.random.shuffle(id)
    np_data = np_data[id]
    train, val = train_test_split(train, test_size=0.1, random_state=0)
    # torch.save({'train': train, 'val': val, 'test': test}, 'split.pt')

    with open('rcv1_train.json', 'w') as f:
        for i in train:
            line = json.dumps({'token': source[i], 'label': labels[i]})
            f.write(line + '\n')
    with open('rcv1_dev.json', 'w') as f:
        for i in val:
            line = json.dumps({'token': source[i], 'label': labels[i]})
            f.write(line + '\n')
    with open('rcv1_test.json', 'w') as f:
        for i in test:
            line = json.dumps({'token': source[i], 'label': labels[i]})
            f.write(line + '\n')

