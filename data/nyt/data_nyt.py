import os
import xml.dom.minidom
from tqdm import tqdm
import json
import re
import tarfile
import shutil
from transformers import AutoTokenizer
from collections import defaultdict
import torch

"""
NYTimes Reference: https://catalog.ldc.upenn.edu/LDC2008T19
"""

sample_ratio = 0.02
train_ratio = 0.7
min_per_node = 200

# tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
source = []
labels = []
label_ids = []
label_dict = {}
sentence_ids = []
hiera = defaultdict(set)
ROOT_DIR = 'Nytimes/'
label_f = 'nyt_label.vocab'


# 2003-07
def read_nyt(id_json):
    f = open(id_json, 'r')
    ids = f.readlines()
    f.close()
    print(ids[:2])
    f = open(label_f, 'r')
    label_vocab_s = f.readlines()
    f.close()
    label_vocab = []
    for label in label_vocab_s:
        label = label.strip()
        label_vocab.append(label)
    id_list = []
    for i in ids:
        id_list.append(int(i[13:-5]))
    print(id_list[:2])
    corpus = []

    for file_name in tqdm(ids):
        xml_path = file_name.strip()
        try:
            sample = {}
            dom = xml.dom.minidom.parse(xml_path)
            root = dom.documentElement
            tags = root.getElementsByTagName('p')
            text = ''
            for tag in tags[1:]:
                text += tag.firstChild.data
            if text == '':
                continue
            source.append(text)
            sample_label = []
            tags = root.getElementsByTagName('classifier')
            for tag in tags:
                type = tag.getAttribute('type')
                if type != 'taxonomic_classifier':
                    continue
                hier_path = tag.firstChild.data
                hier_list = hier_path.split('/')
                if len(hier_list) < 3:
                    continue
                for l in range(1, len(hier_list) + 1):
                    label = '/'.join(hier_list[:l])
                    if label == 'Top':
                        continue
                    if label not in sample_label and label in label_vocab:
                        sample_label.append(label)
            labels.append(sample_label)
            sentence_ids.append(file_name)
            sample['doc_topic'] = []
            sample['doc_keyword'] = []
            corpus.append(sample)
        except AssertionError:
            print(xml_path)
            print('Something went wrong...')
            continue


if __name__ == '__main__':
    files = os.listdir('nyt_corpus/data')
    for year in files:
        month = os.listdir(os.path.join('nyt_corpus/data', year))
        for m in month:
            if os.path.isdir(os.path.join('nyt_corpus/data', year, m)):
                new_month = os.listdir(os.path.join('nyt_corpus/data', year, m))
                for nm in new_month:
                    f = tarfile.open(os.path.join('nyt_corpus/data', year, m, nm))
                    f.extractall(os.path.join('Nytimes', year))
            else:
                f = tarfile.open(os.path.join('nyt_corpus/data', year, m))
                f.extractall(os.path.join('Nytimes', year))
    files = os.listdir('Nytimes')
    for year in files:
        month = os.listdir(os.path.join('Nytimes', year))
        for m in month:
            days = os.listdir(os.path.join('Nytimes', year, m))
            for d in days:
                file = os.listdir(os.path.join('Nytimes', year, m, d))
                for f in file:
                    shutil.move(os.path.join('Nytimes', year, m, d, f), os.path.join('Nytimes', year, f))
    read_nyt('idnewnyt_train.json')
    read_nyt('idnewnyt_val.json')
    read_nyt('idnewnyt_test.json')
    rev_dict = {}
    max_length = 0
    for l in labels:
        max_length = max(max_length, len(l))
        for l_ in l:
            split = l_.split('/')
            if l_ not in label_dict:
                label_dict[l_] = len(label_dict)
            for i in range(1, len(split) - 1):
                hiera[label_dict['/'.join(split[:i + 1])]].add(label_dict['/'.join(split[:i + 2])])
                assert '/'.join(split[:i + 2]) not in rev_dict or rev_dict['/'.join(split[:i + 2])] == '/'.join(
                    split[:i + 1])
                rev_dict['/'.join(split[:i + 2])] = '/'.join(split[:i + 1])
    print(max_length)

    value_dict = {i: v.split('/')[-1] for v, i in
                  label_dict.items()}
    torch.save(value_dict, 'value_dict.pt')
    torch.save(hiera, 'slot.pt')

    train_split = open('idnewnyt_train.json', 'r').readlines()
    dev_split = open('idnewnyt_val.json', 'r').readlines()
    test_split = open('idnewnyt_test.json', 'r').readlines()
    train, test, val = [], [], []
    for i in range(len(sentence_ids)):
        if sentence_ids[i] in train_split:
            train.append(i)
        elif sentence_ids[i] in dev_split:
            val.append(i)
        elif sentence_ids[i] in test_split:
            test.append(i)
        else:
            raise RuntimeError

    with open('nyt_train.json', 'w') as f:
        for i in train:
            line = json.dumps({'token': source[i], 'label': [label_dict[i] for i in labels[i]]})
            f.write(line + '\n')

    with open('nyt_dev.json', 'w') as f:
        for i in val:
            line = json.dumps({'token': source[i], 'label': [label_dict[i] for i in labels[i]]})
            f.write(line + '\n')

    with open('nyt_test.json', 'w') as f:
        for i in test:
            line = json.dumps({'token': source[i], 'label': [label_dict[i] for i in labels[i]]})
            f.write(line + '\n')
