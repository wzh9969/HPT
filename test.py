import torch
import os
from transformers import AutoTokenizer
import datasets
from tqdm import tqdm
from torch.utils.data import DataLoader

from eval import evaluate
from train import parse
import utils
import random

if __name__ == '__main__':
    utils.seed_torch(3)
    parser = parse()
    parser.add_argument('--extra', type=str, default='_macro')
    args = parser.parse_args()

    checkpoint = torch.load(os.path.join('checkpoints', args.name, 'checkpoint_best{}.pt'.format(args.extra)),
                            map_location='cpu')
    batch_size = args.batch
    data_path = args.data
    extra = args.extra
    args = checkpoint['args'] if checkpoint['args'] is not None else args
    args = parser.parse_args(namespace=args)
    print(args)
    data_path = os.path.join('data', args.data if 'data' in args else data_path)

    tokenizer = AutoTokenizer.from_pretrained(args.arch)

    label_dict = torch.load(os.path.join(data_path, 'value_dict.pt'))
    label_dict = {i: v for i, v in label_dict.items()}

    slot2value = torch.load(os.path.join(data_path, 'slot.pt'))
    value2slot = {}
    num_class = 0
    for s in slot2value:
        for v in slot2value[s]:
            value2slot[v] = s
            if num_class < v:
                num_class = v
    num_class += 1
    path_list = [(i, v) for v, i in value2slot.items()]
    for i in range(num_class):
        if i not in value2slot:
            value2slot[i] = -1


    def get_depth(x):
        depth = 0
        while value2slot[x] != -1:
            depth += 1
            x = value2slot[x]
        return depth


    depth_dict = {i: get_depth(i) for i in range(num_class)}
    max_depth = depth_dict[max(depth_dict, key=depth_dict.get)] + 1
    depth2label = {i: [a for a in depth_dict if depth_dict[a] == i] for i in range(max_depth)}

    for depth in depth2label:
        for l in depth2label[depth]:
            path_list.append((num_class + depth, l))

    if args.model == 'prompt':
        if os.path.exists(os.path.join(data_path, args.model)):
            dataset = datasets.load_from_disk(os.path.join(data_path, args.model))
        else:
            dataset = datasets.load_dataset('json',
                                            data_files={'train': 'data/{}/{}_train.json'.format(args.data, args.data),
                                                        'dev': 'data/{}/{}_dev.json'.format(args.data, args.data),
                                                        'test': 'data/{}/{}_test.json'.format(args.data, args.data), })

            prefix = []
            for i in range(max_depth):
                prefix.append(tokenizer.vocab_size + num_class + i)
                prefix.append(tokenizer.vocab_size + num_class + max_depth)
            prefix.append(tokenizer.sep_token_id)


            def data_map_function(batch, tokenizer):
                new_batch = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'labels': []}
                for l, t in zip(batch['label'], batch['token']):
                    new_batch['labels'].append([[-100 for _ in range(num_class)] for _ in range(max_depth)])
                    for d in range(max_depth):
                        for i in depth2label[d]:
                            new_batch['labels'][-1][d][i] = 0
                        for i in l:
                            if new_batch['labels'][-1][d][i] == 0:
                                new_batch['labels'][-1][d][i] = 1
                    new_batch['labels'][-1] = [x for y in new_batch['labels'][-1] for x in y]

                    tokens = tokenizer(t, truncation=True)
                    new_batch['input_ids'].append(tokens['input_ids'][:-1][:512 - len(prefix)] + prefix)
                    new_batch['input_ids'][-1].extend(
                        [tokenizer.pad_token_id] * (512 - len(new_batch['input_ids'][-1])))
                    new_batch['attention_mask'].append(
                        tokens['attention_mask'][:-1][:512 - len(prefix)] + [1] * len(prefix))
                    new_batch['attention_mask'][-1].extend([0] * (512 - len(new_batch['attention_mask'][-1])))
                    new_batch['token_type_ids'].append([0] * 512)

                return new_batch


            dataset = dataset.map(lambda x: data_map_function(x, tokenizer), batched=True)
            dataset.save_to_disk(os.path.join(data_path, args.model))
        dataset['train'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])
        dataset['dev'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])
        dataset['test'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])

        from models.prompt import Prompt

    else:
        raise NotImplementedError

    checkpoint = torch.load(os.path.join('checkpoints', args.name, 'checkpoint_best{}.pt'.format(extra)),
                            map_location='cpu')
    model = Prompt.from_pretrained(args.arch, num_labels=len(label_dict), path_list=path_list, layer=args.layer,
                                   graph_type=args.graph, data_path=data_path, depth2label=depth2label, )
    model.init_embedding()
    model.load_state_dict(checkpoint['param'])
    model.to('cuda')

    test = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)
    model.eval()
    pred = []
    gold = []
    father_count = 0
    father_false = 0

    with torch.no_grad(), tqdm(test) as pbar:
        for batch in pbar:
            batch = {k: v.to('cuda') for k, v in batch.items()}
            output_ids, logits = model.generate(batch['input_ids'], depth2label=depth2label,)
            for out, g, lo in zip(output_ids, batch['labels'], logits):
                case_logits = torch.zeros(logits.size(-1))
                for i in set(out):
                    if value2slot[i] != -1:
                        if value2slot[i] in out:
                            father_count += 1
                        else:
                            father_false += 1
                pred.append(set([i for i in out]))
                gold.append([])
                g = g.view(-1, num_class)
                for ll in g:
                    for i, l in enumerate(ll):
                        if l == 1:
                            gold[-1].append(i)

    print('path acc', father_count / (father_false + father_count))
    scores = evaluate(pred, gold, label_dict)
    macro_f1 = scores['macro_f1']
    micro_f1 = scores['micro_f1']
    print('macro', macro_f1, 'micro', micro_f1)
