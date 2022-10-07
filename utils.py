import torch
import random
import numpy as np


def seed_torch(seed=1029):
    print('Set seed to', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def constraint(batch_id, input_ids, label_dict):
    last_token = input_ids[-1].item()
    if last_token not in label_dict:
        ret = [2]
    else:
        ret = [i + 3 for i in label_dict[input_ids[-1].item() - 3]]
    return ret
