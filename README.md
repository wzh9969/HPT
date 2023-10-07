# Implement of HPT: Hierarchy-aware Prompt Tuning for Hierarchical Text Classification
This repository implements a prompt tuning model for hierarchical text classification. 
This work has been accepted as the long paper [HPT: Hierarchy-aware Prompt Tuning for Hierarchical Text Classification](https://arxiv.org/pdf/2204.13413.pdf)
in EMNLP 2022.

## Requirements

* Python >= 3.6
* torch >= 1.6.0
* transformers >= 4.11.0
* datasets
* torch-geometric == 1.7.2
* torch-scatter == 2.0.8
* torch-sparse == 0.6.12


## Preprocess

Please download the original dataset and then use these scripts.

### Web Of Science

The original dataset can be acquired in [the repository of HDLTex](https://github.com/kk7nc/HDLTex). Preprocessing code could refer to [the repository of HiAGM](https://github.com/Alibaba-NLP/HiAGM) and we provide a copy of preprocessing code here.
Please save the Excel data file `Data.xlsx` in `WebOfScience/Meta-data` as `Data.txt`.

```shell
cd data/WebOfScience
python preprocess_wos.py
python data_wos.py
```

### NYT

The original dataset can be acquired [here](https://catalog.ldc.upenn.edu/LDC2008T19).
Place the **unzipped** folder `nyt_corpus` inside `data/nyt` (or unzip `nyt_corpus_LDC2008T19.tgz` inside `data/nyt`).

```shell
cd data/nyt
# unzip if necessary
# tar -zxvf nyt_corpus_LDC2008T19.tgz -C ./
python data_nyt.py
```

### RCV1-V2

The preprocessing code could refer to the [repository of reuters_loader](https://github.com/ductri/reuters_loader) and we provide a copy here. The original dataset can be acquired [here](https://trec.nist.gov/data/reuters/reuters.html) by signing an agreement.
Place `rcv1.tar.xz` and `lyrl2004_tokens_train.dat` (can be downloaded [here](https://jmlr.csail.mit.edu/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_train.dat.gz)) inside `data/rcv1`.

```shell
cd data/rcv1
python preprocess_rcv1.py ./
python data_rcv1.py
```

## Train

```
usage: train.py [-h] [--lr LR] [--data DATA] [--batch BATCH] [--early-stop EARLY_STOP] [--device DEVICE] --name NAME [--update UPDATE] [--model MODEL] [--wandb] [--arch ARCH] [--layer LAYER] [--graph GRAPH] [--prompt-loss]
                [--low-res] [--seed SEED]

optional arguments:
  -h, --help                show this help message and exit
  --lr LR					Learning rate. Default: 3e-5.
  --data {WebOfScience,nyt,rcv1} Dataset.
  --batch BATCH             Batch size.
  --early-stop EARLY_STOP   Epoch before early stop.
  --device DEVICE           cuda or cpu. Default: cuda.
  --name NAME               A name for different runs.
  --update UPDATE           Gradient accumulate steps.
  --wandb                   Use wandb for logging.
  --seed SEED               Random seed.
```

Checkpoints are in `./checkpoints/DATA-NAME`. Two checkpoints are kept based on macro-F1 and micro-F1 respectively 
(`checkpoint_best_macro.pt`, `checkpoint_best_micro.pt`).

**Example:**
```shell
python train.py --name test --batch 16 --data WebOfScience
```

### Reproducibility

We experiment on one GeForce RTX 3090 GPU (24G) with CUDA version $11.2$. We use a batch size of $16$ to fully tap one GPU.

The model is trained for around 20 epochs before an early stop with ~10 min/epoch.

Our model has no extra hyperparameters: all hyperparameters follow previous works and have not been tuned.

Checkpoints for each dataset can be downloaded [here](https://drive.google.com/drive/folders/1j1PMzo4YLG8oUAnuolvmfn-dA9A43yIS?usp=sharing). Place the `checkpoints` folder inside the main folder (`HPT/checkpoints`). These results are reported in the main experiment.

| Dataset        | Macro-F1           | Micro-F1           |
| -------------- | ------------------ | ------------------ |
| NYT            | 0.7041934624814794 | 0.8041512855978236 |
| RCV1-V2        | 0.6953327068021089 | 0.8726110320904367 |
| Web Of Science | 0.8192644031945633 | 0.8715855067014047 |

## Test

```
usage: test.py [-h] [--device DEVICE] [--batch BATCH] --name NAME [--extra {_macro,_micro}]

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE
  --batch BATCH         Batch size.
  --name NAME           Name of checkpoint. Commonly as DATA-NAME.
  --extra {_macro,_micro}
                        An extra string in the name of checkpoint. Default: _macro
```

Use `--extra _macro` or `--extra _micro`  to choose from using `checkpoint_best_macro.pt` or`checkpoint_best_micro.pt` respectively.

e.g. Test on previous example.

```shell
python test.py --name WebOfScience-test --batch 64
```

Test on provided checkpoints:

```shell
python test.py --name WebOfScience-HPT --batch 64
python test.py --name rcv1-HPT --batch 64
python test.py --name nyt-HPT --batch 64
```

# Citation

```
@inproceedings{wang-etal-2022-hpt,
    title = "{HPT}: Hierarchy-aware Prompt Tuning for Hierarchical Text Classification",
    author = "Wang, Zihan  and
      Wang, Peiyi  and
      Liu, Tianyu  and
      Lin, Binghuai  and
      Cao, Yunbo  and
      Sui, Zhifang  and
      Wang, Houfeng",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.246",
    doi = "10.18653/v1/2022.emnlp-main.246",
    pages = "3740--3751",
}

```
