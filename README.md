# Element Intervention for Open Relation Extraction

This repository is the codes for our paper "[Element Intervention for Open Relation Extraction](https://arxiv.org/abs/2106.09558)".
The data sampling in element intervention relies on a private access elasticsearch API of WIKIDATA, so we will sample and store necessary data for reimplementation, the data will be released once ready.

## Training command:
```bash
python train.py --data_src spo --ent_src spo --is_cxt --is_ent
```

## Testing command:
```bash
python test.py --data_src spo 
```
## Dataset
* [TRex(SPO+DS)](https://hadyelsahar.github.io/t-rex/)
* [WEBNLG](https://gitlab.com/shimorina/webnlg-dataset/-/tree/master/release_v3.0/en)

## Reference
T-Rex data process script refers to：<https://github.com/THU-BPM/SelfORE/blob/unrefactored/data_preprocess/trex_to_ttsv.cpp>.

The context generation relies on the repo: <https://github.com/UKPLab/plms-graph2text>
