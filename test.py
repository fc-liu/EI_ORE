from config import config
from dataloader.dataset import EntityIntervenDataset, NlgEntityIntervenDataset, ContextInterveneDataset, T_RExDataset, NYT_FBDataset
from transformers import BertTokenizer, BertModel
from pytorch_metric_learning import losses, distances
import torch
from torch.utils.data import DataLoader
from model.relation_representation_model import RCModel
from torch import optim
import torch.nn as nn
from torch.nn.functional import pairwise_distance
import os
from tqdm import tqdm
from kmeans_pytorch import kmeans
import eval
import prettytable as pt
from kmeans_pytorch import kmeans
import numpy as np
import random
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
from torch.nn.functional import margin_ranking_loss

ckpt_file_path = "./checkpoint/pretrain/{}".format(config.ckpt_name)
tokenizer = BertTokenizer.from_pretrained(
    config.bert_model, do_basic_tokenize=False)
bert_model = BertModel.from_pretrained(config.bert_model)
tokenizer.add_special_tokens(
    {"additional_special_tokens": [config.e11, config.e12, config.e21, config.e22]})
bert_model.resize_token_embeddings(len(tokenizer))

model = RCModel(bert_model, cls_num=config.num_cls)
if os.path.exists(ckpt_file_path):
    if config.cudas[0] >= 0:
        ckpt = torch.load(ckpt_file_path, map_location=lambda storage,
                          loc: storage.cuda(config.cudas[0]))
    else:
        ckpt = torch.load(
            ckpt_file_path, map_location=lambda storage, loc: storage.cpu())
    model.load_state_dict(ckpt["state_dict"])
    print("######################load fewrel full model from {}#######################".format(
        ckpt_file_path))

if len(config.cudas) > 0:
    model = nn.DataParallel(
        model, device_ids=config.cudas)


tsne = TSNE(n_jobs=4)

best_avg_score = 0


def evaluate(model, dataset, epoch):
    global best_avg_score
    model.eval()
    dataloader = DataLoader(dataset, shuffle=False,
                            batch_size=32, num_workers=2, collate_fn=dataset.collate_fn, pin_memory=True)
    seed = config.seed
    np.random.seed(seed)
    random.seed(seed)
    embeds = []
    labels = []
    for idx, data_batch in enumerate(tqdm(dataloader, postfix="evaluate")):
        tokens, pos1, pos2, mask, rel_id, raw_tokens = data_batch
        _, rep = model(
            *(ts.to(config.cudas[0]) for ts in (tokens, pos1, pos2, mask)))
        embeds.append(rep.detach().cpu())
        labels.extend(rel_id)
        if (idx+1) % 3 == 0:
            torch.cuda.empty_cache()
    embeds = torch.cat(embeds)
    pred_idx, _ = kmeans(X=embeds, num_clusters=config.num_cls,
                         device=torch.device(config.cudas[0]))

    b3_p, b3_r, b3_f1 = eval.bcubed_sparse(labels, pred_idx)
    hom, com, vf = eval.v_measure(labels, pred_idx)
    ari = eval.adjusted_rand_score(labels, pred_idx)
    avg_score = round((b3_f1+vf+ari)/2, 4)
    if avg_score > best_avg_score:
        best_avg_score = avg_score
    model.train()
    return b3_p, b3_r, b3_f1, hom, com, vf, ari


if __name__ == "__main__":
    table = pt.PrettyTable(
        ["B3_Prec", "B3_Rec", "B3_F1", "V_Hom.", "V_Comp", "V_F1", "ARI"])

    if config.data_src == "ds":
        trexDataset = T_RExDataset(tokenizer,
                                   "data/t-rex/json-ds-val.json", visual=True)
    elif config.data_src == "spo":
        trexDataset = T_RExDataset(tokenizer,
                                   "data/t-rex/json-spo-val.json", visual=True)

    b3_prec, b3_rec, b3_f1, hom, com, v_f1, ari = evaluate(model, trexDataset)
    table.add_row([b3_prec, b3_rec, b3_f1, hom, com, v_f1, ari])
    print(table)
