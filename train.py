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
# from sklearn.cluster import KMeans
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


def plot(x, y, pred_idx, dims, epoch, avg_score):
    x = np.asarray(x)
    y = np.asarray(y)
    pred_idx = np.asarray(pred_idx)

    X_tsne = tsne.fit_transform(x)
    X_norm = X_tsne
    plt.figure(figsize=(20, 20))
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y,
                cmap=plt.cm.get_cmap("jet", 25), marker='o')
    plt.xticks([])
    plt.yticks([])
    # plt.clim(-0.5, 9.5)
    plt.savefig('figures/gold-{}-{}.pdf'.format(epoch, avg_score),bbox_inches='tight')

    plt.figure(figsize=(20, 20))
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=pred_idx,
                cmap=plt.cm.get_cmap("jet", 25), marker='o')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('figures/pred-{}-{}.pdf'.format(epoch, avg_score),bbox_inches='tight')


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


def causal_pretrain(model):
    if config.data_src == "ds":
        trexDataset = T_RExDataset(tokenizer,
                                   "data/t-rex/json-ds-val.json", visual=True)
    elif config.data_src == "spo":
        trexDataset = T_RExDataset(tokenizer,
                                   "data/t-rex/json-spo-val.json", visual=True)
    contextDataset = ContextInterveneDataset(
        tokenizer, "data/pretrain.json")
    if config.ent_src == "spo":
        entityDataset = EntityIntervenDataset(tokenizer,
                                              "data/t-rex/json-spo-val.json", "data/t-rex/entity.json")
    elif config.ent_src == "nlg":
        entityDataset = NlgEntityIntervenDataset(
            tokenizer, contextDataset.data_list, "data/t-rex/entity.json")
    elif config.ent_src=="ds":
        entityDataset = EntityIntervenDataset(tokenizer,
                                              "data/t-rex/json-ds-val.json", "data/t-rex/entity.json")
    parameters_to_optimize = model.parameters()

    optimizer = optim.Adam(parameters_to_optimize,
                           config.learning_rate, weight_decay=config.l2_reg_lambda)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_step_size, gamma=config.decay_rate)
    cxt_loss_fn = nn.TripletMarginLoss(margin=2)
    alpha = 10
    cxt_labels = torch.Tensor([0, 0, 0, 1, 0, 1, 1])
    cxt_indices_tuple = ([0], [1, 2, 4], [3, 5, 6])
    ent_y = torch.Tensor([-1, -1, -1]).to(config.cudas[0])
    model.train()
    best_acc = 0

    table = pt.PrettyTable(
        ["B3_Prec", "B3_Rec", "B3_F1", "V_Hom.", "V_Comp", "V_F1", "ARI"])

    for epoch in range(config.epochs):

        entityDataloader = DataLoader(
            entityDataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
        contextDataloader = DataLoader(
            contextDataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
        tq = tqdm(zip(entityDataloader, contextDataloader))
        i = 0
        for entity_batch, context_batch in tq:
            if i % 3200 == 0:
                b3_prec, b3_rec, b3_f1, hom, com, v_f1, ari = evaluate(
                    model, trexDataset, epoch)
                table.add_row([round(b3_prec, 4), round(b3_rec, 4), round(b3_f1, 4), round(
                    hom, 4), round(com, 4), round(v_f1, 4), round(ari, 4)])
                print(table)
            i += 1
            ent_tokens = entity_batch[-1]
            cxt_tokens = context_batch[-1]

            if config.is_ent:
                _, ent_repre = model(
                    *(ts.squeeze_(0).to(config.cudas[0]) for ts in entity_batch[:-1]))
                anchor_dist = pairwise_distance(ent_repre[0].expand(
                    ent_repre.shape[0]-1, -1), ent_repre[1:])
                ent_loss_lv1 = margin_ranking_loss(
                    anchor_dist[1:2].expand(anchor_dist.size(0)-3), anchor_dist[2:-1], ent_y[:-2], margin=0, reduction="sum")/config.pretrain_batch_size

                ent_oth_loss = margin_ranking_loss(
                    anchor_dist[-2:-1], anchor_dist[-1:], ent_y[-1:], margin=2, reduce="sum")/config.pretrain_batch_size

                ent_loss = (ent_oth_loss+ent_loss_lv1)/3

            else:
                ent_loss = torch.tensor(0)

            if config.is_cxt:
                _, cxt_repre = model(
                    *(ts.squeeze_(0).to(config.cudas[0]) for ts in context_batch[:-1]))
                pos_embs = cxt_repre[cxt_indices_tuple[1]]
                neg_embs = cxt_repre[cxt_indices_tuple[-1]]
                pos_n = pos_embs.shape[0]
                neg_n = neg_embs.shape[0]
                if pos_n != neg_n:
                    raise Exception(
                        "positive number not equal negative number")
                anchor_embs = cxt_repre[[0]].expand(pos_n, -1)
                cxt_loss = cxt_loss_fn(anchor_embs, pos_embs, neg_embs)

            else:
                cxt_loss = torch.tensor(0)

            final_loss = ent_loss+cxt_loss
            final_loss.backward()
            if i % config.pretrain_batch_size == 0:
                nn.utils.clip_grad_norm_(parameters_to_optimize, 5)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            ent_loss_item = round(ent_loss.data.item(), 4)
            cxt_loss_item = round(cxt_loss.data.item(), 4)
            tq.set_postfix(ent=ent_loss_item,
                           cxt=cxt_loss_item)

        b3_prec, b3_rec, b3_f1, hom, com, v_f1, ari = evaluate(
            model, trexDataset, epoch)
        table.add_row([b3_prec, b3_rec, b3_f1, hom, com, v_f1, ari])
        print(table)

        if b3_f1 > best_acc:
            best_acc = b3_f1
            torch.save({'state_dict': model.module.state_dict()},
                    ckpt_file_path)

if __name__ == "__main__":

    causal_pretrain(model)
