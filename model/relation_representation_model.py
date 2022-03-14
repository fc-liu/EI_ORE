import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from config import config
from utils import position2mask

gpu_aval = torch.cuda.is_available()



def extract_entity(x, e):
    bsz=x.shape[0]
    emb=x[torch.arange(bsz),e[:,0]]
    # e2=x[torch.arange(bsz),e[:,1]]
    # e_emb=torch.cat((e1,e2),dim=1)
    return emb

class EntityMarkerEncoder(nn.Module):
    def __init__(self):
        super(EntityMarkerEncoder, self).__init__()
        self.active = torch.tanh
        self.layerNorm = nn.LayerNorm(config.hidden_size)
        self.output_sizes = (config.hidden_size, config.hidden_size)

    def forward(self, token_embs, pos1, pos2, mask):
        hidden1 = extract_entity(token_embs, pos1)
        hidden2 = extract_entity(token_embs, pos2)
        return (hidden1, hidden2)


class EntityMarkerClsEncoder(nn.Module):
    def __init__(self):
        super(EntityMarkerClsEncoder, self).__init__()
        self.output_sizes = [FLAGS.hidden_size]

    def forward(self, token_embs, pos1, pos2, mask):

        return token_embs[:, 0]


class RRModel(nn.Module):
    """
    Relation Represemtation Model
    """

    def __init__(self, embedder, relation_encoder):
        super(RRModel, self).__init__()
        self.sentence_encoder = embedder
        self.rel_encoder = relation_encoder
        # self.output_size = self.rel_encoder.output_size
        relation_hidden_size = 0
        for i in self.rel_encoder.output_sizes:
            relation_hidden_size += i
        self.relation_hidden_size = relation_hidden_size
        self.output_size = self.relation_hidden_size

    def forward(self, tokens, pos1, pos2, mask):
        # tokens: batch_size*seq_len
        # pos1,pos2: batch_size*[start, end]
        # mask: batch_size*seq_len
        # bert
        # encoded_layers, _ = self.sentence_encoder(
        #     tokens, output_all_encoded_layers=False)
        # with torch.no_grad():
        encoded_layers = self.sentence_encoder(
            tokens, attention_mask=mask)
        encoded_layers = encoded_layers[0]  # [layer]
        # mask[:,0]=0
        # mask[:,-1]=0
        relation_embs = self.rel_encoder(
            encoded_layers, pos1, pos2, mask)
        shape = len(relation_embs)
        if shape > 1:
            parts = []
            for part in relation_embs:
                parts.append(part)
            rel_rep = torch.cat(parts, dim=-1)
        else:
            rel_rep = relation_embs
        # rel_rep=relation_embs[0]
        return rel_rep


class RCModel(nn.Module):
    def __init__(self, embedder, cls_num):
        super(RCModel, self).__init__()
        self.cls_num = cls_num
        relation_encder = EntityMarkerEncoder()
        self.rr_model = RRModel(embedder, relation_encder)
        self.fc = nn.Linear(self.rr_model.output_size,
                            self.cls_num)
        self.layerNorm = nn.LayerNorm(self.rr_model.output_size)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, tokens, pos1, pos2, mask):
        batch_features = self.rr_model(tokens, pos1, pos2, mask)
        batch_features = self.drop(batch_features)
        batch_features=self.layerNorm(batch_features)
        return None, batch_features
