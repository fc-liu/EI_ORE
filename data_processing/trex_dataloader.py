import torch.utils.data as data
from tqdm import tqdm
import json
import re


class TRExDataSet(data.Dataset):
    def __init__(self, file_path, rel_id_json_file):
        super(TRExDataSet, self).__init__()
        self.rel_2_id, self.id_2_rel = self.load_rel_id(rel_id_json_file)
        self.json_data = self.load_data(file_path)

        self.tot_data = []
        for rel, inses in self.json_data.items():
            self.tot_data.extend(inses)

    def load_rel_id(self, rel_id_json_file):
        rel_2_id = {}
        id_2_rel = {}
        with open(rel_id_json_file, 'r') as f:
            rel_id_dict = json.load(f)
        for rel, id in rel_id_dict.items():
            id_2_rel[id] = rel

        return rel_2_id, id_2_rel

    def load_data(self, file_path):
        json_data = {}
        with open(file_path, 'r') as file:
            for line in tqdm(file.readlines(), desc="loading data"):
                tokens = []
                head_tokens = []
                head_qid = -1
                head_pos = []
                tail_tokens = []
                tail_qid = -1
                tail_pos = []
                head_start = False
                tail_start = False
                line = line.strip()
                sentence, doc_id, rel = line.split("\t")
                raw_tokens = sentence.split()
                for pos, token in enumerate(raw_tokens):
                    if "<e1" in token:
                        head_start = True
                        continue
                    elif "<e2" in token:
                        tail_start = True
                        continue
                    else:
                        if head_start:
                            if "q=" in token:
                                qid, token = token.split(">")
                                head_qid = "Q"+qid.split("\"")[1]
                                head_pos.append(len(tokens))
                            if "</e1" in token:
                                head_start = False
                                token = token.split("</e1")[0]
                            head_tokens.append(token)

                        if tail_start:
                            if "q=" in token:
                                qid, token = token.split(">")
                                tail_qid = "Q"+qid.split("\"")[1]
                                tail_pos.append(len(tokens))
                            if "</e2" in token:
                                tail_start = False
                                token = token.split("</e2")[0]
                            tail_tokens.append(token)

                    tokens.append(token)
                head = [" ".join(head_tokens), head_qid, head_pos]
                tail = [" ".join(tail_tokens), tail_qid, tail_pos]

                if rel in json_data:
                    json_data[rel].append(
                        {"tokens": tokens, "h": head, "t": tail, "rel": rel})
                else:
                    json_data[rel] = [
                        {"tokens": tokens, "h": head, "t": tail, "rel": rel}]
        return json_data

    def __getitem__(self, index):
        return self.tot_data[index]

    def __len__(self):
        return len(self.tot_data)
