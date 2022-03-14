from search import search_for_class, bfs_class_set, bfs_homogeny_entities
from elasticsearch_client import ElasticsearchClient
import os
import random
from tqdm import tqdm
import numpy as np
import json
import re
from threading import RLock
import _thread
import threading

lock = RLock()


def rel_id_json(file_path, out_path):
    rel_id_dict = {}
    idx = 0
    with open(file_path, "r") as f:
        for line in tqdm(f, desc="process relation to id"):
            line = line.strip()
            _, _, rel = line.split("\t")
            if rel not in rel_id_dict:
                rel_id_dict[rel] = idx
                idx += 1
    with open(out_path, 'w') as f:
        json.dump(rel_id_dict, f)


def rel_count_nyt(file_path, out_path):
    rel_count_dict = {}
    with open(file_path, 'r', encoding="ISO-8859-1") as f:
        for line in tqdm(f):
            items = line.strip().split("\t")
            if len(items) < 9:
                continue
            rel = items[-1]
            if rel not in rel_count_dict:
                rel_count_dict[rel] = 1
            else:
                rel_count_dict[rel] += 1
    rel_cnt = sorted(rel_count_dict.items(), key=lambda d: d[1], reverse=True)

    with open(out_path, 'w') as f:
        f.writelines("\n".join([str(k)+": " + str(v) for k, v in rel_cnt]))


def stat_rel(file_path, out_path):
    rel_stat_dict = {}
    with open(file_path, "r") as f:
        for line in tqdm(f):
            line = line.strip()
            _, _, predicate = line.split("\t")
            if predicate in rel_stat_dict:
                rel_stat_dict[predicate] += 1
            else:
                rel_stat_dict[predicate] = 1

    rel_stat_dict = sorted(rel_stat_dict.items(),
                           key=lambda kv: (kv[1], kv[0]))
    with open(out_path, 'w') as f:
        for pred, cnt in rel_stat_dict:
            f.write(pred+": "+str(cnt)+"\n")


def split_trex_train_val(file_path, train_out_path, val_out_path):
    pred_dict = {}
    with open(file_path, 'r') as f:
        for line in tqdm(f):
            line = line.strip()
            _, _, predicate = line.split("\t")
            # predicate = predicate.split("R")[-1]
            if predicate in pred_dict:
                pred_dict[predicate].append(line)
            else:
                pred_dict[predicate] = [line]
    with open(train_out_path, "w") as tf:
        with open(val_out_path, 'w') as vf:
            for predicate, lines in tqdm(pred_dict.items()):
                pred_size = len(lines)

        # val_size=pred_size-round(pred_size*0.8)
        # if val_size<1:
        #     print(predicate+": "+str())

                np_lines = np.asarray(lines)

                id_lit = np.arange(pred_size)
                np.random.shuffle(id_lit)

                train_size = round(pred_size*0.8)
                train_ids = id_lit[:train_size]
                val_ids = id_lit[train_size:]

                train_list = np_lines[train_ids]
                val_list = np_lines[val_ids]

                tf.write("\n".join(train_list))
                vf.write("\n".join(val_list))
                tf.write("\n")
                if len(val_list) > 0:
                    vf.write("\n")

                tf.flush()
                vf.flush()


def trex_split_main(data_dir):
    trex_ds = os.path.join(data_dir, "trex-ds.tsv")
    ds_train = os.path.join(data_dir, "trex-ds-train.tsv")
    ds_val = os.path.join(data_dir, "trex-ds-val.tsv")

    trex_spo = os.path.join(data_dir, "trex-spo.tsv")
    spo_train = os.path.join(data_dir, "trex-spo-train.tsv")
    spo_val = os.path.join(data_dir, "trex-spo-val.tsv")

    split_trex_train_val(trex_ds, ds_train, ds_val)
    split_trex_train_val(trex_spo, spo_train, spo_val)


def rel_statis_main(data_dir):
    ds_train = os.path.join(data_dir, "trex-ds-train.tsv")
    ds_val = os.path.join(data_dir, "trex-ds-val.tsv")
    spo_train = os.path.join(data_dir, "trex-spo-train.tsv")
    spo_val = os.path.join(data_dir, "trex-spo-val.tsv")

    ds_train_out = os.path.join(data_dir, "pred-ds-train.tsv")
    ds_val_out = os.path.join(data_dir, "pred-ds-val.tsv")
    spo_train_out = os.path.join(data_dir, "pred-spo-train.tsv")
    spo_val_out = os.path.join(data_dir, "pred-spo-val.tsv")

    stat_rel(ds_train, ds_train_out)
    stat_rel(ds_val, ds_val_out)
    stat_rel(spo_train, spo_train_out)
    stat_rel(spo_val, spo_val_out)


def rel_id_main(data_dir):
    ds = os.path.join(data_dir, "trex-ds.tsv")
    ds_rel_id = os.path.join(data_dir, "rel-id-ds.json")
    spo = os.path.join(data_dir, "trex-spo.tsv")
    spo_rel_id = os.path.join(data_dir, "rel-id-spo.json")
    rel_id_json(ds, ds_rel_id)
    rel_id_json(spo, spo_rel_id)


def convert_to_rel_json(file_path, out_path):
    json_data = []
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
                            qid, token = token.split("\">")
                            head_qid = "Q"+qid.split("\"")[1]
                        if "</e1" in token:
                            head_start = False
                            token = token.split("</e1")[0]
                        head_pos.append(len(tokens))
                        head_tokens.append(token)

                    if tail_start:
                        if "q=" in token:
                            qid, token = token.split("\">")
                            tail_qid = "Q"+qid.split("\"")[1]

                        if "</e2" in token:
                            tail_start = False
                            token = token.split("</e2")[0]
                        tail_pos.append(len(tokens))
                        tail_tokens.append(token)

                tokens.append(token)
            head = [" ".join(head_tokens), head_qid, head_pos]
            tail = [" ".join(tail_tokens), tail_qid, tail_pos]

            rel_ins = {"tokens": tokens, "h": head, "t": tail, "rel": rel}
            json_data.append(json.dumps(rel_ins))
    with open(out_path, 'w') as f:
        f.write("\n".join(json_data))


def convert_json_main(data_dir):
    ds_train = os.path.join(data_dir, "trex-ds-train.tsv")
    ds_val = os.path.join(data_dir, "trex-ds-val.tsv")
    spo_train = os.path.join(data_dir, "trex-spo-train.tsv")
    spo_val = os.path.join(data_dir, "trex-spo-val.tsv")

    ds_train_out = os.path.join(data_dir, "json-ds-train.json")
    ds_val_out = os.path.join(data_dir, "json-ds-val.json")
    spo_train_out = os.path.join(data_dir, "json-spo-train.json")
    spo_val_out = os.path.join(data_dir, "json-spo-val.json")

    convert_to_rel_json(spo_val, spo_val_out)
    convert_to_rel_json(ds_train, ds_train_out)
    convert_to_rel_json(ds_val, ds_val_out)
    convert_to_rel_json(spo_train, spo_train_out)


def gather_coref(file_lines, entity_dict):
    e1_temp = ".*?<e1 q=\"(.*?)\">(.*?)</e1>.*?"
    e2_temp = ".*?<e2 q=\"(.*?)\">(.*?)</e2>.*?"
    es = ElasticsearchClient()
    global lock

    for line in tqdm(file_lines):
        line = line.strip()
        entity = re.match(e1_temp, line)
        entity_qid = "Q"+entity.group(1)
        entity_name = entity.group(2)

        # lock.acquire()
        if entity_qid not in entity_dict:
            entity_dict[entity_qid] = {}
            entity_dict[entity_qid]['coref'] = {entity_name}
            # lock.release()
            hie_entity = bfs_homogeny_entities(es, entity_qid)
            entity_dict[entity_qid]['level_1'] = hie_entity[0]
            entity_dict[entity_qid]['level_2'] = hie_entity[1]

        else:
            # lock.release()
            entity_dict[entity_qid]['coref'].add(entity_name)

        entity = re.match(e2_temp, line)
        entity_qid = "Q"+entity.group(1)
        entity_name = entity.group(2)

        # lock.acquire()
        if entity_qid not in entity_dict:
            entity_dict[entity_qid] = {}
            entity_dict[entity_qid]['coref'] = {entity_name}
            # lock.release()
            hie_entity = bfs_homogeny_entities(es, entity_qid)
            entity_dict[entity_qid]['level_1'] = hie_entity[0]
            entity_dict[entity_qid]['level_2'] = hie_entity[1]

        else:
            # lock.release()
            entity_dict[entity_qid]['coref'].add(entity_name)


class EntityThread(threading.Thread):
    def __init__(self, file_lines, entity_dict):
        super(EntityThread, self).__init__()
        self.file_lines = file_lines
        self.entity_dict = entity_dict

    def run(self):
        gather_coref(self.file_lines, self.entity_dict)


def gather_cored_main(data_dir):
    trex_ds = os.path.join(data_dir, "trex-ds.tsv")
    trex_spo = os.path.join(data_dir, "trex-spo.tsv")
    out_file = os.path.join(data_dir, "entity_multi_thread.json")
    entity_dict = {}
    thread_size = 4
    thread_list = []
    file_lines = []
    with open(trex_ds, 'r') as f:
        file_lines.extend(f.readlines())

    with open(trex_spo, 'r') as f:
        file_lines.extend(f.readlines())
    line_size = len(file_lines)
    lines_per_thread = int(line_size/thread_size)

    for i in range(thread_size-1):
        thread_list.append(EntityThread(
            file_lines[i*lines_per_thread:(i+1)*lines_per_thread], entity_dict))
    i += 1
    thread_list.append(EntityThread(
        file_lines[i*lines_per_thread:(i+1)*lines_per_thread], entity_dict))

    # gather_coref(trex_ds, entity_dict)
    # gather_coref(trex_spo, entity_dict)

    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join()

    for val in entity_dict.values():
        val['coref'] = list(val['coref'])
    with open(out_file, 'w') as f:
        json.dump(entity_dict, f)


def compute(data_dir):
    trex_ds = os.path.join(data_dir, "trex-ds.tsv")
    trex_spo = os.path.join(data_dir, "trex-spo.tsv")
    with open(trex_ds, "r") as f:
        lines = f.readlines()
        print(len(lines))


if __name__ == "__main__":
    data_dir = "data/t-rex"
    # trex_split_main(data_dir)
    # rel_statis_main(data_dir)
    # rel_id_main(data_dir)
    # convert_json_main(data_dir)
    # gather_cored_main(data_dir)
    # compute(data_dir)
    rel_count_nyt("/home/liufangchao/projects/CausalRE/data/NYT-FB/candidate-2000s.context.filtered.triples.pathfiltered.pos.single-relation.sortedondate.test.80%.txt",
                  "/home/liufangchao/projects/CausalRE/data/NYT-FB/test_rel_id.json")
    # rel_count_nyt("")
