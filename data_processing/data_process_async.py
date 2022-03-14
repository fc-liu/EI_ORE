from search import search_for_class, bfs_class_set, bfs_homogeny_entities
from elasticsearch_client import ElasticsearchClient
import os
import random
from tqdm import tqdm
import numpy as np
import json
import re
import time 

def gather_coref(file_lines, entity_dict):
    e1_temp = ".*?<e1 q=\"(.*?)\">(.*?)</e1>.*?"
    e2_temp = ".*?<e2 q=\"(.*?)\">(.*?)</e2>.*?"
    es = ElasticsearchClient()

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

def gather_cored_main(data_dir):
    trex_ds = os.path.join(data_dir, "trex-ds.tsv")
    trex_spo = os.path.join(data_dir, "trex-spo.tsv")
    out_file = os.path.join(data_dir, "entity_multi_thread.json")
    entity_dict = {}
    file_lines = []
    with open(trex_ds, 'r') as f:
        file_lines.extend(f.readlines())

    with open(trex_spo, 'r') as f:
        file_lines.extend(f.readlines())


    gather_coref(trex_ds, entity_dict)
    gather_coref(trex_spo, entity_dict)

    for val in entity_dict.values():
        val['coref'] = list(val['coref'])
    with open(out_file, 'w') as f:
        json.dump(entity_dict, f)


if __name__ == "__main__":
    data_dir = "../data/t-rex"
    start=time.time()
    gather_cored_main(data_dir)
    end=time.time()
    print(end-start)
