import json
import os
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm
from search import search_for_entity_label
from elasticsearch_client import ElasticsearchClient
import random


def parse_prop():
    soup = BeautifulSoup(open(os.path.join("data/", "wiki_prop.html")))

    prop_dict = {}
    prop_list = soup.find(
        "table", {"class": "wikitable sortable"}).findAll("tr")

    for prop in prop_list:
        tds = prop.findAll("td")
        if not tds:
            continue
        item = {"label": tds[1].text, "description": tds[2].text.strip(), "alias": tds[3].text.strip(
        ), "data_type": tds[4].text.strip(), "count": tds[5].text.strip()}
        prop_dict[tds[0].text] = {
            "label": tds[1].text, "alias": [x.strip() for x in tds[3].text.strip().split(",")]}

    with open(Path("data")/"wiki_prop.json", 'w') as f:
        json.dump(prop_dict, f)


def entity_label(entity_file):
    out_file = "/home/liufangchao/projects/CausalRE/data/t-rex/entity_label.json"
    es = ElasticsearchClient()
    with open(entity_file, 'r') as f:
        entity_dict = json.load(f)
    entity_label_dct = {}
    for qid, val in tqdm(entity_dict.items()):
        label = search_for_entity_label(es, qid)
        if label:
            entity_label_dct[qid] = label

    with open(out_file, 'w') as f:
        json.dump(entity_label_dct, f)


def generate_multi_triplet(instance_file, wiki_prop_file, entity_label_file, out_dir):
    triplet_graph = {}
    entity_label_dict = {}
    out_file = "/home/liufangchao/projects/CausalRE/data/t-rex/trip_graph.json"

    with open(entity_label_file, 'r') as f:
        entity_label_dict = json.load(f)

    with open(wiki_prop_file, 'r') as f:
        prop_dict = json.load(f)

    def reverse_rel(rel):
        if rel.startswith("R"):
            return rel[1:]
        else:
            return "R"+rel

    def get_rel_id(rel_id):
        if rel_id.startswith("R"):
            rel_id = rel_id[1:]
        return "P"+rel_id

    def get_graph(rel_id, h_name, t_name, gen_alias=False):
        if rel_id.startswith("R"):
            rel_id = rel_id[1:]
            temp = h_name
            h_name = t_name
            t_name = temp
        rel_id = "P"+rel_id
        try:
            rel = prop_dict[rel_id]
        except KeyError:
            return None
        rel_name = rel['label']
        if len(rel['alias']) == 0:
            return None
        alias_name = random.choice(rel['alias'])
        ori = "<H> "+h_name+" <R> "+rel_name+" <T> "+t_name
        if gen_alias:
            alias = "<H> "+h_name+" <R> "+alias_name+" <T> "+t_name
            return ori, alias
        else:
            return ori
 
    if os.path.exists(out_file):
        with open(out_file, 'r') as f:
            triplet_graph = json.load(f)
    else:
        with open(instance_file, 'r') as f:
            for line in tqdm(f):
                rel_ins = json.loads(line)
                h_id = rel_ins['h'][1]
                t_id = rel_ins['t'][1]
                p_id = rel_ins['rel']
                if h_id in triplet_graph:
                    if t_id not in triplet_graph[h_id]:
                        triplet_graph[h_id][t_id] = p_id
                else:
                    triplet_graph[h_id] = {t_id: p_id}
                if t_id in triplet_graph:
                    if h_id not in triplet_graph[t_id]:
                        triplet_graph[t_id][h_id] = reverse_rel(p_id)
                else:
                    triplet_graph[t_id] = {h_id: reverse_rel(p_id)}
        with open(out_file, 'w') as f:
            json.dump(triplet_graph, f)

    count = 1
    seen_h_id = []
    ori_f = open(os.path.join(out_dir, "ori.source"), 'w')
    alias_f = open(os.path.join(out_dir, "alias.source"), 'w')
    head_f = open(os.path.join(out_dir, "head_extd.source"), 'w')
    tail_f = open(os.path.join(out_dir, "tail_extd.source"), 'w')
    for h_id, trips in tqdm(triplet_graph.items()):
        ori_list = []
        alias_list = []
        head_list = []
        tail_list = []

        seen_h_id.append(h_id)

        trips_list = list(trips.items())
        trips_size = len(trips_list)
        for i in range(trips_size-1):
            if len(ori_list) > 170:
                break
            t_id, rel_id = trips_list[i]
            if t_id in seen_h_id:
                continue
            try:
                h_name = entity_label_dict[h_id]
                t_name = entity_label_dict[t_id]
            except KeyError:
                continue
            for j in range(i+1, trips_size):
                o_t_id, o_rel_id = trips_list[j]
                if rel_id != o_rel_id:
                    try:
                        t_trip = triplet_graph[t_id]
                    except KeyError:
                        continue
                    res = get_graph(
                        rel_id, h_name, t_name, gen_alias=True)
                    if res is None:
                        continue
                    ori_sen, alias_sen = res

                    t_o_t_id, t_o_rel_id = random.choice(list(t_trip.items()))

                    try:
                        o_t_name = entity_label_dict[o_t_id]
                        t_o_t_name = entity_label_dict[t_o_t_id]
                    except KeyError:
                        continue
                    head_extd_sen = get_graph(o_rel_id, h_name, o_t_name)
                    if head_extd_sen is None:
                        continue
                    tail_extd_sen = get_graph(t_o_rel_id, t_name, t_o_t_name)
                    if tail_extd_sen is None:
                        continue
                    ori_list.append(ori_sen)
                    alias_list.append(alias_sen)
                    head_list.append(" ".join([ori_sen, head_extd_sen]))
                    tail_list.append(" ".join([ori_sen, tail_extd_sen]))
        if len(ori_list) > 0:
            ori_f.writelines("\n".join(ori_list))
            ori_f.write("]n")
            alias_f.writelines("\n".join(alias_list))
            alias_f.write("\n")
            head_f.writelines("\n".join(head_list))
            head_f.write("\n")
            tail_f.writelines("\n".join(tail_list))
            tail_f.write("\n")
    ori_f.close()
    alias_f.close()
    head_f.close()
    tail_f.close()


if __name__ == "__main__":
    train_file = "/home/liufangchao/projects/CausalRE/data/t-rex/json-ds-train.json"
    wiki_prop_file = "/home/liufangchao/projects/CausalRE/data/wiki_prop.json"
    entity_file = "/home/liufangchao/projects/CausalRE/data/t-rex/entity.json"
    entity_label_file = "/home/liufangchao/projects/CausalRE/data/t-rex/entity_label.json"
    out_dir = "/home/liufangchao/projects/CausalRE/data/triplet_nlg_new"
    # entity_label(entity_file)
    generate_multi_triplet(train_file, wiki_prop_file,
                           entity_label_file, out_dir)
