import json
from tqdm import tqdm

rel_file_path = "/home/liufangchao/projects/CausalRE/data/processed_nlg_qid.json"
entity_file_path = "/home/liufangchao/projects/CausalRE/data/t-rex/entity_label.json"
rel_output_path = "/home/liufangchao/projects/CausalRE/data/processed_nlg_qid.json"

with open(entity_file_path, 'r') as f:
    entity_label_dict = json.load(f)

with open(rel_file_path, 'r') as f:
    data_list, ori_data_list = json.load(f).values()

entity_qid_dict = {}

for key, val in tqdm(entity_label_dict.items(), postfix="building entity qid dict"):
    entity_qid_dict[val.lower()] = key
    # for coref


def get_qid(entity_name):
    entity_name = entity_name.lower()
    if entity_name in entity_qid_dict:
        return entity_qid_dict[entity_name]

    return None


for rel_inses in tqdm(data_list):
    ori_rel_ins = rel_inses[0]
    head = ori_rel_ins['h']
    head_name = head[0]
    head_qid = get_qid(head_name)
    if head_qid is None:
        print("head none")
        continue
    head[1] = head_qid

    tail = ori_rel_ins['t']
    tail_name = tail[0]
    tail_qid = get_qid(tail_name)
    if tail_qid is None:
        print("tail none")
        continue
    tail[1] = tail_qid

with open(rel_output_path,'w') as f:
    json.dump({"all_data": data_list,
                "ori_data": ori_data_list}, f)