import os
import re
import json
from tqdm import tqdm

dir_path = "/home2/liufangchao/projects/plms-graph2text/my_webnlg/outputs"

out_path = "/home/liufangchao/projects/CausalRE/data/pretrain.json"

entity_file = "/home/liufangchao/projects/CausalRE/data/t-rex/entity_label.json"
# TODO: whether to search for the entty QID?


def get_pos(sentence_tokens, entity_tokens):
    sen_len = len(sentence_tokens)
    ent_len = len(entity_tokens)
    ret = None
    for i in range(sen_len-ent_len+1):
        if sentence_tokens[i:i+ent_len] == entity_tokens:
            if ret is not None:
                return None
            ret = [pos for pos in range(i, i+ent_len)]

    return ret


def extract_trip(triplet_sen, pred_sen):
    ins_list = []
    # return [{"tokens":[],"h":[name, qid, pos_list],"t":[...]},{}]
    pred_tokens = pred_sen.strip().split()
    triplet_sen = triplet_sen.strip()
    trip_list = re.findall(
        "> (.*?)<", triplet_sen)
    trip_list.append(triplet_sen.split("<T>")[-1])
    trip_list = [phrase.strip().lower() for phrase in trip_list]
    trip_token_list = [phrase.split() for phrase in trip_list]
    for i in range(len(trip_token_list)//3):
        head_token = trip_token_list[i*3]
        tail_token = trip_token_list[i*3+2]
        head_pos = get_pos(pred_tokens, head_token)
        tail_pos = get_pos(pred_tokens, tail_token)
        if head_pos == tail_pos:
            return None

        if head_pos is None or tail_pos is None:
            return None
        ins_dict = {"tokens": pred_tokens, "h": [
            trip_list[i*3], "-1", head_pos], 't': [trip_list[i*3+2], '-1', tail_pos]}
        ins_list.append(ins_dict)
    return ins_list


ins_list = []

with open(out_path, 'w') as f:
    for i in tqdm(range(10)):
        seen_ori_sentences = []

        file_name = "test_model_0"+str(i)
        num = i
        pred_dir_path = os.path.join(dir_path, file_name, "val_outputs")
        source_dir_path = "/home2/liufangchao/projects/plms-graph2text/my_webnlg/data/split_0" + \
            str(num)

        pred_names = [name for name in os.listdir(pred_dir_path)]
        prefix_name = [name.split("_")[0] for name in pred_names]
        source_file_paths = [os.path.join(
            source_dir_path, prefix+".source") for prefix in prefix_name]
        pred_file_paths = [os.path.join(pred_dir_path, pred_name)
                           for pred_name in pred_names]

        s_f = (open(source_file, 'r')
               for source_file in source_file_paths)
        p_f = (open(pred_file, 'r')
               for pred_file in pred_file_paths)

        for s1_line, s2_line, s3_line, s4_line, s5_line, p1_line, p2_line, p3_line, p4_line, p5_line in tqdm(zip(*s_f, *p_f)):
            if p1_line in seen_ori_sentences:
                continue
            else:
                seen_ori_sentences.append(p1_line)
            ins1 = extract_trip(s1_line, p1_line)
            if ins1 is None:
                continue
            ins2 = extract_trip(s2_line, p2_line)
            if ins2 is None:
                continue
            ins3 = extract_trip(s3_line, p3_line)
            if ins3 is None:
                continue
            ins4 = extract_trip(s4_line, p4_line)
            if ins4 is None:
                continue
            ins5 = extract_trip(s5_line, p5_line)
            if ins5 is None:
                continue

            ins_dict = {}
            for ins, prefix in zip([ins1, ins2, ins3, ins4, ins5], prefix_name):
                if "ori" in prefix:
                    ins_dict["ori"] = ins[0]
                    continue
                elif "alias" in prefix:
                    ins_dict['alias'] = ins[0]
                    continue
                elif "head" in prefix:
                    if len(ins) != 2:
                        continue
                    if ins[0] == ins[1]:
                        continue
                    ins_dict['head'] = ins
                    continue
                elif 'tail' in prefix:
                    if len(ins) != 2:
                        continue
                    if ins[0] == ins[1]:
                        continue
                    ins_dict['tail'] = ins
                elif 'other' in prefix:
                    ins_dict['other'] = ins[0]
                    continue
            if len(ins_dict.items()) < 5:
                continue
            s = json.dumps(ins_dict)
            ins_list.append(s)
            if len(ins_list) > 200:
                f.writelines("\n".join(ins_list))
                f.write("\n")
                ins_list = []
