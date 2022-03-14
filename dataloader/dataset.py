import torch.utils.data as data
# from config import config
import json
import random
import copy
from transformers import BertTokenizer
from config import config
import torch
from tqdm import tqdm
import os
from utils import nyt_rel_id, remain_relation


def insert_and_tokenize(tokenizer, tokens, pos1, pos2, marker1, marker2):
    tokens.insert(pos2[-1]+1, marker2[-1])
    tokens.insert(pos2[0], marker2[0])
    tokens.insert(pos1[-1]+1, marker1[-1])
    tokens.insert(pos1[0], marker1[0])
    tokens = tokens.copy()

    tokens = tokenizer.tokenize(" ".join(tokens))

    return tokens


class T_RExDataset(data.Dataset):
    def __init__(self, tokenizer, file_path, max_length=256, visual=False):
        super(T_RExDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.visual = visual
        self.data_list = self.load_data(file_path)
        samp_prob=0.2
        if config.data_src == "ds":
            samp_prob*=0.1
        self.data_list = random.sample(
            self.data_list, int(len(self.data_list)*samp_prob))

    def load_data(self, file_path):
        data_list = []
        sample_count = {}
        with open(file_path, 'r') as f:
            for line in tqdm(f.readlines(), postfix="load t-rex dataset"):
                ins = json.loads(line)
                rel = ins['rel']
                if self.visual:
                    if rel not in remain_relation:
                        continue
                    # if rel in sample_count:
                    #     if sample_count[rel] > 1000:
                    #         continue
                    #     else:
                    #         sample_count[rel] += 1
                    # else:
                    #     sample_count[rel] = 1
                ins['rel'] = remain_relation[rel]
                pos1 = ins['h'][2]
                pos2 = ins['t'][2]
                words = ins['tokens']

                if pos1[0] > pos2[0]:
                    tokens = insert_and_tokenize(self.tokenizer, words, pos2, pos1, [
                        config.e21, config.e22], [config.e11, config.e12])
                else:
                    tokens = insert_and_tokenize(self.tokenizer, words, pos1, pos2, [
                        config.e11, config.e12], [config.e21, config.e22])

                tokens.insert(0, config.cls)

                pos1 = [tokens.index(config.e11), tokens.index(config.e12)]
                pos2 = [tokens.index(config.e21), tokens.index(config.e22)]

                if len(tokens) >= self.max_length:
                    max_right = max(pos2[-1], pos1[-1])
                    min_left = min(pos1[0], pos2[0])
                    gap_length = max_right-min_left
                    if gap_length+3 > self.max_length:
                        continue
                        # tokens = [config.cls, config.e11, config.e12,
                        #           config.e21, config.e22, config.sep]
                    elif max_right+1 < self.max_length:
                        tokens = tokens[:self.max_length-1]
                    elif len(tokens)-min_left+1 < self.max_length:
                        tokens = tokens[min_left:]
                        tokens.insert(0, config.cls)
                    else:
                        tokens = tokens[min_left:max_right+1]
                        tokens.insert(0, config.cls)
                    pos1 = [tokens.index(config.e11), tokens.index(config.e12)]
                    pos2 = [tokens.index(config.e21), tokens.index(config.e22)]
                tokens.append(config.sep)

                ins['h'][-1] = pos1
                ins['t'][-1] = pos2
                ins["raw_tokens"] = ins['tokens']
                ins['tokens'] = tokens

                if len(tokens) > self.max_length:
                    raise Exception("sequence too long")
                data_list.append(ins)
        return data_list

    def __getitem__(self, index):
        ins = self.data_list[index]
        return ins['tokens'], ins['h'][-1], ins['t'][-1], ins['rel']

    def __len__(self):
        return len(self.data_list)

    def collate_fn(self, data):
        tokens, pos1, pos2, rel_id = zip(*data)
        tokens_dict = self.tokenizer(
            tokens, add_special_tokens=False, is_split_into_words=True, return_tensors='pt', truncation=True, max_length=config.max_sentence_length, padding=True)
        tokens_ids = tokens_dict['input_ids']
        mask = tokens_dict['attention_mask']
        pos1 = torch.LongTensor(pos1)
        pos2 = torch.LongTensor(pos2)

        return tokens_ids, pos1, pos2, mask, rel_id, tokens


class NYT_FBDataset(data.Dataset):
    def __init__(self, tokenizer, file_path, visual=True):
        self.tokenizer = tokenizer
        self.data_list = self.load_data(file_path)
        self.visual = visual

    def get_rel_ins(self, sentence_str, head_str, tail_str):
        sen_tokens = self.tokenizer.tokenize(sentence_str)
        sen_len = len(sen_tokens)
        ins = {"tokens": sen_tokens}
        for idx, entity_str in enumerate([head_str, tail_str]):
            ent_tokens = self.tokenizer.tokenize(entity_str)
            ent_len = len(ent_tokens)
            for i in range(sen_len-ent_len+1):
                if sen_tokens[i:i+ent_len] == ent_tokens:
                    entity = [entity_str, '-1', [i, i+ent_len-1]]
                    if idx == 0:
                        ins['h'] = entity
                    else:
                        ins['t'] = entity
        if 'h' in ins and 't' in ins:
            return ins
        else:
            return None

    def load_data(self, file_path):
        data_list = []
        rel_names = []
        sample_count = {}

        # rel_id = {}
        dir_path, file_name = os.path.split(file_path)

        json_file_path = os.path.join(dir_path, "json", file_name)

        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:

                for line in tqdm(f, postfix="load nyt json"):
                    ins = json.loads(line)
                    rel = ins['rel']
                    if rel in nyt_rel_id:
                        # ins['rel'] = nyt_rel_id[rel]
                        if rel in sample_count:
                            if sample_count[rel] > 200:
                                continue
                            else:
                                sample_count[rel] += 1
                        else:
                            sample_count[rel] = 1
                        data_list.append(ins)

            return data_list
        with open(file_path, 'r', encoding="ISO-8859-1") as f:
            for line in tqdm(f, postfix="process nyt"):
                items = line.strip().split("\t")
                if len(items) < 9:
                    continue
                sen_str = items[6]
                rel = items[-1]
                head = items[1]
                tail = items[2]
                rel_ins = self.get_rel_ins(sen_str, head, tail)
                if rel_ins is None:
                    continue
                # if rel in rel_id:
                #     relId = rel_id[rel]
                # else:
                #     relId = len(rel_id)
                #     rel_id[rel] = len(rel_id)

                rel_ins['rel'] = rel
                data_list.append(rel_ins)

        with open(json_file_path, 'w') as f:
            f.writelines("\n".join([json.dumps(json_str)
                                    for json_str in data_list]))
        return data_list

    def __getitem__(self, index):
        ins = self.data_list[index]
        return ins['tokens'], ins['h'][-1], ins['t'][-1], nyt_rel_id[ins['rel']]

    def collate_fn(self, data):
        tokens, pos1, pos2, rel_id = zip(*data)
        tokens_dict = self.tokenizer(
            tokens, add_special_tokens=False, is_split_into_words=True, return_tensors='pt', truncation=True, max_length=config.max_sentence_length, padding=True)
        tokens_ids = tokens_dict['input_ids']
        mask = tokens_dict['attention_mask']
        pos1 = torch.LongTensor(pos1)
        pos2 = torch.LongTensor(pos2)

        return tokens_ids, pos1, pos2, mask, rel_id, tokens

    def __len__(self):
        return len(self.data_list)


class EntityIntervenDataset(data.Dataset):
    def __init__(self, tokenizer, file_path, entity_file_path, max_length=256):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_list = self.load_data(file_path)
        with open(entity_file_path, 'r') as f:
            self.entity_dict = json.load(f)
        self.data_index = [i for i in range(len(self.data_list))]

    def load_data(self, file_path):

        data_list = []
        with open(file_path, 'r') as f:
            for line in tqdm(f.readlines(), postfix="load entity dataset"):
                ins = json.loads(line)
                pos1 = ins['h'][2]
                pos2 = ins['t'][2]
                words = ins['tokens']

                if pos1[0] > pos2[0]:
                    tokens = insert_and_tokenize(self.tokenizer, words, pos2, pos1, [
                        config.e21, config.e22], [config.e11, config.e12])
                else:
                    tokens = insert_and_tokenize(self.tokenizer, words, pos1, pos2, [
                        config.e11, config.e12], [config.e21, config.e22])

                tokens.insert(0, config.cls)

                pos1 = [tokens.index(config.e11), tokens.index(config.e12)]
                pos2 = [tokens.index(config.e21), tokens.index(config.e22)]

                if len(tokens) >= self.max_length:
                    max_right = max(pos2[-1], pos1[-1])
                    min_left = min(pos1[0], pos2[0])
                    gap_length = max_right-min_left
                    if gap_length + 3 > self.max_length:
                        continue
                        # tokens = [config.cls, config.e11, config.e12,
                        #           config.e21, config.e22, config.sep]
                    elif max_right+1 < self.max_length:
                        tokens = tokens[: self.max_length-1]
                    elif len(tokens)-min_left+1 < self.max_length:
                        tokens = tokens[min_left:]
                        tokens.insert(0, config.cls)
                    else:
                        tokens = tokens[min_left: max_right+1]
                        tokens.insert(0, config.cls)
                    pos1 = [tokens.index(config.e11), tokens.index(config.e12)]
                    pos2 = [tokens.index(config.e21), tokens.index(config.e22)]
                tokens.append(config.sep)

                ins['h'][-1] = pos1
                ins['t'][-1] = pos2
                ins["raw_tokens"] = ins['tokens']
                ins['tokens'] = tokens

                if len(tokens) > self.max_length:
                    raise Exception("sequence too long")
                data_list.append(ins)
        return data_list

    def intervene(self, ins):
        isHead = random.choice([True, False])

        ret_ins_list = []
        entity = []
        if isHead:
            entity = ins['h']
            keep_ent = ins['t']
        else:
            entity = ins['t']
            keep_ent = ins['h']
        qid = entity[1]
        pos = entity[-1]
        tokens = ins["tokens"]
        entity_interven_dict = self.entity_dict[qid]
        for i, ent_typ in enumerate(["coref", "level_1", "level_2", "other"]):
            if i == 0:
                ent_list = entity_interven_dict[ent_typ].copy()
                ent_list.remove(entity[0])
                if len(ent_list) == 0:
                    ent_str = entity[0]
                else:
                    ent_str = random.choice(ent_list)
            elif i == 3:
                qid = random.choice(list(self.entity_dict.keys()))
                ent_str = random.choice(self.entity_dict[qid]['coref'])
            else:
                ent_list = list(entity_interven_dict[ent_typ].items())
                if len(ent_list) == 0:
                    ent_str = entity[0]
                else:
                    qid, ent_str = random.choice(list(
                        entity_interven_dict[ent_typ].items()))
            new_keep_ent = [item for item in keep_ent]
            ent_tokens = self.tokenizer.tokenize(ent_str)
            intv_ins = {}
            ent_left = tokens[: pos[0]+1]
            ent_right = tokens[pos[-1]:]

            ent_left.extend(ent_tokens)
            ent_left.extend(ent_right)

            if isHead:
                new_ent = [ent_str, qid, [ent_left.index(
                    config.e11), ent_left.index(config.e12)]]
                new_keep_ent = [keep_ent[0], keep_ent[1], [ent_left.index(
                    config.e21), ent_left.index(config.e22)]]
                intv_ins = {"tokens": ent_left, 'h': new_ent,
                            't': new_keep_ent}
            else:
                new_ent = [ent_str, qid, [ent_left.index(
                    config.e21), ent_left.index(config.e22)]]
                new_keep_ent = [keep_ent[0], keep_ent[1], [ent_left.index(
                    config.e11), ent_left.index(config.e12)]]
                intv_ins = {"tokens": ent_left, 'h': new_keep_ent,
                            't': new_ent}
            ret_ins_list.append(intv_ins)
        # oth_ins = random.choice(self.data_list)
        # ret_ins_list.append(oth_ins)

        return ret_ins_list

    def __getitem__(self, index):
        """
        return: ori, coref_positive, level1 instance, level2 instance, other instance
        """
        ret_tokens = []
        ret_pos1 = []
        ret_pos2 = []
        ins = self.data_list[index]
        intvn_list = self.intervene(ins)
        ins_list = [ins]
        ins_list.extend(intvn_list)
        for item in ins_list:
            ret_tokens.append(item["tokens"])
            ret_pos1.append(item['h'][-1])
            ret_pos2.append(item['t'][-1])
        tokens_dict = self.tokenizer(
            ret_tokens, add_special_tokens=False, is_split_into_words=True, return_tensors='pt', truncation=True, max_length=config.max_sentence_length, padding=True)
        tokens_ids = tokens_dict['input_ids']
        mask = tokens_dict['attention_mask']
        ret_pos1 = torch.LongTensor(ret_pos1)
        ret_pos2 = torch.LongTensor(ret_pos2)
        return tokens_ids, ret_pos1, ret_pos2, mask, ret_tokens

    def __len__(self):
        return len(self.data_list)


class NlgEntityIntervenDataset(data.Dataset):
    def __init__(self, tokenizer, data_list, entity_file_path, max_length=256):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_list = data_list
        with open(entity_file_path, 'r') as f:
            self.entity_dict = json.load(f)
        self.data_index = [i for i in range(len(self.data_list))]

    def intervene(self, ins):
        isHead = random.choice([True, False])

        ret_ins_list = []
        entity = []
        if isHead:
            entity = ins['h']
            keep_ent = ins['t']
        else:
            entity = ins['t']
            keep_ent = ins['h']
        qid = entity[1]
        pos = entity[-1]
        tokens = ins["tokens"]
        entity_interven_dict = self.entity_dict[qid]
        for i, ent_typ in enumerate(["coref", "level_1", "level_2", "other"]):
            if i == 0:
                ent_list = entity_interven_dict[ent_typ].copy()
                ent_list = [ent.lower() for ent in ent_list]
                if entity[0] in ent_list:
                    ent_list.remove(entity[0])
                if len(ent_list) == 0:
                    ent_str = entity[0]
                else:
                    ent_str = random.choice(ent_list)
            elif i == 3:
                qid = random.choice(list(self.entity_dict.keys()))
                ent_str = random.choice(self.entity_dict[qid]['coref'])
            else:
                ent_list = list(entity_interven_dict[ent_typ].items())
                if len(ent_list) == 0:
                    ent_str = entity[0]
                else:
                    qid, ent_str = random.choice(list(
                        entity_interven_dict[ent_typ].items()))
            new_keep_ent = [item for item in keep_ent]
            ent_tokens = self.tokenizer.tokenize(ent_str)
            intv_ins = {}
            ent_left = tokens[: pos[0]+1]
            ent_right = tokens[pos[-1]:]

            ent_left.extend(ent_tokens)
            ent_left.extend(ent_right)

            if (config.e11 not in ent_left) or (config.e12 not in ent_left) or (config.e21 not in ent_left) or (config.e22 not in ent_left):
                # print("marker None")
                return None

            if isHead:
                new_ent = [ent_str, qid, [ent_left.index(
                    config.e11), ent_left.index(config.e12)]]
                new_keep_ent = [keep_ent[0], keep_ent[1], [ent_left.index(
                    config.e21), ent_left.index(config.e22)]]
                intv_ins = {"tokens": ent_left, 'h': new_ent,
                            't': new_keep_ent}
            else:
                new_ent = [ent_str, qid, [ent_left.index(
                    config.e21), ent_left.index(config.e22)]]
                new_keep_ent = [keep_ent[0], keep_ent[1], [ent_left.index(
                    config.e11), ent_left.index(config.e12)]]
                intv_ins = {"tokens": ent_left, 'h': new_keep_ent,
                            't': new_ent}
            ret_ins_list.append(intv_ins)
        # oth_ins = random.choice(self.data_list)
        # ret_ins_list.append(oth_ins)

        return ret_ins_list

    def __getitem__(self, index):
        """
        return: ori, coref_positive, level1 instance, level2 instance, other instance
        """
        ret_tokens = []
        ret_pos1 = []
        ret_pos2 = []
        ins = None
        intvn_list = None
        while intvn_list is None:
            ins = self.data_list[index][0]
            intvn_list = self.intervene(ins)
            index += 1
        ins_list = [ins]
        ins_list.extend(intvn_list)
        for item in ins_list:
            ret_tokens.append(item["tokens"])
            ret_pos1.append(item['h'][-1])
            ret_pos2.append(item['t'][-1])
        tokens_dict = self.tokenizer(
            ret_tokens, add_special_tokens=False, is_split_into_words=True, return_tensors='pt', truncation=True, max_length=config.max_sentence_length, padding=True)
        tokens_ids = tokens_dict['input_ids']
        mask = tokens_dict['attention_mask']
        ret_pos1 = torch.LongTensor(ret_pos1)
        ret_pos2 = torch.LongTensor(ret_pos2)
        return tokens_ids, ret_pos1, ret_pos2, mask, ret_tokens

    def __len__(self):
        return len(self.data_list)


class ContextInterveneDataset(data.Dataset):
    def __init__(self, tokenizer, file_path, max_length=256):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_list, self.ori_data_list = self.load_data(file_path)

    def load_data(self, file_path):
        all_data_list = []
        ori_data_list = []
        # seen_ori_ins = []
        processed_file = "/home/liufangchao/projects/CausalRE/data/processed_nlg_qid.json"
        if os.path.exists(processed_file):
            with open(processed_file, 'r') as f:
                data_list, ori_data_list = json.load(f).values()
                return data_list, ori_data_list

        with open(file_path, 'r') as f:
            for line in tqdm(f.readlines(), postfix="load context dataset"):
                inses = json.loads(line)
                # ori_words = inses['ori']['tokens']
                # if ori_words in seen_ori_ins:
                #     print("hunt")
                #     continue
                # else:
                #     seen_ori_ins.append(ori_words)
                ins_list = [inses['ori'], inses['alias'],
                            *inses['head'], *inses['tail'], inses['other']]
                processed_list = []
                for ins in ins_list:
                    pos1 = ins['h'][2]
                    pos2 = ins['t'][2]
                    words = ins['tokens']

                    if pos1[0] > pos2[0]:
                        tokens = insert_and_tokenize(self.tokenizer, words, pos2, pos1, [
                            config.e21, config.e22], [config.e11, config.e12])
                    else:
                        tokens = insert_and_tokenize(self.tokenizer, words, pos1, pos2, [
                            config.e11, config.e12], [config.e21, config.e22])

                    tokens.insert(0, config.cls)

                    pos1 = [tokens.index(config.e11), tokens.index(config.e12)]
                    pos2 = [tokens.index(config.e21), tokens.index(config.e22)]

                    if len(tokens) >= self.max_length:
                        max_right = max(pos2[-1], pos1[-1])
                        min_left = min(pos1[0], pos2[0])
                        gap_length = max_right-min_left
                        if gap_length+3 > self.max_length:
                            continue
                            # tokens = [config.cls, config.e11, config.e12,
                            #           config.e21, config.e22, config.sep]
                        elif max_right+1 < self.max_length:
                            tokens = tokens[: self.max_length-1]
                        elif len(tokens)-min_left+1 < self.max_length:
                            tokens = tokens[min_left:]
                            tokens.insert(0, config.cls)
                        else:
                            tokens = tokens[min_left: max_right+1]
                            tokens.insert(0, config.cls)
                        pos1 = [tokens.index(config.e11),
                                tokens.index(config.e12)]
                        pos2 = [tokens.index(config.e21),
                                tokens.index(config.e22)]
                    tokens.append(config.sep)

                    ins['h'][-1] = pos1
                    ins['t'][-1] = pos2
                    # ins["raw_tokens"] = ins['tokens']
                    ins['tokens'] = tokens

                    if len(tokens) > self.max_length:
                        raise Exception("sequence too long")
                    processed_list.append(ins)
                all_data_list.append(processed_list)
                ori_data_list.append(processed_list[0])
        with open(processed_file, 'w') as f:
            json.dump({"all_data": all_data_list,
                       "ori_data": ori_data_list}, f)

        return all_data_list, ori_data_list

    def __getitem__(self, index):
        """
        return ori, alias relation instance, head populate same instance, head populate other instance, tail populate same instance, tail populate other instance
        """
        ret_tokens = []
        ret_pos1 = []
        ret_pos2 = []
        # oth_ins = random.choice(self.ori_data_list)
        ins_list = self.data_list[index]
        # ins_list.append(oth_ins)
        for item in ins_list:
            ret_tokens.append(item["tokens"])
            ret_pos1.append(item['h'][-1])
            ret_pos2.append(item['t'][-1])
        tokens_dict = self.tokenizer(
            ret_tokens, add_special_tokens=False, is_split_into_words=True, return_tensors='pt', truncation=True, max_length=config.max_sentence_length, padding=True)
        tokens_ids = tokens_dict['input_ids']
        mask = tokens_dict['attention_mask']
        ret_pos1 = torch.LongTensor(ret_pos1)
        ret_pos2 = torch.LongTensor(ret_pos2)
        return tokens_ids, ret_pos1, ret_pos2, mask, ret_tokens

    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    file_path = "/home/liufangchao/projects/CausalRE/data/t-rex/json-spo-val.json"
    nlg_file_path = "/home/liufangchao/projects/CausalRE/data/pretrain.json"
    entity_file_path = "/home/liufangchao/projects/CausalRE/data/t-rex/entity.json"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [config.e11, config.e12, config.e21, config.e22]})
    # import time
    # start = time.time()
    # entity_dataset = EntityIntervenDataset(
    #     tokenizer, file_path, entity_file_path)
    cxt_dataset = ContextInterveneDataset(
        tokenizer, nlg_file_path)
    # entity_dataset = NlgEntityIntervenDataset(
    #     tokenizer, cxt_dataset.data_list, entity_file_path)
    # for item in entity_dataset:
    #     print(0)
        # break

    # dataset = T_RExDataset(tokenizer, file_path)
    # end = time.time()
    # print(end-start)
    # start = time.time()
    # for item in dataset:
    #     print()
    #     break
    # end = time.time()
    # print(end-start)

    nyt_file = "/home/liufangchao/projects/CausalRE/data/NYT-FB/candidate-2000s.context.filtered.triples.pathfiltered.pos.single-relation.sortedondate.txt"
    # rel=set()
    # with open(nyt_file, 'r', encoding = "ISO-8859-1") as f:
    #     lines=f.readlines()
    #     for line in tqdm(lines):
    #         line=line.strip()
    #         items=line.split('\t')
    #         if len(items) >= 9:
    #             rel.add(items[-1])

    # print("\n".join(rel))
    # print(len(rel))
    # dataset = NYT_FBDataset(tokenizer, nyt_file)
