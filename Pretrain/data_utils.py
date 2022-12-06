import json
from functools import reduce
import numpy as np
import random

def load_text(filename):
    fp = open(filename, "rt", encoding="utf-8")
    data = [line.strip() for line in fp]
    fp.close()
    return data

def load_json(filename):
    fp = open(filename, "rt", encoding="utf-8")
    data = [json.loads(line.strip()) for line in fp]
    fp.close()
    return data

def clean_text(text):
    text = text.replace("\u201c", "\"").replace("\u201d", "\"")
    return text

def get_bert_sections(bert_tokens, seg_tokens, debug=False):
    '''
    @return sections: list of ints, same length with sentText.split() + 2, sum to len(bert_tokens)
                      stands for a glove token is split into how many bert tokens
    ----------
    @example:
        input: 'I \'m sleeping'
        bert_tokens: ['[CLS]', 'I', '\'m', 'sleep', '##ing', '[SEP]']
        output: [1, 1, 1, 2, 1]
    '''
    # bert_tokens: [CLS], w1, w2, ..., [SEP]
    # sentText = sentText.replace('  ', ' ')
    if debug:
        print("bert_tokens:")
        print("|".join(bert_tokens))
        print("seg_tokens:")
        print("|".join(seg_tokens))
    sections = [1] # [CLS] is a word with section 1
    bert_token_pointer = 1 # start from the next token of [CLS]
    new_seg_tokens = seg_tokens + [""]
    for i, word in enumerate(seg_tokens):
        word = word.lower()
        next_word = new_seg_tokens[i+1].lower()
        cur_bert_token = bert_tokens[bert_token_pointer]
        if cur_bert_token == word or (cur_bert_token == "[UNK]" and next_word.startswith(bert_tokens[bert_token_pointer+1])):
            if debug:
                print("Single token word! %s" % word)
            sections.append(1)
            bert_token_pointer += 1
        else:
            if debug:
                print("Multi token word!")
            j = bert_token_pointer + 1
            # word_buffer = [cur_bert_token]
            while j < len(bert_tokens) - 1:
                word_buffer = ''.join(bert_tokens[bert_token_pointer:j])
                if debug:
                    print("bert_token_pointer: %d, j: %d, word_buffer: %s, word: %s" % (bert_token_pointer, j, word_buffer, word))
                if word_buffer == word:
                    break
                if next_word.startswith(bert_tokens[j]) and len(word) <= len(word_buffer):
                    break
                else:
                    j += 1
            if debug:
                print("bert_token_pointer: %d, j: %d, word_buffer: %s" % (bert_token_pointer, j, word_buffer))
            sections.append(j - bert_token_pointer)
            bert_token_pointer = j
            if debug:
                print(sections)
    sections.append(1) # [SEP] is a word with section 1
    if debug:
        print(sections)
    assert len(sections) == len(seg_tokens) + 2 # [CLS] and [SEP] make len(sections) two more than len(seg_tokens)
    assert sum(sections) == len(bert_tokens)
    return sections

def map_seg_and_lm_indices(sections):
    seg_indices = list(range(len(sections)))
    lm_indices = list(range(sum(sections)))

    seg_to_lm_dict = dict()
    lm_token_pointer = 0
    for i in seg_indices:
        seg_to_lm_dict[i] = lm_indices[lm_token_pointer:lm_token_pointer+sections[i]]
        lm_token_pointer += sections[i]
    assert list(seg_to_lm_dict.keys()) == seg_indices
    assert sorted(reduce(lambda x, y: x + y, list(seg_to_lm_dict.values()))) == lm_indices
    lm_to_seg_dict = dict()
    for k, v in seg_to_lm_dict.items():
        for j in v:
            lm_to_seg_dict[j] = k
    return seg_to_lm_dict, lm_to_seg_dict

def parse_triple(triple_dict, debug=False):
    text = triple_dict["text"]
    relation = triple_dict["pred"]
    modal_verb = triple_dict["moda"]
    adverbial = triple_dict["adv_pred"]
    e1_start_index, e1_end_index = text.index("<e1/>"), text.index("<e1\\>")
    e2_start_index, e2_end_index = text.index("<e2/>"), text.index("<e2\\>")
    e1 = text[(e1_start_index+5):e1_end_index]
    e2 = text[(e2_start_index+5):e2_end_index]
    if debug:
        print(text)
        print(e1, relation, e2)
    e1_modifier, e2_modifier = [], []
    i = 0
    while i < len(text)-5:
        if text[i:i+5] == "<m1/>":
            j = i+1
            while j < len(text)-5:
                if text[j:j+5] == "<m1\\>":
                    break
                else:
                    j += 1
            e1_modifier.append(text[i+5:j])
        elif text[i:i+5] == "<m2/>":
            j = i+1
            while j < len(text)-5:
                if text[j:j+5] == "<m2\\>":
                    break
                else:
                    j += 1
            e2_modifier.append(text[i+5:j])
        i += 1
    if debug:
        print(e1_modifier, e2_modifier)
    return {
        "e1": {
            "text": e1, # str
            "modifier": e1_modifier # list
        },
        "e2": {
            "text": e2, # str
            "modifier": e2_modifier # list
        },
        "relation": {
            "text": relation, # str
            "modal_verb": modal_verb, # str
            "adverbial": adverbial # list
        }
    }

def padding(data, padding_value=0):
    '''
    pad data to maximum length
    data: list(list), unpadded data
    padding_value: int, filled value
    '''
    sen_len = max([len(d) for d in data])
    padded_data = [d + [padding_value] * (sen_len-len(d)) for d in data]
    padded_mask = [[1] * len(d) + [0] * (sen_len-len(d)) for d in data]
    return padded_data, padded_mask

def whole_word_masking(bert_token_ids, lm2seg_dict, seg2lm_dict, seg_len_to_lm_ids_dict, \
    mlm_rate=0.15, mask_rate=0.8, replace_rate=0.1):
    '''
    @input bert_tokens: list of int, input token ids of a sentence
    @input lm2seg_dict: dict(int: int), lm token index to word index mapping dict
    @input seg2lm_dict: dict(int, list), word index to lm token indices mapping dict
    @input seg_len_to_lm_ids_dict: dict(int, list), word length to list of words with same length
    @input mlm_rate: float, total rate of MLM tokens
    @input mask_rate: float, rate of [MASK] among MLM tokens
    @input replace_rate: float, rate of replacement among MLM tokens
    ----------
    @return mlm_tokens: tokens after MLM processing
    @return labels: original labels of the MLM tokens
    '''
    assert mlm_rate <= 1.
    assert mask_rate + replace_rate <= 1.
    bert_token_ids
    mlm_mask = np.random.rand(len(bert_token_ids)) < mlm_rate
    mlm_mask *= np.array([t not in [101, 102] for t in bert_token_ids]) # 101   : [CLS], 102: [SEP]
    mask_indices = np.nonzero(mlm_mask)[0]
    whole_word_indices = [seg2lm_dict[lm2seg_dict[_index]] for _index in mask_indices]
    labels = np.array(bert_token_ids, copy=True)
    inputs = np.array(bert_token_ids, copy=True)
    for word_indices in whole_word_indices:
        mlm_mask[word_indices[0]:(word_indices[-1]+1)] = 1
        prob = np.random.rand()
        if prob < mask_rate:
            inputs[word_indices[0]:(word_indices[-1]+1)] = 103 # 103: [MASK]
        elif prob < mask_rate + replace_rate:
            replace_word = random.choice(seg_len_to_lm_ids_dict[len(word_indices)])
            inputs[word_indices[0]:(word_indices[-1]+1)] = np.array(replace_word)
        else:
            pass
    labels[~mlm_mask] = -100
    return inputs, labels

def entity_text(entity):
    if entity["modifier"]:
        return "%s的%s" % ("、".join(entity["modifier"]), entity["text"])
    else:
        return entity["text"]

def relation_text(relation):
    return "%s%s%s" % (relation["modal_verb"], ''.join(relation["adverbial"]), relation["text"])

def negative_sampling(triples, return_style="raw", debug=False):
    assert return_style in ["raw", "hrt"], "Expecting 'return_style' to be 'raw' or 'hrt', got %s" % return_style
    heads = list({str(t["e1"]) for t in triples})
    tails = list({str(t["e2"]) for t in triples})
    if debug:
        print(len(triples), len(heads), len(tails))

    hr_to_true_tails_dict = dict()
    rt_to_true_heads_dict = dict()
    for t in triples:
        head, relation, tail = str(t["e1"]), str(t["relation"]), str(t["e2"])
        if (head, relation) not in hr_to_true_tails_dict:
            hr_to_true_tails_dict[(head, relation)] = []
        hr_to_true_tails_dict[(head, relation)].append(tail)
        if (relation, tail) not in rt_to_true_heads_dict:
            rt_to_true_heads_dict[(relation, tail)] = []
        rt_to_true_heads_dict[(relation, tail)].append(head)

    # corrupt heads
    neg_head_samples, neg_tail_samples = [], []
    for t in triples:
        head, relation, tail = str(t["e1"]), str(t["relation"]), str(t["e2"])
        true_head_strs = rt_to_true_heads_dict[(relation, tail)]
        head_indices = [heads.index(h) for h in true_head_strs]
        p = np.array([1] * len(heads), dtype=np.float64)
        p[head_indices] = 0
        p /= np.sum(p)
        neg_head = eval(np.random.choice(heads, replace=False, p=p))
        neg_sample = {
            "e1": neg_head,
            "e2": t["e2"],
            "relation": t["relation"]
        }
        neg_head_samples.append(neg_sample)
        
    for t in triples:
        head, relation, tail = str(t["e1"]), str(t["relation"]), str(t["e2"])
        true_tail_strs = hr_to_true_tails_dict[(head, relation)]
        tail_indices = [tails.index(t) for t in true_tail_strs]
        p = np.array([1] * len(tails), dtype=np.float64)
        p[tail_indices] = 0
        p /= np.sum(p)
        neg_tail = eval(np.random.choice(tails, replace=False, p=p))
        neg_sample = {
            "e1": t["e1"],
            "e2": neg_tail,
            "relation": t["relation"]
        }
        neg_tail_samples.append(neg_sample)
    
    if return_style == "raw":
        return [{
            "positive": t, 
            "neg_head": nh,
            "neg_tail": nt,
        } for t, nh, nt in zip(triples, neg_head_samples, neg_tail_samples)]
    else:
        return [{
            "positive": {"h": entity_text(t["e1"]), "r": relation_text(t["relation"]), "t": entity_text(t["e2"])},
            "neg_head": {"h": entity_text(nh["e1"]), "r": relation_text(nh["relation"]), "t": entity_text(nh["e2"])},
            "neg_tail": {"h": entity_text(nt["e1"]), "r": relation_text(nt["relation"]), "t": entity_text(nt["e2"])},
        } for t, nh, nt in zip(triples, neg_head_samples, neg_tail_samples)]