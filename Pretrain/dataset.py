import os, pickle, random, bisect
import numpy as np
import torch

from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler, BatchSampler

from data_utils import load_text, clean_text, load_json, padding, parse_triple
from data_utils import whole_word_masking, negative_sampling
from data_utils import get_bert_sections, map_seg_and_lm_indices

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BuildingKEPLERDataset(Dataset):
    def __init__(self, config, logger=print):
        super(BuildingKEPLERDataset, self).__init__()
        self.config = config
        self.logger = logger
        self.raw_text = load_text(os.path.join(config["data_dir"], config["raw_text_filename"]))
        self.raw_text = [clean_text(t) for t in self.raw_text if len(t) > 10]
        logger("Load raw text from %s." % os.path.join(config["data_dir"], config["raw_text_filename"]))
        self.segs = load_text(os.path.join(config["data_dir"], config["raw_text_seg_filename"]))
        self.segs = [[ss for ss in s.split(' ') if ss] for s in self.segs]
        logger("Load segs from %s." % os.path.join(config["data_dir"], config["raw_text_seg_filename"]))
        tmp_triples = load_json(os.path.join(config["data_dir"], config["triple_filename"]))
        self.triples = [parse_triple(t) for t in tmp_triples]
        logger("Load %d knowledge triples from %s." % (len(self.triples), os.path.join(config["data_dir"], config["triple_filename"])))
        self.seg_len_to_lm_ids_dict = pickle.load(open(os.path.join(config["data_dir"], "seg_len_to_lm_ids_dict.pkl"), 'rb'))
        self.tokenizer = BertTokenizer.from_pretrained(config["lm_path"])
        self.resample()

    def resample(self):
        self.logger("Resample!")
        self.wwm()
        self.ns()
        assert len(self.mlm_data) == len(self.ns_data)

    def wwm(self):
        self.mlm_data = []
        for i, t in enumerate(self.raw_text):
            bert_token_ids = self.tokenizer.encode(t)
            bert_tokens = self.tokenizer.convert_ids_to_tokens(bert_token_ids)
            seg2lm_dict, lm2seg_dict = map_seg_and_lm_indices(get_bert_sections(bert_tokens, self.segs[i]))
            inputs, labels = whole_word_masking(bert_token_ids, lm2seg_dict, \
                seg2lm_dict, self.seg_len_to_lm_ids_dict)
            self.mlm_data.append({"inputs": list(inputs), "labels": list(labels)})
    
    def ns(self):
        self.ns_data = []
        samples = negative_sampling(self.triples, return_style="hrt")
        encoded_samples = [{
            "positive": {k: self.tokenizer.encode(v) for k, v in s["positive"].items()},
            "neg_head": {k: self.tokenizer.encode(v) for k, v in s["neg_head"].items()},
            "neg_tail": {k: self.tokenizer.encode(v) for k, v in s["neg_tail"].items()}
        } for s in samples]
        self.ns_data = random.choices(encoded_samples, k=len(self.raw_text))
    
    def __len__(self):
        return len(self.mlm_data)
    
    def __getitem__(self, index):
        return (self.mlm_data[index], self.ns_data[index])
    
class GroupBatchRandomSampler(Sampler):
    def __init__(self, data_source, batch_size, group_interval):
        super(GroupBatchRandomSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.group_interval = group_interval

        max_len = max([len(d[0]["inputs"]) for d in self.data_source])
        breakpoints = np.arange(group_interval, max_len, group_interval)
        self.groups = [[] for _ in range(len(breakpoints) + 1)]
        for idx, d in enumerate(self.data_source):
            i = bisect.bisect_left(breakpoints, len(d[0]["inputs"]))
            self.groups[i].append(idx)
        self.batch_indices = []
        for g in self.groups:
            self.batch_indices.extend(list(BatchSampler(SubsetRandomSampler(g), self.batch_size, False)))

    def __iter__(self):
        batch_indices = []
        for g in self.groups:
            batch_indices.extend(list(BatchSampler(SubsetRandomSampler(g), self.batch_size, False)))
        return (batch_indices[i] for i in torch.randperm(len(self.batch_indices)))

    def __len__(self):
        return len(self.batch_indices)
    
def collate_fn(batch_data):
    padded_batch = dict()
    input_ids, input_mask = padding([d[0]["inputs"] for d in batch_data])
    labels, _ = padding([d[0]["labels"] for d in batch_data])
    padded_batch["mlm_input"] = torch.tensor(input_ids, dtype=torch.long, device=device)
    padded_batch["mlm_mask"] = torch.tensor(input_mask, dtype=torch.bool, device=device)
    padded_batch["mlm_labels"] = torch.tensor(labels, dtype=torch.long, device=device)

    alias_dict = {"pos": "positive", "nh": "neg_head", "nt": "neg_tail"}
    for _type in ["pos", "nh", "nt"]:
        for e in ["h", "r", "t"]:
            ke, mask = padding([d[1][alias_dict[_type]][e] for d in batch_data])
            padded_batch["ke_%s_%s" % (_type, e)] = torch.tensor(ke, dtype=torch.long, device=device)
            padded_batch["ke_%s_%s_mask" % (_type, e)] = torch.tensor(mask, dtype=torch.bool, device=device)

    return padded_batch
