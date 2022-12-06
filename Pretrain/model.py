#!/usr/bin/python3.7
#encoding=utf-8

import os, sys
import json, time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import BertModel
from transformers import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup

import dataset, utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CrossEntropyLossWithMask(nn.Module):
    '''
    @param weight: (class_num) or None (default None)
    -----------
    @input logits: (batch_size, class_num, d1, d2, ..., dk)
    @input target: (batch_size, d1, d2, ..., dk)
    @input mask: broadcastable with target
    -----------
    @return loss: (1)
    '''
    def __init__(self, weight=None, ignore_index=-100):
        super(CrossEntropyLossWithMask, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
    
    def forward(self, logits, target, mask):
        if self.weight is not None:
            mask = mask * self.weight[target]
        score = nn.CrossEntropyLoss(reduction="none", ignore_index=self.ignore_index)(logits, target) * mask
        return score.sum() / mask.sum()

class BuildingKEPLER(nn.Module):
    '''
    @param path: path of BERT weights
    ----------
    @input mlm_input/mask/label: (batch_size, seq_len)
    @input ke_pos/nh/nt_h/r/t: (batch_size, ke_len)
    @input ke_pos/nh/nt_h/r/t_mask: (batch_size, ke_len)
    ----------
    @return mlm_loss: (1), ke_loss: (1)
    '''
    def __init__(self, config, vocab_size):
        super(BuildingKEPLER, self).__init__()
        lm_config = json.load(open(os.path.join(config["lm_path"], "config.json")))
        self.gamma = config["gamma"]
        self.vocab_size = vocab_size
        self.hidden_dim = lm_config["hidden_size"]
        self.model = BertModel.from_pretrained(config["lm_path"])
        self.classifier = nn.Linear(self.hidden_dim, vocab_size)
        # init params
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def mlm_loss(self, logits, labels, mask):
        # logits: (batch_size, seq_len, vocab_size)
        # labels: (batch_size, seq_len)
        # mask:   (batch_size, seq_len)
        new_logits = logits.permute(0, 2, 1) # (batch_size, vocab_size, seq_len)
        loss = CrossEntropyLossWithMask(ignore_index=-100)(new_logits, labels, mask)
        return loss
    
    def ke_loss(self, h_pos, r_pos, t_pos, h_nh, r_nh, t_nh, h_nt, r_nt, t_nt):
        # h/r/t_embed: (batch_size, hidden_dim)
        def transe(h, r, t):
            return self.gamma - torch.norm(h+r-t, p=2, dim=-1)
        pos_score = transe(h_pos, r_pos, t_pos)
        neg_head_score = transe(h_nh, r_nh, t_nh)
        neg_tail_score = transe(h_nt, r_nt, t_nt)
        loss = -F.logsigmoid(pos_score) - 1/2 * (F.logsigmoid(-neg_head_score) + F.logsigmoid(-neg_tail_score))
        loss = loss.mean()
        return loss
    
    def forward(self, batch):
        mlm_embeds = self.model(batch["mlm_input"], batch["mlm_mask"])[0]
        mlm_logits = self.classifier(mlm_embeds)
        mlm_loss = self.mlm_loss(mlm_logits, batch["mlm_labels"], batch["mlm_mask"])

        h_pos = self.model(batch["ke_pos_h"], batch["ke_pos_h_mask"])[1]
        r_pos = self.model(batch["ke_pos_r"], batch["ke_pos_r_mask"])[1]
        t_pos = self.model(batch["ke_pos_t"], batch["ke_pos_t_mask"])[1]
        h_nh = self.model(batch["ke_nh_h"], batch["ke_nh_h_mask"])[1]
        r_nh = self.model(batch["ke_nh_r"], batch["ke_nh_r_mask"])[1]
        t_nh = self.model(batch["ke_nh_t"], batch["ke_nh_t_mask"])[1]
        h_nt = self.model(batch["ke_nt_h"], batch["ke_nt_h_mask"])[1]
        r_nt = self.model(batch["ke_nt_r"], batch["ke_nt_r_mask"])[1]
        t_nt = self.model(batch["ke_nt_t"], batch["ke_nt_t_mask"])[1]
        ke_loss = self.ke_loss(h_pos, r_pos, t_pos, h_nh, r_nh, t_nh, h_nt, r_nt, t_nt)

        return mlm_loss, ke_loss

def prepare():
    parser = argparse.ArgumentParser(description="Train BERT Model for RE.")
    parser.add_argument("--config", nargs='?', default="config.json", help="Config file name.")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as fp:
        config = json.load(fp)
        fp.close()
    output_path = config["output_path"]
    prefix = config.get("prefix", "BuildingKEPLER") + "_"
    OUTPUT_DIR = os.path.join(output_path, prefix + time.strftime('%Y%m%d%H%M', time.localtime()))
    if os.path.exists(OUTPUT_DIR):
        os.system("rm -rf %s" % OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR)
    os.system("cp %s %s" % (args.config, OUTPUT_DIR))
    log_fp = open(os.path.join(OUTPUT_DIR, "log.txt"), 'w')
    logger = utils.Logger(log_fp)
    try:
        length = max([len(arg) for arg in config.keys()])
        for arg, value in config.items():
            logger("%s | %s" % (arg.ljust(length).replace("_", " "), str(value)))
        _dataset = dataset.BuildingKEPLERDataset(config, logger=logger)
        _sampler = dataset.GroupBatchRandomSampler(_dataset, config["batch_size"], 10)
        loader = DataLoader(dataset=_dataset, batch_sampler=_sampler, collate_fn=dataset.collate_fn)
        logger("Init data loader.")
        vocab_size = _dataset.tokenizer.vocab_size
        model = BuildingKEPLER(config, vocab_size).to(device)
        optimizer = AdamW(model.parameters(), lr=config["lr"])
        total_steps = config["epochs"] * len(loader)
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, \
            num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)
        logger("Init model.")
        return {
            "OUTPUT_DIR": OUTPUT_DIR,
            "config": config,
            "data_loader": loader,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "model": model,
            "logger": logger
        }
    except KeyboardInterrupt:
        logger("Interrupted.")
        logger.fp.close()
        os.system("rm -rf %s" % OUTPUT_DIR)
    except Exception as e:
        import traceback
        logger("Got exception.")
        logger.fp.close()
        print(traceback.format_exc())
        os.system("rm -rf %s" % OUTPUT_DIR)