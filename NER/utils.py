import os
import csv
import re
import json
import pandas as pd
import numpy as np

def read_txt(path):
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        try:
            with open(path, 'r', encoding='ansi') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(path, 'r', encoding='utf8') as f:
                lines = f.readlines()
    return lines

def write_txt(lines, path):
    try:
        with open(path, 'w', encoding='utf8') as f:
            for line in lines:
                f.write(line+'\n')
    except UnicodeDecodeError:
        try:
            with open(path, 'w', encoding='ansi') as f:
                for line in lines:
                    f.write(line+'\n')
        except UnicodeDecodeError:
            with open(path, 'w') as f:
                for line in lines:
                    f.write(line+'\n')

def pd_read_csv(path):
    try:
        lines = pd.read_csv(path)
    except UnicodeDecodeError:
        try:
            lines = pd.read_csv(path, encoding='ansi')
        except UnicodeDecodeError:
            lines = pd.read_csv(path, encoding='utf8')
    return lines

def pd_write_csv(df, path):
    df.to_csv(path, encoding='utf8', index=None)

def dic_read_csv(path):
    try:
        with open(path, 'r') as f:
            lines = list(csv.reader(f))
    except UnicodeDecodeError:
        try:
            with open(path, 'r', encoding='ansi') as f:
                lines = list(csv.reader(f))
        except UnicodeDecodeError:
            with open(path, 'r', encoding='utf8') as f:
                lines = list(csv.reader(f))
    for i in range(len(lines)):
        if len(lines[i])==2:
            lines[i].append('')
    lines = np.array(lines)
    lines = {lines[0][i]: lines[1:,i] for i in range(len(lines[0]))}
    return lines

def read_csv(path):
    try:
        with open(path, 'r') as f:
            lines = list(csv.reader(f))
    except UnicodeDecodeError:
        try:
            with open(path, 'r', encoding='ansi') as f:
                lines = list(csv.reader(f))
        except UnicodeDecodeError:
            with open(path, 'r', encoding='utf8') as f:
                lines = list(csv.reader(f))
    # lines_ = []
    # for i in range(len(lines)):
    #     if lines[i][0] == '': continue
    #     lines_.append(lines[i])
    return lines

def write_csv(lines, path):
    try:
        with open(path, 'w', encoding='utf8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(lines)
    except UnicodeEncodeError:
        try:
            with open(path, 'w', encoding='ansi', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(lines)
        except UnicodeEncodeError:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(lines)
    return lines

def write_RE_json(lines, path):
    with open(path, 'w', encoding='utf8') as f:
        for l in lines:
            f.write(json.dumps(l)+'\n')

def read_RE_json(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [json.loads(l) for l in lines]

def read_json(path):
    with open(path, 'r', encoding='utf8') as f:
        res = json.load(f)
    return res

def write_json(path, content):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(content, f)

def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if '\u4e00' <= uchar<='\u9fa5':
        return True
    else:
        return False

def find_all_tag(sent, word, tag=None):
    '''
    找到sentence中所有word的位置，且在tag中有正确的标注
    '''
    res = []
    len_w = len(word)
    idx = sent.find(word)
    pattern = 'BI*E'
    if tag:
        while idx != -1:
            if len_w==1 and tag[idx] == 'S':
                res.append([idx, idx+len_w])
            elif re.match(pattern, ''.join(tag[idx:idx+len_w])):
                res.append([idx, idx+len_w])
            idx = sent.find(word, idx + 1)
    else:
        while idx != -1:
            res.append([idx, idx+len_w])
            idx = sent.find(word, idx + 1)
    return res

