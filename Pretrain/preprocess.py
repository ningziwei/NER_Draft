import pickle

from transformers import BertTokenizer
from ltp import LTP

from data_utils import get_bert_sections
from data_utils import load_text, clean_text
from data_utils import map_seg_and_lm_indices

if __name__ == "__main__":
    raw_text = load_text("../data/Pretrain/raw_text.txt")
    raw_text = [clean_text(t) for t in raw_text if len(t) > 10]
    
    def generate_seg():
        ltp = LTP(path="../model/LTP")
        processed_count = 0
        segs = []
        while processed_count < len(raw_text):
            tmp_segs, _ = ltp.seg(raw_text[processed_count:(processed_count+32)])
            segs.extend(tmp_segs)
            processed_count += 32
        assert len(segs) == len(raw_text)
        fp = open("../data/Pretrain/raw_text_seg.txt", 'wt', encoding="utf-8")
        [fp.write("%s\n" % ' '.join(s)) for s in segs]
        fp.close()
        del ltp
    
    generate_seg()

    segs = load_text("../data/Pretrain/raw_text_seg.txt")
    segs = [[ss for ss in s.split(' ') if ss] for s in segs]
    tokenizer = BertTokenizer.from_pretrained("../model/chinese_roberta_wwm_ext")

    def generate_seg_len_to_lm_ids_dict():
        seg_len_to_lm_ids_dict = dict()
        for i, t in enumerate(raw_text):
            bert_token_ids = tokenizer.encode(t)
            bert_tokens = tokenizer.convert_ids_to_tokens(bert_token_ids)
            seg2lm_dict, _ = map_seg_and_lm_indices(get_bert_sections(bert_tokens, segs[i]))
            for _, lm_ids in seg2lm_dict.items():
                if len(lm_ids) not in seg_len_to_lm_ids_dict:
                    seg_len_to_lm_ids_dict[len(lm_ids)] = set()
                if len(lm_ids) == 1 and bert_token_ids[lm_ids[0]] in [101, 102]:
                    continue
                seg_len_to_lm_ids_dict[len(lm_ids)].add(tuple([bert_token_ids[_id] for _id in lm_ids]))
        for k in seg_len_to_lm_ids_dict.keys():
            seg_len_to_lm_ids_dict[k] = list(seg_len_to_lm_ids_dict[k])
        pickle.dump(seg_len_to_lm_ids_dict, open("../data/Pretrain/seg_len_to_lm_ids_dict.pkl", 'wb'))
    
    generate_seg_len_to_lm_ids_dict()