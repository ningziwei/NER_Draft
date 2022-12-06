import numpy as np
from itertools import chain
from gensim.models import KeyedVectors
from sentence_transformers.util import semantic_search

class TxtSim:
    '''计算词级和句子级的文本相似度'''
    def __init__(self, wv_from_text, jieba, sbert=None):
        '''
        wv_from_text: 预训练词向量模型，用来算词汇的语义相似度
        sbert: sentence-bert模型，用来计算句子的语义相似度
        '''
        self.wv_from_text = wv_from_text
        self.k2idx = wv_from_text.key_to_index
        self.jieba = jieba
        self.sbert = sbert
    
    def uni_vec(self, vec):
        '''向量归一化'''
        norm = np.linalg.norm(vec,axis=-1)
        vec = vec/np.expand_dims(norm,-1)
        return vec
    
    def deal_oov(self, txt):
        '''
        用sbert给不在预训练词表中的字符生成向量
        将向量从sbert的384维降到预训练词表的200维
        '''
        vec = self.sbert.encode(txt)
        vec_ = np.zeros(200)
        vec_[:192] = (vec[::2] + vec[1::2])/2
        return vec_

    def get_txt_vec(self, txt, mode='wv'):
        '''
        计算一个句子的向量
        sbert适合对词表中没有的文本编码
        若没有，则用词向量的均值编码
        '''
        if isinstance(txt, str): txt = [txt]
     
        if mode=='sent':
            txt_vecs = self.sbert.encode(txt)
            return self.uni_vec(txt_vecs)
        elif mode=='wv':
            if all([t in self.k2idx for t in txt]):
                return self.wv_from_text[txt]
            txt_vecs = []
            for t in txt:
                if t in self.k2idx:
                    txt_vecs.append(self.wv_from_text[t])
                else:
                    # 先用jieba分词，若分词结果不在词表中，则用sbert生成
                    words = list(self.jieba.cut(t))
                    word_len = np.array([len(w) if w in self.k2idx else 1 for w in words])
                    words_vec = np.array([self.wv_from_text[w] if w in self.k2idx \
                        else self.deal_oov(w) for w in words])
                    txt_vecs.append(np.dot(word_len, words_vec))
            txt_vecs = np.array(txt_vecs)
            return self.uni_vec(txt_vecs)
        else:
            raise ValueError('检查参数mode设置，可选项为wv和sent')

    def get_synon(self, word, topn=50, thres=0.75, mode='wv'):
        '''
        返回排名前topn且相似度大于thres的词汇
        return [[词汇, 语义相似度, 语义相似度和编辑相似度加权分数]]
        '''
        w_vec = self.get_txt_vec(word, mode)
        top_x = self.wv_from_text.most_similar(positive=w_vec, topn=topn)
        # [('消防通道', 0.95)]
        res = [x for x in top_x if x[1]>thres and len(x[0])>1]
        res = [
            [x[0],x[1],(3*x[1]+2*self.norm_edit_score(x[0],word))/5] 
            for x in res]
        res = sorted(res, key=lambda x: x[-1], reverse=True)
        return res

    def cos_sim(self, a, b, mode='wv'):
        '''
        计算两组字符串的余弦相似度
        a,b: 字符串或嵌入向量
        mode: 当预训练词表中没有对应字符串时，生成向量的方式
          'wv'-分词后取预训练向量的加权平均
          'sent'-直接用sbert生成
        '''
        if isinstance(a, str): a = [a]
        if isinstance(b, str): b = [b]
        a_vec = self.get_txt_vec(a,mode) if isinstance(a[0], str) else a
        b_vec = self.get_txt_vec(b,mode) if isinstance(b[0], str) else b
        return np.dot(a_vec, b_vec.T)
    
    def char_wt_score(self, w1, w2):
        '''
        计算两个字符串的字符相似度
        w1: 查询的内容
        w2: 数据库中已有的内容
        '''
        char_wt = np.sqrt(range(1,len(w1)+1))
        char_score = 0
        for i in range(len(w1)):
            if w1[i] in w2:
                char_score += char_wt[i]
        char_score /= np.sum(char_wt)
        return char_score

    def coexist_score(self, query, txt):
        '''计算字符共现得分'''
        co_len = 0
        for word in query:
            if word in txt:
                co_len += len(word)
        return co_len/len(query)

    def norm_edit_score(self, str1, str2):
        '''
        用归一化的编辑距离求字符串之间的相似度
        '''
        matrix = np.zeros((len(str1)+1, len(str2)+1))
        matrix[0,:] = np.arange(len(str2)+1)
        matrix[:,0] = np.arange(len(str1)+1)
        for i in range(1, len(str1)+1):
            for j in range(1, len(str2)+1):
                d = 1 - (str1[i-1] == str2[j-1])
                matrix[i][j] = min(
                    matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+d)
        edit_dis = matrix[len(str1)][len(str2)]/max(len(str1),len(str2))
        return 1-edit_dis

    def lcs_score(self, x, y, use_conv=True):
        '''
        x: 字符串
        y: 字符串
        use_conv: 是否计算聚合度
        计算x和y的最长公共子序列
        x是query，y是数据库中文本
        返回lcs得分、聚合度及lcs匹配结果
        '''
        if not x or not y: return 0, 0, []
        dp = [[0]*(len(y)+1) for _ in range(len(x)+1)]
        curr = 0
        for i in range(1, len(x)+1):
            last = curr
            curr = i
            for j in range(1, len(y)+1):
                if x[i-1]==y[j-1]:
                    dp[curr][j]=dp[last][j-1]+1
                else:
                    dp[curr][j]=max(dp[last][j],dp[curr][j-1])
        i = len(x)
        j = len(y)
        cs = []     # 最大子序列匹配结果在两个序列中的位置编号
        while i>0 and j>0:
            if x[i-1]==y[j-1]:
                if dp[i][j]==dp[i][j-1]:
                    j -= 1
                else:
                    cs.append((i-1, j-1, x[i-1]))
                    i -= 1
                    j -= 1
            else:
                if dp[i-1][j]>dp[i][j-1]:
                    i -= 1
                else:
                    j -= 1
        cs.reverse()
        if len(cs)<=1 and len(x)>1 and len(y)>1: return 0, 0, cs
        if use_conv:
            gap = np.array([cs[i][1]-cs[i-1][1]-1 for i in range(1,len(cs))])
            # 间隔次数越多，聚合度越小
            penal = (gap>0).sum()
            # 间隔字符越多，聚合度越小
            gap_num = gap.sum() + (penal-1)*penal*3
            # lcs 匹配结果的聚合程度，聚合度越高，得分越高
            conv = 1 if gap_num==0 else 1-0.5/(1+1.6**(-(gap_num/2.5-2.5)))
            return len(cs)*conv/len(x), conv, cs
        else:
            return len(cs)/len(x), 1, cs

    def is_word(self, query):
        '''判断query是词汇还是句子'''
        syn = self.get_synon(query)
        return len(syn)>5 and syn[0][2]>0.8

    def word_sim_score(self, w1, w2, use_edit=False, mode='wv'):
        '''
        计算两个字符串的相似度
        w1: 查询的内容
        w2: 数据库中已有的内容
        use_edit: 是否使用编辑距离辅助计算
        wt: 当w由词表中多个词汇组成时，wt为False则取多个词向量的均值，
            wt为True时，用sqrt函数根据词汇位置对词汇向量加权
        '''
        cos_score = self.cos_sim(w1,w2,mode)[0][0]
        if use_edit and len(w1)>3 or len(w2)>3:
            char_score = self.char_wt_score(w1,w2)
            edit_score = self.norm_edit_score(w1,w2)
            wt = 1/2*(1-np.exp(-len(w1)/8))
            score = wt*(char_score+edit_score)/2 + (1-wt)*cos_score
        else:
            score = cos_score
        return score

    def sent_sim_score(self, s1, s2, use_lcs=False, mode='sent'):
        '''
        计算两个句子的相似度
        '''
        score = self.cos_sim(s1,s2,mode)[0][0]
        if use_lcs:
            lcs_score = self.lcs_score(s1,s2,use_conv=False)[0]
            score = 0.6*score+0.4*lcs_score
        return score

    def sent_n_most_sim(self, s1, s2, topn=1, mode='sent'):
        '''找到s2中跟s1最接近的句子'''
        if isinstance(s1, np.ndarray):
            query_embed = s1
        else:
            query_embed = self.get_txt_vec(s1,mode)
        if isinstance(s2, np.ndarray):
            corpus_embed = s2
        else:
            corpus_embed = self.get_txt_vec(s2,mode)
        hit = semantic_search(query_embed, corpus_embed, top_k=topn)
        return hit


if __name__=='__main__':
    word_vec_file = 'D:\Download\dataset\\tencent-arch-zh-d200-v0.2.0-tp200.txt'
    wv_from_text = KeyedVectors.load_word2vec_format(
            word_vec_file, binary=False, no_header=False)
    txt_simer = TxtSim(wv_from_text)