from ltp import LTP
from itertools import chain

class WordSpliter:
    '''
    HED的落点没有COO则认为前面所有东西都是定语
    "的"之前的都当作定语
    如果有多个等， 则在每个等后面的第一个连词处分开

    连词前是中心语且连词前后两部分相似度极低
    连词前的中心语跟前后两个词组分别计算相似度
    为增加精度，当前中心语后加其词性
    词组的中心语跟当前中心语词性相同时在中心语后加词性
    则按连词分
    按连词分后，对每一部分：
    没有等或等之前的东西
        按照COO关系拆解，先找到所有中心语
        确定修饰每个中心语的范围构成词组，对每个词组
            词组中有的，则认为的前面是modi
            找modi中的实体
        若COO中只有第一个有ATT，后面都没有ATT，则定语和每个COO内容做笛卡尔乘积
    等之后的东西
        足够短且跟之前所有元素的相似度都足够低且被之前的元素修饰
            则认为等前的内容跟等后的内容组合起来构成实体
        否则
            认为等前的东西都是等后类型实体的列举，具有包含关系
    '''
    def __init__(self, ltp, word_sim) -> None:
        self.ltp = ltp
        self.word_sim = word_sim

    def get_spd(self, txt):
        result = self.ltp.pipeline([txt], tasks = ["cws","dep","pos"])
        seg = result.cws[0]
        pos = result.pos[0]
        dep = result.dep[0]
        dep = list(zip(range(1,1+len(seg)), dep['head'], dep['label']))
        return seg,pos,dep

    def is_cont_att(self, d):
        '''判断当前word是否修饰下一个word'''
        return d[-1]=='ATT' and d[1]-d[0]==1

    def have_coo(dep, p):
        '''判断当前word是否有COO'''
        for d in dep:
            if d[-1]=='COO' and (d[0]==p or d[1]==p):
                return True
        return False

    def get_break_cont(self, seg, pos, dep):
        '''
        找到所有需要分割的连词
        连词前的词汇是中心语且该词汇的依赖是COO或HED，
        则计算连词前后两个词组的相似度，若小于一定阈值，
        则认为连词前后是两段枚举，可以分开
        返回值是从0开始的位置，不算root
        '''
        break_cont_pos = []
        word_num = len(seg)
        for p in range(word_num):
            # 连词的前一个词是中心语
            if pos[p]=='c' and dep[p-1][2] in ['COO','HED']:
                curr_start = p - 1
                # 找到当前词组的起始位置
                for i in range(p,-1,-1):
                    if dep[i][1]==p and dep[i][2] not in ['WP','LAD']:
                        curr_start = i
                # 构建连词前后的词组
                curr_word = ''.join(seg[curr_start:p])
                next_word = ''.join(seg[p+1:dep[p][1]])
                len_same = len(curr_word)==len(next_word)
                pos_same = pos[p-1]==pos[dep[p][1]-1]
                sim_score = self.word_sim.cos_sim(curr_word, next_word)[0][0]
                # 如果长度相等、词性一致、连词简单且相似度还行，则认为不必拆分
                if len_same and pos_same and seg[p] in '或和' and sim_score>0.4:
                    continue
                if sim_score<0.5: break_cont_pos.append(p)
        return break_cont_pos      

    def find_word_pos(self, seg, val):
        '''找到满足要求的元素所在位置'''
        pos = len(seg)-1
        while pos>-1 and seg[pos]!=val: pos -= 1
        return pos

    def touch_hed(self, dep):
        '''判断最后一个词是否通过COO或HED与Root相连'''
        if dep[-1][-1]=='HED': return True
        p = len(dep) - 1
        while p>0:
            if dep[p][-1]=='COO':
                p = dep[p][1] - 1
            elif dep[p][-1]=='HED':
                return True
            else:
                return False

    def get_ent_in_modi(self, modi):
        '''提取定语中的实体'''
        if not modi: return ['','']
        seg,pos,dep = self.get_spd(modi)
        if pos[-1] in ['m','q']: return ['','']
        start = 0
        end = len(seg)

        if not self.touch_hed(dep):
            while start<end and pos[start] in ['nd','v','p']:
                start += 1
        while end>start and pos[end-1] in ['nd','p'] or \
            (pos[end-1]=='v' and dep[-1][-1]=='HED'):
            end -= 1
        if dep[end-1][-1]=='VOB':
            start = max(start,dep[end-1][1]-1)
        return ['', ''.join(seg[start:end])]

    def split_ent_base(self, seg, pos, dep):
        '''
        直接按照COO进行划分的基础分割函数
        return: [
            ['', '设备机房'], ['', '电梯机房'], 
            ['', '水箱间'], ['', '天线']
        ]
        '''
        def build_T2H(dep):
            '''
            根据依存树构建每个词的发射字典
            txt: 设备机房、电梯机房、水箱间、天线
            dep: [
                (1, 2, 'ATT'), (2, 0, 'HED'), (3, 5, 'WP'), 
                (4, 5, 'ATT'), (5, 2, 'COO'), (6, 7, 'WP'), 
                (7, 2, 'COO'), (8, 9, 'WP'), (9, 2, 'COO')
            ]
            return {
                2: {'ATT': [1], 'COO': [5, 7, 9]}, 
                0: {'HED': [2]}, 
                5: {'ATT': [4]}
            }
            '''
            dep_T2H = {}
            for d in dep:
                if d[2] in ['WP','LAD']: continue
                if d[1] in dep_T2H:
                    dep_T2H[d[1]][d[2]] = dep_T2H[d[1]].get(d[2],[])+[d[0]]
                else:
                    dep_T2H[d[1]] = {d[2]:[d[0]]}
            return dep_T2H
        
        def find_smallest(dep_T2H, p):
            '''
            寻找当前中心语的修饰覆盖的范围
            由于不存在交叉的情况，所以只要往前找最小的即可
            return: 第一个词对应的位置
            '''
            if p==0: return 1
            p_out = list(dep_T2H.get(p,{}).values())
            p_out = list(chain(*p_out))
            if not p_out or min(p_out)>p: return p
            smallest = find_smallest(dep_T2H, min(p_out))
            return smallest
        
        def find_modi_attn(attn):
            '''根据标志词“的”划分出定语和实体的内含部分'''
            modi = []
            if '的' in attn:
                modi_end = self.find_word_pos(attn,'的')
                modi = attn[:modi_end]
                attn = attn[modi_end+1:]
            modi = ''.join(modi)
            attn = ''.join(attn)
            return modi, attn

        # 构建发射字典
        dep_T2H = build_T2H(dep)
        # 获取所有中心语所在位置
        head = dep_T2H[0]['HED'][0]
        coo_list = [head] + dep_T2H.get(head,{}).get('COO',[])
        # 每个元素都向前找直到断点作为该词组的范围
        ent_span = []
        for p in coo_list:
            start = find_smallest(dep_T2H,p)
            if p>head and dep_T2H.get(p,{}).get('COO',[]):
                p = max(dep_T2H[p]['COO'])
            ent_span.append([start, p])
        
        ent_list = []
        # 若并列中心语都没有att则与第一个词组共享修饰语
        if all([s[1]==s[0] for s in ent_span[1:]]):
            s = ent_span[0]
            attn = seg[s[0]-1:s[1]-1]
            modi, attn = find_modi_attn(attn)
            ent_list = [[modi, attn+seg[s[1]-1]] for s in ent_span]
        # 若后面的中心语有att则分别求定语
        else:
            for s in ent_span:
                ent = seg[s[0]-1:s[1]]
                modi, ent = find_modi_attn(ent)
                ent_list.append([modi, ent])
        
        modi_list = list(set([e[0] for e in ent_list]))
        modi_ent_list = [self.get_ent_in_modi(m) for m in modi_list]
        modi_ent_list = [m for m in modi_ent_list if m[1]]
        return ent_list, modi_ent_list

    def is_ellip(self, lis_pre, lis_post):
        '''
        判断是否为省略枚举的实体
        如果是，等后面的词要跟前面的词逐个拼接
        lis_pre: [['', '国道'], ['', '省道']], 等前面的词
        lis_post: [['', '干线公路']], 等后面的词
        '''
        # 足够短且相似度足够低则认为要拼接
        if len(lis_post)==1 and len(lis_post[0][1])<4 and not lis_post[0][0]:
            word_post = lis_post[0][1]
            word_pre = [e[1] for e in lis_pre]
            sim = self.word_sim.cos_sim(word_post, word_pre)
            if max(sim[0])<0.4: return True
        return False

    def split_ent(self, txt, ent_list, modi_ent_list, incl_tri):
        '''分割长实体'''
        if txt[:2]=='下列' and len(txt)<8: 
            return [['',txt]],[],[]
        seg,pos,dep = self.get_spd(txt)
        # print('151', seg,pos,dep)
        word_num = len(seg)
        '''HED的落点没有COO且实体前为‘的’则认为‘的’之前都是定语'''
        if dep[-1][-1] == 'HED':
            p = word_num-2
            while p>-1 and self.is_cont_att(dep[p]): p -= 1
            if seg[p]=='的':
                modi = ''.join(seg[:p+1])
                ent = ''.join(seg[p+1:])
                ent_list.append([modi, ent])
                p -= 1
                while p>-1 and pos[p]=='v': p -= 1
                modi = ''.join(seg[:p+1])
                if modi:
                    modi_ent1,modi_ent2,incl_t = self.split_ent(modi,[],[],[])
                    modi_ent_list += modi_ent1
                    modi_ent_list += modi_ent2
                    incl_tri += incl_t
                return ent_list, modi_ent_list, incl_tri
        
        deng_pos = []
        for i in range(word_num):
            if seg[i] == '等': deng_pos.append(i)
        '''如果有多个等， 则在每个等后面的第一个连词处分开'''
        if len(deng_pos)>1:
            split_p = [-1]
            for p in deng_pos[:-1]:
                while p<word_num:
                    if pos[p]=='c' or seg[p]=='、':
                        split_p.append(p)
                        break
                    p += 1
            split_p += [len(seg)]
            for i in range(len(split_p)-1):
                txt_ = ''.join(seg[split_p[i]+1:split_p[i+1]])
                # print('189', txt_)
                ent_l,modi_ent_l,incl_t = self.split_ent(txt_,[],[],[])
                ent_list += ent_l
                modi_ent_list += modi_ent_l
                incl_tri += incl_t
            return ent_list, modi_ent_list, incl_tri
        
        '''找到可以分割的连词'''
        break_cont_pos = [-1]
        break_cont_pos += self.get_break_cont(seg,pos,dep)
        break_cont_pos += [word_num]
        bcp_num = len(break_cont_pos)
        '''处理连词分割后的每一部分'''
        for i in range(bcp_num-1):
            seg_ = seg[break_cont_pos[i]+1:break_cont_pos[i+1]]
            # 有等则分等前等后，没等则直接split_ent_base
            if '等' in seg_:
                p = self.find_word_pos(seg_, '等')
                # 处理等前词组
                txt_pre = ''.join(seg_[:p])
                # print('205', txt_pre, seg_)
                ent_l_pre,modi_ent_l,_ = self.split_ent(txt_pre,[],[],[])
                modi_ent_list += modi_ent_l
                # 处理等后的词组
                txt_post = ''.join(seg_[p+1:])
                seg_,pos_,dep_ = self.get_spd(txt_post)
                ent_l_post,modi_ent_l = self.split_ent_base(seg_,pos_,dep_)
                modi_ent_list += modi_ent_l
                # 如果是省略类型，则等前部分要和等后部分拼接
                if self.is_ellip(ent_l_pre,ent_l_post):
                    word_post = ent_l_post[0][1]
                    for e in ent_l_pre:
                        e[1] += word_post
                    ent_list += ent_l_pre
                # 如果是枚举类型，则等后部分和等前部分具有包含关系
                else:
                    ent_list += ent_l_pre
                    ent_list += ent_l_post
                    incl_tri = [[ent_l_post[0],e] for e in ent_l_pre]
            # 其余情况用split_ent_base处理
            else:
                if bcp_num>2:
                    txt_ = ''.join(seg_)
                    seg_,pos_,dep_ = self.get_spd(txt_)
                    ent_l,modi_ent_l = self.split_ent_base(seg_,pos_,dep_)
                else:
                    ent_l,modi_ent_l = self.split_ent_base(seg,pos,dep)
                ent_list += ent_l
                modi_ent_list += modi_ent_l
        return ent_list,modi_ent_list,incl_tri


if __name__=='__main__':
    import jieba
    from gensim.models import KeyedVectors
    from model_TS import TxtSim
    txts = [
        '非水溶性液体外浮顶储罐、内浮顶储罐、直径大于18m的固定顶储罐及水溶性甲、乙、丙类液体立式储罐',
        '高架仓库或高层仓库',
        '设备机房、电梯机房、水箱间、天线等突出物',
        '国道、省道等干线公路及快速路等道路',
        '儿童活动场所、老年人照料设施中的老年人活动场所、医疗建筑中的治疗室和病房、教学建筑中的教学用房',
        '医疗建筑中的治疗室和病房',
        '地下室的底板、外墙以及上部有覆土的地下室顶板',
        '生产过程中散发的可燃气体、蒸气、粉尘或纤维与供暖管道、散热器表面接触能引起燃烧的场所',
        '国道、省道等干线公路及快速路等道路',
        '配件加工、修制和修车材料、燃料的储存、发放',
        '乙、丙、丁、戊类仓库、民用建筑',
        '入侵和紧急报警系统、视频监控系统、出入口控制系统、停车库（场）安全管理系统',
        '客运管理、乘客信息管理、设备维修及信息管理等运营调度和指挥功能',
        '修车材料、燃料的储存、发放',
        '住宅建筑内的汽车库、锅炉房和建筑中的下列场所',
        '附设在工业与民用建筑内的可燃油油浸变压器、充有可燃油的高压电容器和多油开关'
    ]
    word_vec_tenc = 'D:\Download\ArchData\data\PreData\\arch-zh-d200-tencent-tp200.txt'
    wv_from_text_word = KeyedVectors.load_word2vec_format(
        word_vec_tenc, binary=False, no_header=False)
    ltp = LTP('LTP/small')
    word_sim = TxtSim(wv_from_text_word, jieba)
    ent_spliter = WordSpliter(ltp, word_sim)
    for txt in txts:
        print('261',txt)
        a,b,c=ent_spliter.split_ent(txt,[],[],[])
        print(a,b)