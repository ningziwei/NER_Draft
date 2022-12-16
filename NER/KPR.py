'''
Key Phrase Recognition
关键词抽取
1）定中短语抽取；
2）定语识别，区分定语从句、实体定语和非实体定语；
3）删除定语从句后重新确定关键词范围，得到更合理的并列关系；并列中心语识别
4）中心语分类，实体、属性；
'''
from itertools import chain


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

def find_biggest(dep_T2H, p):
    '''
    寻找当前中心语的覆盖范围
    由于不存在交叉的情况，所以只要往后找最大的即可
    return: 最后一个词对应的位置
    '''
    p_out = list(dep_T2H.get(p,{}).values())
    p_out = list(chain(*p_out))
    if not p_out or max(p_out)<p: return p
    biggest = find_biggest(dep_T2H, max(p_out))
    return biggest

def have_dep(idx, tag, dep, dep_T2H):
    '''判断当前词是否做对应成分'''
    f1 = tag in dep_T2H[idx+1]
    f2 = False
    if dep[idx][2] == 'COO':
        f2 = have_dep(dep[idx][1]-1, tag, dep, dep_T2H)
    return f1 or f2

def is_skip_word(dep, dep_T2H, idx):
    '''判断当前词是否需要被跳过'''
    # 状语，设置泄压设施时，泄压部位应能在爆炸作用达到结构最大耐受压强前泄压，其泄压方向不应朝向人员聚集的场所和人行通道；
    # 不是方位名词时基本是条件状语，不适合以此为中心语提取短语
    # 在温度大于100时，加油站应设在供应站和调压站、加油站；当地铁站位于道路下方，加油站应设在供应站和调压站、加油站
    # if dep[i][2]=='POB' and pos[i]!='nd': continue
    # 在体育馆内部，体育场要打开抽湿器，关闭加热器
    # 当温度大于30，湿度大于50时，体育场要打开抽湿器，关闭加热器
    f1 = dep[idx][2]=='ADV'
    f2 = False
    f3 = False
    # f4 = idx>0 and seg[idx-1]=='的'
    if dep[idx][2]=='POB':                              # 介宾短语中宾语的修饰语有主谓结构：当温度大于30，湿度大于50时，体育场要打开抽湿器，关闭加热器
        att_idx = dep_T2H[idx+1].get('ATT', [-1])[0]
        for tag in ['SBV','FOB','VOB']:
            if tag in dep_T2H[att_idx]:
                f2 = True
                break
    for tag in ['SBV','FOB','VOB']:                     # 谓语不可能是主宾语的中心词：当温度大于30，湿度大于50时，我们要保证体育场打开抽湿器，关闭加热器
        if have_dep(idx, tag, dep, dep_T2H):
            f3 = True
            break
    return f1 or f2 or f3
    
def is_anchor_word(seg, pos, dep, idx):
    '''判断当前词是否可以当作锚点'''
    f1 = 'n' in pos[idx] or pos[idx] in ['m', 'q']      # 数词和量词可以当中心点：应根据本规范第2.1节进行合规性判定。
    f2 = dep[idx][2] in ['SBV','FOB','VOB']
    f3 = idx>0 and seg[idx-1]=='的'                     # 前面是“的”则认为该词可以做中心语：本规范适用于新建、扩建、改建的民用与工业建筑中自动喷水灭火系统的设计。
    f4 = False
    if dep[idx][2]=='COO':
        f4 = is_anchor_word(seg, pos, dep, dep[idx][1]-1)
    return f1 or f2 or f3 or f4

def is_concat_coo(dep, pre_span, idx):
    '''
    判断当前范围是否需要跟上一个范围合并
    如果当前词的COO能链接到上一个span，则合并
    如果当前范围跟上一个范围中间是“的”，则合并
    '''
    f1 = dep[idx][2]=='COO'
    f2 = pre_span and pre_span[0]<=dep[idx][1]<=pre_span[1]
    return f1 and f2

def is_concat_uatt(seg, pre_span, start):
    '''
    判断当前范围是否需要跟上一个范围合并
    如果当前词的COO能链接到上一个span，则合并
    如果当前范围跟上一个范围中间是“的”，则合并
    '''
    f1 = start>2 and seg[start-2]=='的'
    f2 = pre_span and pre_span[-1]==start-1
    # if pre_span and seg[start-1]=='建筑':
    #     print('88', seg[start-1], seg[start-2])
    #     print(f1,f2)
    #     print(pre_span[-1],start)
    return f1 and f2

def get_so(seg, pos, dep, dep_T2H):
    '''
    抽取定中短语，attribute-head phrase
    '''
    phrase_spans = [[]]
    for i in range(len(pos)):
        if is_skip_word(dep, dep_T2H, i):
            continue
        if is_anchor_word(seg, pos, dep, i):
            start = find_smallest(dep_T2H, i+1)
            while True:                                                 # 在找当前范围的同时判断之前的是否被覆盖
                if phrase_spans[-1] and start<=phrase_spans[-1][0]:
                    phrase_spans.pop()
                else:
                    break
            end = find_biggest(dep_T2H, i+1)
            pre_span = phrase_spans[-1]
            if pre_span and start>=pre_span[0] and end<=pre_span[1]:
                continue
            if is_concat_coo(dep, pre_span, i):
                start = pre_span[0]
                phrase_spans.pop()
            phrase_spans.append([start, end])
    
    phrase_spans_ = [[]]
    for i in range(1, len(phrase_spans)):
        if is_concat_uatt(seg, phrase_spans_[-1], phrase_spans[i][0]):
            phrase_spans_[-1][1] = phrase_spans[i][1]
        else:
            phrase_spans_.append(phrase_spans[i])
    phrase_spans = phrase_spans_[1:]
    print('131', phrase_spans)
    # phrases = [seg[s[0]-1:s[1]] for s in phrase_spans]
    # print('noun phrases', phrases)
    return phrase_spans
    
def split_AHP():
    '''
    分别得到定语和中心语
    定语类别：Null 形容词 实体 定语从句 定中短语
    中心语类别：实体 属性
    给水厂应对制水生产中的主要设施、设备制定和实施巡查维护保养制度；应对主要工艺运行情况及其运行中的动态技术参数，制定和实施质量控制点检验制度。
    ['生产', '、', '运输', '、', '储存', '、', '使用', '的', '过程', '中']
    ['各', '建（构）', '筑物', '的', '功能', '、', '运行', '和', '维护', '的', '要求']
    ['以及', '必要', '的', '试验', '验证'], ['相似', '条件', '下', '已', '有', '的', '运行', '经验']
    X的A、B和C
    去掉长定语得到正常
    给水厂的设计规模应满足供水范围规定年限内最高日的综合生活用水量、工业企业用水量、浇洒道路和绿地用水量、管网漏损水量及未预见用水量的要求，并应考虑非常规水资源利用引起的规模降低。
    综合生活用水量 工业企业用水量 浇洒道路 绿地用水量 管网漏损水量 未预见用水量
    ['供水', '范围', '规定', '年限', '内', '最高', '日', '的', '综合', '生活', '用水量', '、', '工业', '企业', '用水量', '、', '浇洒', '道路', '和', '绿地', '用水量'], ['、', '管网'], ['及', '未', '预见', '用水量']
    ['国家', '规定', '的', '地下水', '质量', '标准', '中', 'Ⅰ', '、', 'II', '类']
    ['地表水', '环境', '质量', '标准', '中', 'II', '类']
    ['两', '个'], ['及', '以上', '可', '独立', '运行', '的', '系列', '或', '分格']
    A0的A和B0的B
    ['盛水', '构筑物', '上', '所有', '可', '触及', '的', '外露', '导电', '部件', '和', '进出', '构筑物', '的', '金属', '管道']
    
    给水厂的设计规模应满足供水范围规定年限内最高日的综合生活用水量、工业企业用水量、浇洒道路和绿地用水量、管网漏损水量及未预见用水量的要求
    # 1.在原句的句法分析结果中找到从句，判断从句类别，定语、状语、主语、宾语
    # 乙、丙、丁、戊类仓库、民用建筑
    # 2.与前后两个相邻词相似度极高则删掉间隔
    # 地下水质量标准中Ⅰ、II 类
    # 3.与后面第一个词的相似度明显高于最后一个词的相似度则删掉间隔
        
    去掉括号里的东西
        保留括号内的东西
            原料库房与设备间均应有保持良好通风的设备，换气次数应为（8~12）次/h
        括号内很长，是对前文的解释
            大面积的多层地下建筑物（如地下车库、商场、运动场等）
            复杂地质条件下的坡上建筑物（包括高边坡）
            基坑工程、边坡工程设计时，应根据支护（挡）结构破坏可能产生的后果（危及人的生命、造成经济损失、对社会或环境产生影响等）的严重性，采用不同的安全等级。
        括号内很短，是对前文的替换，多了个抽取结果
            支护（挡）结构安全等级的划分应符合表 2.2.4 的规定。
            所有建（构）筑物的地基计算均应满足承载力要求；
        括号内很短，是对前文的替换，对抽取结果无影响
            土和（或）水对建筑材料的腐蚀性；
    
    除农村小型集中式供水，城乡给水工程中的取水工程、净（配）水工程、转输厂站的供电负荷等级不应低于表2.0.24的规定
    
    得到定中短语后，提取“的”引导的定语
        只有头部的一个，则分别看定语和中心语是否有连词
        整体看有连词则扔到后面做分辨
        没有连词则作为一个整体即可
    对于分离出来的定语
        定语从句则直接分辨出三元组
        定中短语则递归调用一下本函数
    并列则直接执行以下操作；长定短中，则取长定执行下面操作

        建筑无法设置泄压设施或泄压面积不符合要求时，建筑中存在可燃气体、蒸气、粉尘或纤维爆炸危险的部位的建筑承重结构应满足抗爆要求。
    1.去掉长状语
        建筑中存在可燃气体、蒸气、粉尘或纤维爆炸危险的部位的建筑承重结构应满足抗爆要求。

    用原句中对应部分的句法分析结果做如下处理，不断简化句子，引导ltp得到正确的解析结果
        给水厂的设计规模应满足供水范围规定年限内最高日的综合生活用水量、工业企业用水量、浇洒道路和绿地用水量、管网漏损水量及未预见用水量的要求
        保养厂应能承担营运车辆的高级保养任务及相应的配件加工、修制和修车材料、燃料的储存、发放等
        控制中心应具备行车调度、电力调度、环境与设备调度、防灾指挥、客运管理、乘客信息管理、设备维修及信息管理等运营调度和指挥功能，并应对城市轨道交通系统运营的全过程进行集中监控和管理
        给水厂应对制水生产中的主要设施、设备制定和实施巡查维护保养制度，应对主要工艺运行情况及其运行中的动态技术参数，制定和实施质量控制点检验制度
        建筑中存在可燃气体、蒸气、粉尘或纤维爆炸危险的部位的建筑承重结构应满足抗爆要求。
    1.去掉解析范围内部“的”引导的定语后重新做句法解析，做好定语和词语的对应
    1.去掉“的”引导的长定语、“的”引导的动词短定语-所有定语都是动词，去掉无用的“等”，连续并列合并
        给水厂的设计规模应满足综合生活用水量、工业企业用水量、浇洒道路和绿地用水量、管网漏损水量及未预见用水量的要求
        保养厂应能承担高级保养任务及配件加工、修制和修车材料、燃料的储存
        控制中心应具备行车调度、电力调度、环境与设备调度、防灾指挥、客运管理、乘客信息管理、设备维修及信息管理等运营调度功能，并应对全过程进行集中监控
        建筑中存在可燃气体或纤维爆炸危险的部位的建筑承重结构应满足抗爆要求。
    2.去掉解析范围内部动词做定语的部分，并做好定语和词语的对应
    2.去掉“ATT”引导的动词短定语-所有定语都是动词，连续并列合并(直接COO且都是单词)，去掉短定语(被这一个词单独修饰)
        给水厂的设计规模应满足用水量、工业企业用水量、道路和绿地用水量、管网漏损水量及未预见用水量的要求
        保养厂应能承担高级保养任务及配件加工、修制和材料、燃料的储存
        控制中心应具备行车调度、电力调度、环境与设备调度、指挥、客运管理、乘客信息管理、设备维修及信息管理等运营调度功能，并应对全过程进行集中监控
        建筑中存在气体或纤维爆炸危险的部位的建筑承重结构应满足抗爆要求。
    3.离得很近的简单并列只保留第一个，去掉无用的“等”，然后重新做句法分析
    3.连续并列合并
        给水厂的设计规模应满足用水量、工业企业用水量、道路用水量、管网漏损水量及未预见用水量的要求
        保养厂应能承担高级保养任务及配件加工和材料的储存
        控制中心应具备行车调度、电力调度、环境与设备调度、指挥、客运管理、乘客信息管理、设备维修及信息管理等运营调度功能，并应对全过程进行集中监控
        建筑中存在气体爆炸危险的部位的建筑承重结构应满足抗爆要求。
    4.用现有方法做并列解析
    # 4.每个连词前中心语考虑语义、词性、字符相似度来计算
    # 相似度计算：综合考虑语义、词性和字符相似度
    '''

'''
多个主句，分析完一个后再分析下一个
    化验室所用的计量分析仪器必须定期进行计量检定，经检定合格方可使用。
情态动词可能在一个状语之前
    水处理药剂必须计量投加。
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
        for p in range(len(seg)):
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


    def split_ent_base(self, seg, pos, dep):
        '''
        直接按照COO进行划分的基础分割函数
        return: [
            ['', '设备机房'], ['', '电梯机房'], 
            ['', '水箱间'], ['', '天线']
        ]
        '''
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
        
        ent_list = [[seg[s[0]-1:s[1]]] for s in ent_span]
        # for s in ent_span:
        #     ent_list.append([seg[s[0]-1:s[1]]])

        return ent_span, ent_list

    def is_ellip(self, lis_pre, lis_post):
        '''
        判断是否为省略枚举的实体
        如果是，等后面的词要跟前面的词逐个拼接
        国道、省道等干线公路及快速路等道路
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

    def split_ent_(self, txt, ent_span, ent_list, incl_tri):
        '''
        分割长实体
        建筑物设计应包括平面与空间布局、结构和门窗等与风险防范相关的内容。
        句法分析完全错误：实体装置设计应包括安防设备的自身实体保护和保护目标的近身式保护箱等内容。
        局部突出屋面的楼梯间、电梯机房、水箱间等辅助用房水平投影面积占屋顶平面面积不超过1/4者
        '''
        if txt[:2]=='下列' and len(txt)<8: 
            return [['',txt]],[],[]
        seg,pos,dep = self.get_spd(txt)
        # print('151', seg,pos,dep)
        word_num = len(seg)
        
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
                ent_l, incl_t = self.split_ent(txt_,[],[],[])
                ent_list += ent_l
                incl_tri += incl_t
            return ent_list, incl_tri
        
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
                ent_s_pre, ent_l_pre, _ = self.split_ent(txt_pre,[],[],[])
                # 处理等后的词组
                txt_post = ''.join(seg_[p+1:])
                seg_,pos_,dep_ = self.get_spd(txt_post)
                ent_s_post, ent_l_post = self.split_ent_base(seg_,pos_,dep_)
                # 如果是省略类型，则等前部分要和等后部分拼接
                if self.is_ellip(ent_l_pre, ent_l_post):
                    for i in ent_l_pre:
                        ent_s_pre[i] += ent_s_post[0]
                        ent_l_pre[i] += ent_l_post[0]
                    ent_span += ent_s_pre
                    ent_list += ent_l_pre
                # 如果是枚举类型，则等后部分和等前部分具有包含关系
                else:
                    ent_span += ent_s_pre
                    ent_span += ent_s_post
                    ent_list += ent_l_pre
                    ent_list += ent_l_post
                    incl_tri = [[ent_l_post[0],e] for e in ent_l_pre]
            # 其余情况用split_ent_base处理
            else:
                if bcp_num>2:
                    txt_ = ''.join(seg_)
                    seg_,pos_,dep_ = self.get_spd(txt_)
                    ent_s, ent_l = self.split_ent_base(seg_,pos_,dep_)
                else:
                    ent_s, ent_l = self.split_ent_base(seg,pos,dep)
                ent_span += ent_s
                ent_list += ent_l
        return ent_span, ent_list, incl_tri

    def split_ent(self, txt, span, ent_span, ent_list, incl_tri):
        '''
        先做句子预处理，得到so_span的部分，多个SBV、VOB的范围若连续则当作一个
        简单可分则直接分：短语块只有不超过一个等，按连词分后类型都一样，则直接按连词分
        有等则用此模块：一个等需要判断是否有包含关系；多个等需要判断合理的断开位置
        否则用前向后向算法：处理到足够简单后，直接用并列关系得到结果即可
        
        分割长实体
        建筑物设计应包括平面与空间布局、结构和门窗等与风险防范相关的内容。
        句法分析完全错误：实体装置设计应包括安防设备的自身实体保护和保护目标的近身式保护箱等内容。
            短语块只有不超过一个等，按连词分后类型都一样，则直接按连词分
        局部突出屋面的楼梯间、电梯机房、水箱间等辅助用房水平投影面积占屋顶平面面积不超过1/4者
        '''
        if txt[:2]=='下列' and len(txt)<8:
            return [['',txt]],[],[]
        seg,pos,dep = self.get_spd(txt)
        # print('151', seg,pos,dep)
        word_num = len(seg)
        
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
                ent_l, incl_t = self.split_ent(txt_,[],[],[])
                ent_list += ent_l
                incl_tri += incl_t
            return ent_list, incl_tri
        
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
                ent_s_pre, ent_l_pre, _ = self.split_ent(txt_pre,[],[],[])
                # 处理等后的词组
                txt_post = ''.join(seg_[p+1:])
                seg_,pos_,dep_ = self.get_spd(txt_post)
                ent_s_post, ent_l_post = self.split_ent_base(seg_,pos_,dep_)
                # 如果是省略类型，则等前部分要和等后部分拼接
                if self.is_ellip(ent_l_pre, ent_l_post):
                    for i in ent_l_pre:
                        ent_s_pre[i] += ent_s_post[0]
                        ent_l_pre[i] += ent_l_post[0]
                    ent_span += ent_s_pre
                    ent_list += ent_l_pre
                # 如果是枚举类型，则等后部分和等前部分具有包含关系
                else:
                    ent_span += ent_s_pre
                    ent_span += ent_s_post
                    ent_list += ent_l_pre
                    ent_list += ent_l_post
                    incl_tri = [[ent_l_post[0],e] for e in ent_l_pre]
            # 其余情况用split_ent_base处理
            else:
                if bcp_num>2:
                    txt_ = ''.join(seg_)
                    seg_,pos_,dep_ = self.get_spd(txt_)
                    ent_s, ent_l = self.split_ent_base(seg_,pos_,dep_)
                else:
                    ent_s, ent_l = self.split_ent_base(seg,pos,dep)
                ent_span += ent_s
                ent_list += ent_l
        return ent_span, ent_list, incl_tri

