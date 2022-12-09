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
    f3 = False
    f4 = idx>0 and seg[idx-1]=='的'                     # 前面是“的”则认为该词可以做中心语：本规范适用于新建、扩建、改建的民用与工业建筑中自动喷水灭火系统的设计。
    if dep[idx][2]=='COO':
        f3 = is_anchor_word(seg, pos, dep, dep[idx][1]-1)
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
    
