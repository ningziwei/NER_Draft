from ltp import LTP
from itertools import chain

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

class RuleNER:
    '''
    用规则系统做NER
    基本上所有以名词为中心语的短语都可以当作实体先抽出来
    '''
    def __init__(self, ltp) -> None:
        self.ltp = ltp

    def get_spd(self, txt):
        result = self.ltp.pipeline([txt], tasks = ["cws","dep","pos"])
        seg = result.cws[0]
        pos = result.pos[0]
        dep = result.dep[0]
        return seg, pos, dep
    
    def get_noun_phrases(self, txt):
        '''提取出以名词为中心语的短语'''
        seg, pos, dep = self.get_spd(txt)
        # 构建发射字典
        dep_T2H = build_T2H(dep)
        # for i in range()


if __name__=='__main__':
    ltp = LTP('LTP/base')
    rule_ner = RuleNER(ltp)
    txt = '建筑的选址和总平面布局应符合减小火灾危害，方便灭火救援的要求，并应符合下列规定'
    print(rule_ner.get_spd(txt))
