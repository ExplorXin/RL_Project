import gymnasium as gym
from gymnasium import spaces
from gym.utils import seeding

# 定义牌的分数，其中，A=1，2~10=牌的点数，J/Q/K=0.5.随机发牌就是随机从deck中选择一张牌
deck = [1,2,3,4,5,6,7,8,9,10,0.5,0.5,0.5]
#人牌值
p_val = 0.5
#限制值
dest = 10.5
#随机发牌，随机从deck中选择一张牌
def draw_card(np_random):
    return np_random.choice(deck)
#随机发到手一张牌
def draw_hand(np_random):
    return [draw_card(np_random)]
#当前手牌总分
def sum_hand(hand):
    return sum(hand)
#获取手牌的数量
def get_card_num(hand):
    return len(hand)
#获取手牌中的人牌数
def get_p_num(hand):
    count =0
    for i in hand:
        if i == p_val:
            count += 1
    return count

#手上的牌是否爆掉
def gt_bust(hand):
    return sum_hand(hand) > dest
# 判断是否刚好达到了十点半
def is_dest(hand):
    return sum_hand(hand) == dest
# 判断是否比比十点半小
def lt_dest(hand):
    return sum_hand(hand) < dest

# 判断是否为人五小（手中牌为5张，且都为人牌）
def is_rwx(hand):
    return True if get_p_num(hand) == 5 else False
# 判断是否为天王（手中牌为5张，且牌面点数总和为十点半）
def is_tw(hand):
    return True if get_card_num(hand) == 5 and is_dest(hand) else False
# 判断是否为五小（手中牌为5张，且总点数小于十点半）
def is_wx(hand):
    return True if get_card_num(hand) == 5 and lt_dest(hand) else False

# 根据手牌返回结果（牌型、回报、结束状态）
def hand_types(hand):
    # 默认为平牌
    type = 1
    reward = 1
    done = False

    if gt_bust(hand):
        # 爆牌
        type = 0
        reward = -1
        done = True
    elif is_rwx(hand):
        # 人五小
        type = 5
        reward = 5
        done = True
    elif is_tw(hand):
        # 天王
        type = 4
        reward = 4
        done = True
    elif is_wx(hand):
        # 五小
        type = 3
        reward = 3
        done = True
    elif is_dest(hand):
        # 十点半
        type = 2
        reward = 2
        done = True
    return type,reward,done

# 庄家和玩家比较手牌
def com(dealer,player):
    # 规则：庄家大，返回True，玩家大，返回Fasle，当点数相同比较手牌，庄家手牌数<=玩家，返回Fasle，大于则返回True
    dealer_score = sum_hand(dealer)
    player_score = sum_hand(player)
    if dealer_score > player_score:
        return True
    elif dealer_score < player_score:
        return False
    else:
        dealer_num = get_card_num(dealer)
        player_num = get_card_num(player)
        return True if dealer_num >= player_num else False
    
# 创建十点半的环境
class HalftenEnv(gym.Env):
    def __init__(self):
        # 行为空间：停牌，叫牌