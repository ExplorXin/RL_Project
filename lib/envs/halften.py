import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

# 定义牌的分数，其中，A=1，2~10=牌的点数，J/Q/K=0.5.随机发牌就是随机从deck中选择一张牌
# 人牌：J/Q/K
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
    return sum(1 for i in hand if i == p_val)

#手上的牌是否爆掉
def gt_bust(hand):
    return sum_hand(hand) > dest
# 判断是否刚好达到了十点半
def is_dest(hand):
    return sum_hand(hand) == dest
# 判断是否比十点半小
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
def cmp(dealer,player):
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
        return dealer_num >= player_num



# -----创建十点半的环境-----
class HalftenEnv(gym.Env):
    def __init__(self):
        # 行为空间：停牌，叫牌
        self.action_space = spaces.Discrete(2)
        # 状态空间：(玩家手牌数总分，玩家总牌数，玩家人牌数)
        # 玩家手牌总分数：21个状态
        # 玩家手牌数：5个状态
        # 玩家人牌数：6个状态
        self.observation_space = spaces.Tuple((
            spaces.Discrete(21),
            spaces.Discrete(5),
            spaces.Discrete(6)))
        self._seed()
        
        # 开始牌局
        self.reset()
        
        # 行为数
        self.nA = 2

    #获取随机种子
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    # 基于当前的状态和输入动作，得出下一步的状态、回报、是否结束
    # 1)若动作为叫牌：给玩家发一张手牌，改变玩家手牌状态. 判断玩家当前牌型，返回玩家手牌状态、回报、是否结束
    # 2)若动作为停牌：庄家开始补牌，点数比玩家大，庄家胜，游戏结束，否则继续补牌至分出胜负(点数相同比较手牌，庄家手牌>=玩家，庄家胜，否则继续补牌)
    def step(self, action):
        assert self.action_space.contains(action)
        reward = 0
        terminated = False
        truncated = False

        if action:
            self.player.append(draw_card(self.np_random))
            _, reward, terminated = hand_types(self.player)
        else:
            terminated = True

            # 玩家停牌之后，庄家开始补牌
            self.dealer = draw_hand(self.np_random)
            # 因只有一张手牌，无特殊牌型，只需直接比较大小
            result = cmp(self.dealer, self.player)
            if result:
                reward = -1
            else:
                while not result:
                    # 继续给庄家补牌
                    self.dealer.append(draw_card(self.np_random))
                    # 判断庄家牌型
                    dealer_type, dealer_reward, dealer_done = hand_types(self.dealer)
                    # 出现特殊牌型，终止游戏. 上式为庄家回报，转成玩家回报时应为负值
                    if dealer_done:
                        reward = -dealer_reward
                        break
                    # 还未终止，则对比庄家和玩家的手牌分数
                    result = cmp(self.dealer, self.player)
                    if result:
                        reward = -1
                        break

        return self._get_obs(), reward, terminated, truncated, {}

        # 获取当前状态空间(玩家手牌数总分，玩家总牌数，玩家人牌数)
    def _get_obs(self):
        return (sum_hand(self.player), 
                get_card_num(self.player), 
                get_p_num(self.player))
    
    # 牌局初始化
    # def reset(self, seed=None, option=None):
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player = draw_hand(self.np_random)
        
        return self._get_obs(), {}
