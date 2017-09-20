# -*- coding: utf-8 -*-
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import time

# 取得Function id，使用actions.FunctionCall來呼叫Function
_BUILD_PYLON         = actions.FUNCTIONS.Build_Pylon_screen.id
_BUILD_GATEWAY       = actions.FUNCTIONS.Build_Gateway_screen.id
_NOOP                = actions.FUNCTIONS.no_op.id
_SELECT_POINT        = actions.FUNCTIONS.select_point.id
_TRAIN_ZEALOT        = actions.FUNCTIONS.Train_Zealot_quick.id
_RALLY_UNITS_MINIMAP = actions.FUNCTIONS.Rally_Units_minimap.id
_SELECT_ARMY         = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP      = actions.FUNCTIONS.Attack_minimap.id

# 取得Screen Feature中的敵我資訊，以及單位的type id
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index # value in [0, 4], denoting [background, self, ally, neutral, enemy]
_UNIT_TYPE       = features.SCREEN_FEATURES.unit_type.index # 超長的list，將所有單位的id都列了出來

# Unit IDs
# https://github.com/Blizzard/s2client-api/blob/master/include/sc2api/sc2_typeenums.h
_PROTOSS_GATEWAY = 62
_PROTOSS_NEXUS   = 59
_PROTOSS_PYLON   = 60
_PROTOSS_PROBE   = 84

# Parameters
_PLAYER_SELF = 1 # 用於_PLAYER_RELATIVE中來取得自己的座標
_SUPPLY_USED = 3 # https://github.com/deepmind/pysc2/blob/master/docs/environment.md
_SUPPLY_MAX  = 4 # 在Structured的General player information中，3為目前人口數，4為人口上限，用於還沒滿人口時就持續造兵
_SCREEN      = [0]
_MINIMAP     = [1] # 大地圖編號0，小地圖編號1
_QUEUED      = [1] # add to queue  https://github.com/deepmind/pysc2/blob/master/pysc2/lib/actions.py#L204
_SELECT_ALL  = [0]

class RuleBaseAgent(base_agent.BaseAgent):
    nexus_top_left   = None
    pylon_built      = False
    probe_selected   = False
    gateway_built    = False
    gateway_selected = False
    gateway_rallied  = False
    army_selected    = False
    army_rallied     = False

    # 由於Simple64出生點只會是左上或右下，因此寫一個簡單的function輔助我們擺放建築物
    # 若不是在左上角，星核的位置就會是比較大的數字，因此使用減法萊表示建造的建築物是
    # 放在星核的左方或上方。反之若在左上，就建在星核的右方與下方。
    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.nexus_top_left:
            return [x - x_distance, y - y_distance]
        else:
            return [x + x_distance, y + y_distance]

    # 改寫step函數，由一連串條件式組成來操控agent
    def step(self, obs):
        super(RuleBaseAgent, self).step(obs)

        # 觀察速度，若不設置會跑很快，想快速做實驗的話此行可以註解掉
        time.sleep(0.05)

        # 取得星核在左上或是右下的資訊
        if self.nexus_top_left is None:
            # 從observation的minimap的其中一個feature來取得自己基地的座標，
            # 由於知道地圖是Simple64，因此若在左上則y座標會小於31
            nexus_y, nexus_x = (obs.observation["minimap"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.nexus_top_left = nexus_y.mean() <= 31
            print('minimap nexus_y:', nexus_y.mean())
            # return minimap nexus_y: [21 21 21 21 21 22 22 22 22 23 23 23 23 24 24 24 24 25]

        # rule 1: 如果水晶塔還沒建造且探測機還沒被圈選，就圈選探測機
        # 如果水晶塔還沒建造但探測機已經被圈選了，就建造水晶塔
        if not self.pylon_built:
            if not self.probe_selected:
                unit_type = obs.observation["screen"][_UNIT_TYPE] # 列出screen上所有單位
                probe_y, probe_x = (unit_type == _PROTOSS_PROBE).nonzero() # 找出星核座標

                target = [probe_x[0], probe_y[0]] # 選擇第一隻探測機

                self.probe_selected = True
                # select_point的arg需要的形式像是[[0], [23, 46]]，[0]的意思是screen，後面的list代表座標
                return actions.FunctionCall(_SELECT_POINT, [_SCREEN, target])

            # 看建造水晶塔在這個observation中是否為合法的action
            elif _BUILD_PYLON in obs.observation["available_actions"]:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                nexus_y, nexus_x = (unit_type == _PROTOSS_NEXUS).nonzero() # 找出星核的位置
                #print('nexus location x:', nexus_x.mean())
                #print('nexus location y:', nexus_y.mean())

                # 找出星核上方或下方的位置並建造
                target = self.transformLocation(int(nexus_x.mean()), 0, int(nexus_y.mean()), 20)

                self.pylon_built = True
                return actions.FunctionCall(_BUILD_PYLON, [_SCREEN, target])

        # rule 2: 如果水晶建造了但星門(兵營)還沒建造，則建造星門
        elif not self.gateway_built:
            if _BUILD_GATEWAY in obs.observation["available_actions"]:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                pylon_y, pylon_x = (unit_type == _PROTOSS_PYLON).nonzero()
                print('pylon location x:', pylon_x)
                print('pylon location y:', pylon_y)
                #pylon location x: [32 33 34 35 31 32 33 34 35 36 30 31 32 33 34 35 36 37 30 31 32 33 34 35 36 37 30 31 32 33 34 35 36 37 30 31 32 33 34 35 36 37 31 32 33 34 35 36 32 33 34 35]
                #pylon location y: [ 5  5  5  5  6  6  6  6  6  6  7  7  7  7  7  7  7  7  8  8  8  8  8  8  8  8  9  9  9  9  9  9  9  9 10 10 10 10 10 10 10 10 11 11 11 11 11 11 12 12 12 12]


                target = self.transformLocation(int(pylon_x.mean()), 10, int(pylon_y.mean()), 0)

                #if (unit_type == _PROTOSS_PYLON).any():
                #    self.gateway_built = True

                # 確認星門有被建造，才停止這個rule
                if (unit_type == _PROTOSS_GATEWAY).any():
                    self.gateway_built = True

                return actions.FunctionCall(_BUILD_GATEWAY, [_SCREEN, target])

        # rule 3: 如果水晶、星門都建造了，則派兵駐守斜坡(斜坡座標用hardcode的形式寫下)
        elif not self.gateway_rallied:
            # 必須先選擇gateway，才能設置集合點
            if not self.gateway_selected:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                gateway_y, gateway_x = (unit_type == _PROTOSS_GATEWAY).nonzero()

                # 確認有選擇到星門
                if gateway_y.any():
                    target = [int(gateway_x.mean()), int(gateway_y.mean())]
                    self.gateway_selected = True
                    return actions.FunctionCall(_SELECT_POINT, [_SCREEN, target])
            else:
                self.gateway_rallied = True
                if self.nexus_top_left:
                    return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_MINIMAP, [29, 21]])
                else:
                    return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_MINIMAP, [29, 46]])

        # rule 4: 如果人口還沒達到上限，就一直訓練狂戰士
        elif obs.observation["player"][_SUPPLY_USED] < obs.observation["player"][_SUPPLY_MAX] and \
                _TRAIN_ZEALOT in obs.observation["available_actions"]:
            return actions.FunctionCall(_TRAIN_ZEALOT, [_QUEUED]) # [1]代表True，也就是選擇訓練狂戰士

        # rule 5: 如果人口滿了，則進攻對方基地
        elif not self.army_rallied: # 軍隊集結
            if not self.army_selected: # 圈選軍隊
                if _SELECT_ARMY in obs.observation["available_actions"]:
                    self.army_selected = True
                    # self.gateway_selected = False # 不確定這步是幹嘛的

                    return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
            elif _ATTACK_MINIMAP in obs.observation["available_actions"]:
                self.army_rallied = True
                # self.army_selected = False # 不確定這步是幹嘛的

                # 進攻與自己星核相對的位置(hardcode)
                if self.nexus_top_left:
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_MINIMAP, [39, 45]])
                else:
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_MINIMAP, [21, 24]])

        # 如果現在的observation不符合任一條規則，則什麼都不做
        return actions.FunctionCall(_NOOP, [])
