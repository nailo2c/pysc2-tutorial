# -*- coding: utf-8 -*-
import sys

import gflags as flags
from baselines import deepq

from pysc2.env import sc2_env
from pysc2.lib import actions

import deepq_mineral_shards

# Define the constant
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECE_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL  = [0]
_NOT_QUEUED  = [0]

step_mul = 8

FLAGS = flags.FLAGS

# main function, create env, define model, learn from observation and save model
def main():
    FLAGS(sys.argv)
    with sc2_env.SC2Env(map_name="CollectMineralShards", step_mul=step_mul) as env:
        # CNN
        model = deepq.models.cnn_to_mlp(
                convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                hiddens=[256],
                dueling=True
            )

        act = deepq_mineral_shards.learn(
                env,
                q_func=model,
                num_actions=4, # 輸出的動作只有up, down, left, right
                lr=1e-5,
                max_timesteps=2000000,
                buffer_size=100000,
                exploration_fraction=0.5,
                exploration_final_eps=0.01,
                train_freq=4,
                learning_starts=100000,
                target_network_update_freq=1000,
                gamma=0.99,
                prioritized_replay=True
            )
        act.save("mineral_shards.pkl")


if __name__ == '__main__':
    main()