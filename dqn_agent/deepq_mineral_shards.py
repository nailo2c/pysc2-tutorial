# -*- coding: utf-8 -*-
import numpy as np
import os
import dill
import tempfile
import tensorflow as tf
import zipfile

import baselines.common.tf_util as U

from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

from pysc2.lib import actions as sc2_actions
from pysc2.env import environment
from pysc2.lib import features
from pysc2.lib import actions

from absl import flags

# 這段code的整體架構參考:
# https://github.com/openai/baselines/blob/master/baselines/deepq/simple.py

# Define constant and function id
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1

_NO_OP         = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN   = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY   = actions.FUNCTIONS.select_army.id

_NOT_QUEUED = [0]
_SELECT_ALL = [0]

FLAGS = flags.FLAGS



# 參考baselines裡的ActWrapper進行簡單改寫
class ActWrapper(object):
    def __init__(self, act):
        self._act = act

    @staticmethod
    def load(path, act_params, num_cpu=16):
        with open(path, "rb") as f:
            model_data = dill.load(f)
        act = deepq.build_act(**act_params)
        sess = U.make_session(num_cpu=num_cpu)
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            U.load_state(os.path.join(td, "model"))

        return ActWrapper(act)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def save(self, path):
        """Save model to a pickle located at `path`"""
        with tempfile.TemporaryDirectory() as td:
            U.save_state(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            dill.dump((model_data), f)



def load(path, act_params, num_cpu=16):
    return ActWrapper.load(path, act_params=act_params, num_cpu=num_cpu)



def learn(env,
          q_func,
          num_actions=4,
          lr=5e-4,
          max_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=1,
          checkpoint_freq=10000,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          num_cpu=16,
          param_noise=False,
          param_noise_threshold=0.05,
          callback=None):

    # Create all the functions necessary to train the model
    
    # Returns a session that will use <num_cpu> CPU's only
    sess = U.make_session(num_cpu=num_cpu)
    sess.__enter__()
    
    # Creates a placeholder for a batch of tensors of a given shape and dtyp
    def make_obs_ph(name):
        return U.BatchInput((64,64), name=name)
    
    # act, train, update_target are function, debug is dict
    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=num_actions,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10
    )
    
    # Choose use prioritized replay buffer or normal replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
        
    # Create the schedule for exploration starting from 1
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # SC2的部分開始

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()
    
    episode_rewards = [0.0]
    saved_mean_reward = None
    
    path_memory = np.zeros((64, 64))
    
    obs = env.reset()
    
    # Select all marines
    obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])])

    # obs is tuple, obs[0] is 'pysc2.env.environment.TimeStep', obs[0].observation is dictionary.
    player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
    
    # 利用path memory記憶曾經走過的軌跡
    screen = player_relative + path_memory
    
    # 取得兩個陸戰隊的中心位置
    player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
    player = [int(player_x.mean()), int(player_y.mean())]


    reset = True
    with tempfile.TemporaryDirectory() as td:
        model_saved = False
        model_file = os.path.join(td, "model")
        
        for t in range(max_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                if param_noise_threshold >= 0.:
                    update_param_noise_threshold = param_noise_threshold
                else:
                    update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(num_actions))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True
            # np.array()[None] 是指多包一個維度在外面 e.g. [1] -> [[1]]
            action = act(np.array(screen)[None], update_eps=update_eps, **kwargs)[0]
            reset = False
            
            coord = [player[0], player[1]]
            rew = 0

            # 只有四個action，分別是上下左右，走過之後在路徑上留下一整排-3，目的是與水晶碎片的id(=3)相抵銷，代表有順利採集到。
            path_memory_ = np.array(path_memory, copy=True)
            if (action == 0): # UP
                
                if (player[1] >= 16):
                    coord = [player[0], player[1] - 16]
                    path_memory_[player[1] - 16: player[1], player[0]] = -3
                elif (player[1] > 0):
                    coord = [player[0], 0]
                    path_memory_[0 : player[1], player[0]] = -3
                
            elif (action == 1): # DOWN
                
                if (player[1] <= 47):
                    coord = [player[0], player[1] + 16]
                    path_memory_[player[1] : player[1] + 16, player[0]] = -3
                elif (player[1] > 47):
                    coord = [player[0], 63]
                    path_memory_[player[1] : 63, player[0]] = -3
                    
            elif (action == 2): # LEFT
                
                if (player[0] >= 16):
                    coord = [player[0] - 16, player[1]]
                    path_memory_[player[1], player[0] - 16 : player[0]] = -3
                elif (player[0] < 16):
                    coord = [0, player[1]]
                    path_memory_[player[1], 0 : player[0]] = -3
                    
            elif (action == 3): # RIGHT
                
                if (player[0] <= 47):
                    coord = [player[0] + 16, player[1]]
                    path_memory_[player[1], player[0] : player[0] + 16] = -3
                elif (player[0] > 47):
                    coord = [63, player[1]]
                    path_memory_[player[1], player[0] : 63] = -3
            
            # 更新path_memory
            path_memory = np.array(path_memory_)

            # 如果不能移動陸戰隊，想必是還沒圈選到陸戰隊，圈選他們
            if _MOVE_SCREEN not in obs[0].observation["available_actions"]:
                obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])])
                
            # 移動陸戰隊
            new_action = [sc2_actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, coord])]
            
            # 取得環境給的observation
            obs = env.step(actions=new_action)
            
            # 這裡要重新取得player_relative，因為上一行的obs是個有複數資訊的tuple
            # 但我們要存入replay_buffer的只有降維後的screen畫面
            player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
            new_screen = player_relative + path_memory

            # 取得reward
            rew = obs[0].reward
            
            # StepType.LAST 代表done的意思
            done = obs[0].step_type == environment.StepType.LAST
            
            # Store transition in the replay buffer
            replay_buffer.add(screen, action, rew, new_screen, float(done))
            
            # 確實存入之後就能以新screen取代舊screen
            screen = new_screen
            
            episode_rewards[-1] += rew

            if done:
                # 重新取得敵我中立關係位置圖
                obs = env.reset()
                # player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
                
                # # 還是看不懂為何要加上path_memory
                # screen = player_relative + path_memory
                
                # player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
                # player = [int(player_x.mean()), int(player_y.mean())]
                    
                # # 圈選全部的陸戰隊(為何要在done observation做這件事情?)
                # env.step(actions=[sc2_actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])])
                episode_rewards.append(0.0)
                
                # 清空path_memory
                path_memory = np.zeros((64, 64))
                
                reset = True
                
            # 定期從replay buffer中抽experience來訓練，以及train target network
            if t > learning_starts and t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                # 這裡的train來自deepq.build_train
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)
            
            # target network        
            if t > learning_starts and t % target_network_update_freq == 0:
                # 同樣來自deepq.build_train
                # Update target network periodically
                update_target()
                
            # 下LOG追蹤reward
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode", mean_100ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()
                
            # 當model進步時，就存檔下來
            if (checkpoint_freq is not None and t > learning_starts and
                   num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                            saved_mean_reward, mean_100ep_reward))
                    U.save_state(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward
            if model_saved:
                if print_freq is not None:
                    logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
                U.load_state(model_file)
                
        return ActWrapper(act)