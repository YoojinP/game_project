import numpy as np
import tensorflow as tf
import os
from pyarrow import csv
import re
from config import Config

class Memory:
    def __init__(self, obs_shape, hparams, model, player_num):
        self.player_num = player_num
        self._hparams = hparams
        self.size = self._hparams.num_steps
        self.obses   = np.zeros((self.player_num, )+(self.size, )+obs_shape)
        self.actions = np.zeros((self.player_num, )+(self.size, ))
        self.rewards = np.zeros((self.player_num, )+(self.size,   1))
        self.dones   = np.zeros((self.player_num, )+(self.size,   1))
        self.values  = np.zeros((self.player_num, )+(self.size+1, 1))
        self.policy  = np.zeros((self.player_num, )+(self.size,   1))
        self.deltas  = np.zeros((self.player_num,)+(self.size,   1))
        self.discounted_rew_sum = np.zeros((self.player_num,)+(self.size, 1))
        self.gae = np.zeros((self.player_num,)+(self.size+1, 1))

        self.sample_i = 0
        self.model = model

    def __len__(self):
        return self.sample_i

    def add(self, pl_id, obs, action, reward, done, value, policy):
        idx = self.sample_i
        if idx < self.size:
            self.obses[pl_id, idx] = obs
            self.actions[pl_id, idx] = action
            self.rewards[pl_id, idx] = reward
            self.dones[pl_id, idx] = done
            self.values[pl_id, idx] = value
            self.policy[pl_id, idx] = policy
        else:
            self.values[pl_id, idx] = value

        if pl_id == (self.player_num-1):
            self.sample_i += 1

    def compute_gae(self):
        for pl in range(self.player_num):
            for i in reversed(range(self.size+1)):
                self.deltas[pl, i - 1] = self.rewards[pl, i - 1] + self._hparams.gamma * self.values[pl, i] * (1 - self.dones[pl, i - 1]) - self.values[pl, i - 1]
            self.gae[pl, -1] = self.deltas[pl, -1]
            for t in reversed(range(self.size-1)):
                self.gae[pl, t] = self.deltas[pl, t] + (1 - self.dones[pl, t]) * (self._hparams.gamma * self._hparams.lambda_) * self.gae[pl, t + 1]
            self.discounted_rew_sum[pl] = self.gae[pl, :-1] + self.values[pl, :-1]
            self.gae = (self.gae - np.mean(self.gae[pl, :-1])) / (np.std(self.gae[pl, :-1]) + 1e-8)
        return

    def sample(self, pl_id, idxes):
        batch_obs = tf.convert_to_tensor(self.obses[pl_id, idxes], dtype=tf.float32)
        batch_act = tf.convert_to_tensor(self.actions[pl_id, idxes], dtype=tf.int32)
        batch_adv = tf.squeeze(tf.convert_to_tensor(self.gae[pl_id, idxes], dtype=tf.float32))
        batch_pi = tf.squeeze(tf.convert_to_tensor(self.policy[pl_id, idxes], dtype=tf.float32))
        batch_sum = tf.squeeze(tf.convert_to_tensor(self.discounted_rew_sum[pl_id, idxes], dtype=tf.float32))
        return batch_obs, batch_act, batch_adv, batch_sum, batch_pi

    def reset(self):
        self.sample_i = 0 # [0 for i in range(self.player_num)]
        self.obses = np.zeros_like(self.obses)
        self.actions = np.zeros_like(self.actions)
        self.rewards = np.zeros_like(self.rewards)
        self.values = np.zeros_like(self.values)
        self.policy = np.zeros_like(self.policy)
        self.deltas = np.zeros_like(self.deltas)
        self.discounted_rew_sum = np.zeros_like(self.discounted_rew_sum)
        self.gae = np.zeros_like(self.gae)


class Env:
    def __init__(self, csvrs, config):
        self.csvr_idx = 0
        self.sample_idx = 0
        self.csvrs = csvrs
        self.reader = self.csvrs[self.csvr_idx]
        self.config = config
        self.done_idxes= []
        self.player_num = config.player_num
        print(len(self))

        # self.get_done()

    def __len__(self):
        _size = 0
        for cs in self.csvrs:
            _size += len(cs.loc[cs['gamestate'] == 1])
        return _size

    def get_done(self):
        text = ''.join(map(str, list(self.reader['gamestate'])))
        # [text.start() for text in re.finditer('01', text)] +
        self.done_idxes = [text.start() for text in re.finditer('10',text)] + [text.start() for text in re.finditer('01',text)] + [len(text)-1]  # gamestate 로 판단하는 done 정보

    def preprocess_row(self):
        while self.reader.loc[self.sample_idx, 'gamestate'] != 1:
            self.sample_idx += 1
            if self.sample_idx >= len(self) - 1:
                self.csvr_idx += 1
                self.reader = self.csvrs[self.csvr_idx]
                self.sample_idx = 0
            # return self.preprocess_row()  # TODO 재귀가 가능한가??

        obses = np.zeros((self.player_num,) + (self.config.observation_n,))
        action_set = np.zeros((self.player_num,))
        rewards = np.zeros((self.player_num, 1))
        dones = np.zeros((self.player_num, 1))

        obss = [self.reader.loc[self.sample_idx, key] for key in
                ['whoattack', 'ballclear', 'ball_x', 'ball_y', 'ball_z', 'home1_pos', 'home1_x','home1_y',
                 'home1_z', 'home1_mainstate', 'home1_getball', 'home1_action', 'home2_pos', 'home2_x', 'home2_y', 'home2_z',
                 'home2_mainstate', 'home2_getball', 'home2_action', 'home3_pos', 'home3_x', 'home3_y', 'home3_z', 'home3_mainstate',
                 'home3_getball', 'home3_action', 'away1_pos', 'away1_x', 'away1_y', 'away1_z', 'away1_mainstate', 'away1_getball',
                 'away1_action', 'away2_pos', 'away2_x', 'away2_y', 'away2_z', 'away2_mainstate', 'away2_getball', 'away2_action',
                 'away3_pos', 'away3_x', 'away3_y', 'away3_z', 'away3_mainstate', 'away3_getball', 'away3_action']]
        actions = [int(self.reader.loc[self.sample_idx, key]) for key in ['home1_action', 'home2_action', 'home3_action', 'away1_action', 'away2_action', 'away3_action']]
        done = True if self.reader.loc[self.sample_idx+1, 'gamestate'] == 11 or 12 else False  # 0 = not done, 1 = done

        for pl in range(self.player_num):
            myID = pl % 3
            myTeam = pl // 3
            obs = [myID, myTeam] + obss
            obs = np.array(obs)
            action = actions[pl]
            action = np.array(action)

            if self.reader.loc[self.sample_idx, 'whoattack'] == 0 and self.reader.loc[self.sample_idx, 'gamestate'] == 5:  # 득점, home팀
                reward = 1 if myTeam == 0 else -1
            elif self.reader.loc[self.sample_idx, 'whoattack'] == 1 and self.reader.loc[self.sample_idx, 'gamestate'] == 5:  # 득점, 상대팀
                reward = -1 if myTeam == 0 else 1
            else:
                reward = 0

            obses[pl] = obs
            action_set[pl] = action
            rewards[pl] = reward
            dones[pl] = done

        self.sample_idx += 1
        return obses, action_set, rewards, dones

    def step(self):
        obses, actions, rewards, dones = self.preprocess_row()
        obs = tf.convert_to_tensor(obses, dtype=tf.float32)  # [pl_id]
        act = tf.convert_to_tensor(actions, dtype=tf.int32)
        terminal = tf.squeeze(tf.convert_to_tensor(dones, dtype=tf.float32))
        reward = tf.squeeze(tf.convert_to_tensor(rewards, dtype=tf.int32))
        return obs, act, terminal, reward


def load_data(data_path):
    readers = []
    for file in os.listdir(data_path):
        file_name = file.split('.')[0]
        print(file_name)
        reader = csv.read_csv(f'{data_path}/{file_name}.csv').to_pandas()
        readers.append(reader)
    return readers


if __name__ == "__main__":
    data_path = './data/agent'
    readers = load_data(data_path)
    config = Config()
    env = Env(readers, config)