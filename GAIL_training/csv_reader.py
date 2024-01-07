import os
import numpy as np
import tensorflow as tf
import pandas as pd
import re
import time
import pyarrow as pa
from pyarrow import csv
from tqdm import tqdm
class ExpertDataset:
    def __init__(self, obs_shape, agent_shape, num_steps):
        self.player_num = 6
        self.size = num_steps
        self.obses = np.zeros((self.player_num, )+(self.size, )+obs_shape)
        self.actions = np.zeros((self.player_num, )+(self.size, ), dtype=int)
        self.terminal = np.zeros((self.size, ))
        self.sample_idx = 0
        self.idxes = [0 for i in range(self.player_num)]

    def __len__(self):
        return self.idxes[0]

    def move_sample_idx(self, sample_num, epoch):
        self.sample_idx += sample_num * (epoch-1)
        return

    def add(self, pl_id, obs, action, done): # 플레이어당 한개씩
        self.obses[pl_id, self.idxes[pl_id]] = obs
        self.actions[pl_id, self.idxes[pl_id]] = action
        self.terminal[self.idxes[pl_id]] = done
        self.idxes[pl_id] += 1

    def sample(self, pl_id, idexes):  # sample batch_size
        mini_batch_act = []
        mini_batch_obs = []
        for i in idexes:  # range(batch_size):
            idx = (self.sample_idx + i) % self.size
            mini_batch_obs.append(self.obses[pl_id, idx])  # self.idxes[pl_id]
            mini_batch_act.append(self.actions[pl_id, idx])  # self.idxes[pl_id]
        mini_batch_obs = tf.convert_to_tensor(mini_batch_obs, dtype=tf.float32)
        mini_batch_act = tf.convert_to_tensor(mini_batch_act, dtype=tf.int32)
        return mini_batch_obs, mini_batch_act


    def reset(self):
        self.idxes = [0 for i in range(self.player_num)]
        self.obses = np.zeros_like(self.obses)
        self.actions = np.zeros_like(self.actions)
        self.terminal = np.zeros_like(self.terminal)

def csv_read(file_path):
    csv_set= []
    max_len = 0
    for file in os.listdir(file_path):
        file_name = file.split('.')[0]
        reader = csv.read_csv(f'{file_path}/{file_name}.csv').to_pandas()
        max_len += len(reader)
        csv_set.append(reader)
    return csv_set, max_len

def csv_to_dataset(csv_set, dataset):
    for reader in csv_set:
        for idx, row in tqdm(reader.iterrows()):  # row idx
            if row['gamestate'] != 1:
                continue
            obss = [row[key] for key in ['whoattack', 'ballclear', 'ball_x', 'ball_y', 'ball_z', 'home1_pos', 'home1_x','home1_y',
                                         'home1_z', 'home1_mainstate', 'home1_getball', 'home1_action', 'home2_pos', 'home2_x', 'home2_y', 'home2_z',
                                         'home2_mainstate', 'home2_getball', 'home2_action', 'home3_pos', 'home3_x', 'home3_y', 'home3_z', 'home3_mainstate',
                                         'home3_getball', 'home3_action', 'away1_pos', 'away1_x', 'away1_y', 'away1_z', 'away1_mainstate', 'away1_getball',
                                         'away1_action', 'away2_pos', 'away2_x', 'away2_y', 'away2_z', 'away2_mainstate', 'away2_getball', 'away2_action',
                                         'away3_pos', 'away3_x', 'away3_y', 'away3_z', 'away3_mainstate', 'away3_getball', 'away3_action']]
            acts = [int(row[key]) for key in ['home1_action', 'home2_action', 'home3_action', 'away1_action', 'away2_action', 'away3_action']]
            for pl in range(dataset.player_num):
                myID = pl % 3
                myTeam = pl // 3
                obs = [myID, myTeam] + obss
                obs = np.array(obs)
                act = acts[pl]
                act = np.array(act)
                done = True if reader.loc[idx+1, 'gamestate'] == 11 or 12 else False  # 1= Done, 0 = Not Done
                dataset.add(pl, obs, act, done)

    return dataset

def load_exp(path):
    obs_len = 49  # 47 + myID, myTeam
    agent_len = 1
    csv_set, max_len = csv_read(path)
    expd = ExpertDataset((obs_len, ), (agent_len, ), max_len)
    expd = csv_to_dataset(csv_set, expd)
    return expd


if __name__ == "__main__":
    expd = load_exp('./data/expert')
    ob, ac = expd.sample(1 , [1,2,3,4,5,10,29,294,182,23,521])
    print()

    # def csv_to_dataset(csv_set, dataset):
    #     for reader in csv_set:
    #         for idx in tqdm(reader.index):  # row idx
    #             if reader.loc[idx, 'gamestate'] != 1:
    #                 continue
    #             obss = [reader.loc[idx,key] for key in ['whoattack', 'ballclear', 'ball_x', 'ball_y', 'ball_z', 'home1_pos', 'home1_x','home1_y',
    #                                                      'home1_z', 'home1_mainstate', 'home1_getball', 'home1_action', 'home2_pos', 'home2_x', 'home2_y', 'home2_z',
    #                                                      'home2_mainstate', 'home2_getball', 'home2_action', 'home3_pos', 'home3_x', 'home3_y', 'home3_z', 'home3_mainstate',
    #                                                      'home3_getball', 'home3_action', 'away1_pos', 'away1_x', 'away1_y', 'away1_z', 'away1_mainstate', 'away1_getball',
    #                                                      'away1_action', 'away2_pos', 'away2_x', 'away2_y', 'away2_z', 'away2_mainstate', 'away2_getball', 'away2_action',
    #                                                      'away3_pos', 'away3_x', 'away3_y', 'away3_z', 'away3_mainstate', 'away3_getball', 'away3_action']]
    #             acts = [int(reader.loc[idx, key]) for key in ['home1_action', 'home2_action', 'home3_action', 'away1_action', 'away2_action', 'away3_action']]
    #             for pl in range(dataset.player_num):
    #                 myID = pl % 3
    #                 myTeam = pl // 3
    #                 obs = [myID, myTeam] + obss
    #                 obs = np.array(obs)
    #                 act = acts[pl]
    #                 act = np.array(act)
    #                 done = True if reader.loc[idx+1, 'gamestate'] == 11 or 12 else False  # 1= Done, 0 = Not Done
    #                 dataset.add(pl, obs, act, done)
    #     return dataset
