import random
import sys
import os
import time

from tqdm import tqdm
import numpy as np
import tensorflow as tf

from network.generator import Generator
from network.discriminator import Discriminator

import matplotlib.pyplot as plt
from memory import Memory, Env, load_data
from csv_reader import load_exp
from config import Config
from util import set_global_seeds

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():  # args
    config = Config()
    action_n = config.action_n
    observation_space = (config.observation_n, )
    set_global_seeds(config.seed)
    epoch = 1  # int(args[1])
    
    # with tf.device("/gpu:0"):  # gpu를 사용하는 경우
    with tf.device("/cpu:0"):
        generator = Generator(
            num_actions=action_n,
            input_shape= observation_space,
            config=config
        )
        discriminator = Discriminator(
            num_obs=observation_space[0],
            num_actions=action_n,
            config=config
        )

    if config.load_model:
        generator.policy_value_network.load_weights(f'./weights/ppo_ep{config.load_ep}.h5')
        discriminator.network.load_weights(f'./weights/discriminator_ep{config.load_ep}.h5')


    # ----- load expert trajectories -----
    expert_path = "./data/expert"
    expert_dataset = load_exp(expert_path)

    # ----- load agent trajectories -----
    data_path = './data/agent'
    readers = load_data(data_path)
    env = Env(readers, config)

    loss_graph = []

    agent_memory = Memory(observation_space, config, generator, config.player_num)

    total_len = len(env) if len(env) <= len(expert_dataset) else len(expert_dataset)
    sample_num = total_len if config.num_steps > total_len else config.num_steps
    print(f"total_len : {total_len}, sample_num: {sample_num}")

    # ===== add sample =====
    for _ in tqdm(range(sample_num)): # 시간 엄청 소요됨 --> 분석 필요
        obs, action, done, reward = env.step()
        for pl in range(config.player_num):
            if done[pl]:
                _, last_value = generator.step(obs[pl])
                act_prob = None
                value = last_value
            else:
                probs, value = generator.step(obs[pl])
                act_prob = probs[action[pl]-1]
            agent_memory.add(pl, obs[pl], action[pl], reward[pl], done[pl], value, act_prob)

    start_time = time.time()
    expert_dataset.move_sample_idx(config.num_steps, epoch)

    # ===== train reward giver(discriminator) =====
    for _ in range(config.num_discriminator_epochs):
        idxes = [idx for idx in range(sample_num)]  # config.num_steps
        random.shuffle(idxes)
        for start in range(0, total_len, config.batch_size):  # len(agent_memory)
            minibatch_indexes = idxes[start:start + config.batch_size]
            for pl in range(config.player_num):
                agent_obs, agent_act, _, _, _ = agent_memory.sample(pl, minibatch_indexes)
                demo_obs, demo_act = expert_dataset.sample(pl, minibatch_indexes)  # config.batch_size
                total_loss, agent_loss, demo_loss, agent_acc, demo_acc = discriminator.train(demo_obs, demo_act,
                                                                                             agent_obs, agent_act)
        disc_rew = 0
        for pl in range(config.player_num):
            actions = tf.constant(agent_memory.actions[pl], dtype=tf.int32)
            observations = tf.constant(agent_memory.obses[pl], dtype=tf.float32)
            reward_signals = discriminator.inference(observations, actions).numpy()
            agent_memory.rewards[pl] = reward_signals
            rew_mean = np.mean(reward_signals)
            disc_rew += rew_mean

    print("discriminator train time", time.time() -start_time)
    start_time =time.time()

    # ===== train agent(generator) =====
    agent_memory.compute_gae()
    losses = []
    for _ in range(config.num_generator_epochs):
        idxes = [idx for idx in range(sample_num)]  # config.num_steps
        random.shuffle(idxes)
        T = 0
        for start in range(0, len(agent_memory), config.batch_size):
            minibatch_indexes = idxes[start:start+config.batch_size]
            semi_loss = 0
            for pl in range(config.player_num):
                batch_obs, batch_act, batch_adv, batch_sum, batch_pi_old = agent_memory.sample(pl, minibatch_indexes)
                loss, policy_loss, value_loss, entropy_loss, policy, kl, frac = generator.train(batch_obs, batch_act, batch_pi_old, batch_adv, batch_sum)
                semi_loss += loss
            T += (semi_loss/config.player_num)
        losses.append(T)
    loss_graph.append(sum(losses)/ len(losses))

    print("generator train time", time.time() - start_time)

    agent_memory.reset()

    # ===== save model per epoch  =====
    generator.policy_value_network.save(f'./weights/ppo_ep{epoch}.h5')
    discriminator.network.save(f'./weights/discriminator_ep{epoch}.h5')

if __name__ == "__main__":
    main()  # sys.argv, epoch 번호 받기 (1부터)

