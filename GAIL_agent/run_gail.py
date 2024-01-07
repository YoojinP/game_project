from GAIL_training.network.generator import Generator
from GAIL_training.config import Config
import tensorflow as tf

class GAIL_agent:
    def __init__(self):  # 모델 다 읽어오기
        self.config = Config()
        self.model = Generator(
            num_actions=self.config.action_n,
            input_shape= (self.config.observation_n,),
            config=self.config
        )
        self.model.load_weights(f'./weights/generator.h5')

    # obss = ['whoattack', 'ballclear', 'ball_x', 'ball_y', 'ball_z', 'home1_pos', 'home1_x','home1_y',
    #          'home1_z', 'home1_mainstate', 'home1_getball', 'home1_action', 'home2_pos', 'home2_x', 'home2_y', 'home2_z',
    #          'home2_mainstate', 'home2_getball', 'home2_action', 'home3_pos', 'home3_x', 'home3_y', 'home3_z', 'home3_mainstate',
    #          'home3_getball', 'home3_action', 'away1_pos', 'away1_x', 'away1_y', 'away1_z', 'away1_mainstate', 'away1_getball',
    #          'away1_action', 'away2_pos', 'away2_x', 'away2_y', 'away2_z', 'away2_mainstate', 'away2_getball', 'away2_action',
    #          'away3_pos', 'away3_x', 'away3_y', 'away3_z', 'away3_mainstate', 'away3_getball', 'away3_action']]
    def predict(self, obs):  # obs 를 주면 action을 반환
        action_prob, _ = self.model.step(obs)
        action = tf.math.argmax(action_prob)
        return action