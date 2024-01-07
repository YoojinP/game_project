from typing import NamedTuple

class Config(NamedTuple):
    # learning config
    seed: int = 1234  # random seed
    log_step: int = 100  # log frequency
    # hyper-parameters
    batch_size: int = 4  # batch size
    num_generator_epochs: int = 3  # 3, number of epochs for training ppo policy
    num_discriminator_epochs: int = 1  # 3, number of epochs for training discriminator
    num_steps: int = 80000 # 128  # horizon

    num_units: int = 64  # fc units
    # gamma: float = 0.99  # discount rate
    # lambda_: float = 0.95  # gae discount rate
    clip: float = 0.2  # clipping c
    vf_coef: float = 0.5  # coefficient of value loss
    ent_coef: float = 0.01  # coefficient of entropy
    learning_rate: float = 2.5e-4  # learning rate
    gradient_clip: float = 0.5  # gradinet clipping

    player_num: int = 6
    action_n: int = 13
    observation_n: int = 49
    load_model: bool = False
    load_ep: int = 1