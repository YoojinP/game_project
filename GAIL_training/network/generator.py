import tensorflow as tf
EPS = 1e-8
class Generator(tf.Module):
    def __init__(self, num_actions, input_shape, config):
        super(Generator, self).__init__(name='ppo_model')
        self.num_actions = num_actions
        self.input_shape = input_shape
        self.batch_size = config.batch_size
        self.config = config
        self.policy_value_network = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        self.gradient_clip = config.gradient_clip

    @tf.function
    def step(self, obs):
        obs = tf.expand_dims(obs, 0)
        pi, v = self.policy_value_network(obs)
        pi = tf.squeeze(pi)
        return pi, v

    @tf.function
    def train(self, obs, action, pi_old, advantage, rew_sum):  # batch_obs, batch_act, batch_pi_old, batch_adv, batch_sum
        with tf.GradientTape() as tape:
            policy, value = self.policy_value_network(obs)
            value = tf.squeeze(value)
            one_hot_action = tf.one_hot(action, self.num_actions)
            pi = tf.squeeze(tf.reduce_sum(policy * one_hot_action, axis=1, keepdims=True))
            ratio = tf.divide(pi, pi_old + EPS)
            clipped_advantage = tf.where(advantage > 0, (1+self.config.clip)*advantage, (1-self.config.clip)*advantage)
            l_clip = (-1) * tf.reduce_mean(tf.minimum(ratio * advantage, clipped_advantage))
            l_vf = tf.reduce_mean(tf.square(rew_sum-value))
            entropy = - tf.reduce_sum(policy * tf.math.log(policy + EPS), axis=1)
            l_ent = tf.reduce_mean(entropy)
            loss = l_clip + l_vf * self.config.vf_coef - l_ent * self.config.ent_coef
        grads = tape.gradient(loss, self.policy_value_network.trainable_variables)
        if self.gradient_clip is not None:
            clipped_grads = []
            for grad in grads:
                clipped_grads.append(tf.clip_by_norm(grad, self.gradient_clip))
            grads = clipped_grads
        grads_and_vars = zip(grads, self.policy_value_network.trainable_variables)
        self.optimizer.apply_gradients(grads_and_vars)
        new_policy, _ = self.policy_value_network(tf.convert_to_tensor(obs, dtype=tf.float32))
        new_prob = tf.reduce_sum(new_policy * one_hot_action, axis=1)
        kl = tf.reduce_mean(pi * tf.math.log(new_prob+EPS) - pi * tf.math.log(pi+EPS))
        clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0), self.config.clip), tf.float32))

        return loss, l_clip, l_vf, l_ent, policy, kl, clipfrac

    def _build_model(self):
        input_x = tf.keras.Input(shape=self.input_shape)
        h1 = tf.keras.layers.Dense(units=self.config.num_units, activation="relu")(input_x)
        h2 = tf.keras.layers.Dense(units=self.config.num_units, activation="relu")(h1)
        policy = tf.keras.layers.Dense(self.num_actions, activation="softmax")(h2)
        value = tf.keras.layers.Dense(1)(h2)
        model = tf.keras.Model(inputs=input_x, outputs=[policy, value])
        return model
