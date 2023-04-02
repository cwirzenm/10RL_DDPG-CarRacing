import utils
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, Concatenate
from keras.initializers.initializers_v2 import RandomUniform
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


class ActionNoise:
    def __init__(self, steering, pedals, mean=np.zeros(2, dtype=np.float32), theta=0.3, dt=5e-2, x_initial=None):
        """Based on Ornstein-Uhlenbeck process"""
        self.noise = None
        self.y_prev = None
        self.theta = theta
        self.mean = mean
        self.std_dev = np.array([steering, pedals], dtype=np.float32)
        self.dt = dt
        self.x_initial = x_initial
        self.train = True
        self.reset()

    def __call__(self) -> np.ndarray:
        """Iterate according to the Ornstein-Uhlenbeck formula"""
        if self.train:
            self.noise = self.noise + self.theta * (self.mean - self.noise) * self.dt + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
            return self.noise
        else: return np.zeros(2)

    def reset(self):
        """Reset noise object to its initial state"""
        self.noise = self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)

    def disable(self): self.train = False


class Buffer:
    def __init__(self, state_space: tuple, action_space: tuple, buffer_capacity=50000, batch_size=64):
        """Object for storing transitions and updating the network weights"""
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

        self.buffer_counter = 0
        self.state_buffer = np.zeros((self.buffer_capacity, *state_space))
        self.action_buffer = np.zeros((self.buffer_capacity, *action_space))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, *state_space))

    def record(self, obs_tuple):
        """
        :param obs_tuple: Transition (S, A, R, S')
        """
        index = self.buffer_counter % self.buffer_capacity

        # saving the transition in a buffer
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter += 1

    def get_batches(self) -> tf.tuple:
        """Collect batches for gradient updates"""

        record_range = min(self.buffer_counter, self.buffer_capacity)  # sampling range
        batch_indices = np.random.choice(record_range, self.batch_size)  # sample indices randomly

        # get indices and convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices], dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        return tf.tuple([state_batch, action_batch, reward_batch, next_state_batch])


class DDPG:
    def __init__(self, env: gym.envs.registration, noise: ActionNoise, learning_rates: tuple, episodes: int, gamma: float, tau: float, saving_dir: str):
        self.env = env
        self.noise = noise
        self.episodes = episodes
        self.G = gamma
        self.T = tau
        self.saving_dir = saving_dir

        actor_lr, critic_lr = learning_rates
        self.actor_optimizer = Adam(actor_lr)
        self.critic_optimizer = Adam(critic_lr)

        self.state_space = env.observation_space.shape
        self.action_space = 2,  # we're combining acceleration and braking into one axis

        # initialise the buffer
        self.buffer = Buffer(self.state_space, self.action_space, buffer_capacity=50000, batch_size=64)

        # initialise the networks
        self.actor = self.actor_network()
        self.critic = self.critic_network()
        self.target_actor = self.actor_network()
        self.target_critic = self.critic_network()
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    def actor_network(self) -> Model:
        """Defining the actor network"""
        # initialize small weights due to sensitive tanh activation
        weights_init = RandomUniform(minval=-0.005, maxval=0.005)

        inputs = Input(shape=self.state_space, name='state_input')
        outputs = Conv2D(filters=16, kernel_size=5, strides=4, activation='relu', use_bias=False, kernel_initializer=weights_init, name='conv2d_1')(inputs)
        outputs = Conv2D(filters=32, kernel_size=3, strides=3, activation='relu', use_bias=False, kernel_initializer=weights_init, name='conv2d_2')(outputs)
        outputs = Conv2D(filters=32, kernel_size=3, strides=3, activation='relu', use_bias=False, kernel_initializer=weights_init, name='conv2d_3')(outputs)
        outputs = Flatten(name='flatten')(outputs)
        outputs = Dense(64, activation='relu', kernel_initializer=weights_init, name='dense_1')(outputs)
        # acceleration and braking are in one axis
        outputs = Dense(2, activation='tanh', kernel_initializer=weights_init, name='next_state_output')(outputs)
        model = Model(inputs, outputs, name='actor')
        return model

    def critic_network(self) -> Model:
        """Defining the critic network"""
        state_inputs = Input(shape=self.state_space, name='state_input')
        state_outputs = Conv2D(filters=16, kernel_size=5, strides=4, activation='relu', use_bias=False, name='state_conv2d_1')(state_inputs)
        state_outputs = Conv2D(filters=32, kernel_size=3, strides=3, activation='relu', use_bias=False, name='state_conv2d_2')(state_outputs)
        state_outputs = Conv2D(filters=32, kernel_size=3, strides=3, activation='relu', use_bias=False, name='state_conv2d_3')(state_outputs)
        state_outputs = Flatten(name='state_flatten')(state_outputs)

        action_inputs = Input(shape=2, name='action_input')

        concat = Concatenate(name='concat')([state_outputs, action_inputs])
        outputs = Dense(64, activation='relu', name='concat_dense_1')(concat)
        outputs = Dense(32, activation='relu', name='concat_dense_2')(outputs)
        outputs = Dense(1, name='action_value_output')(outputs)

        # Outputs single value for give state-action
        model = Model([state_inputs, action_inputs], outputs, name='critic')
        return model

    @staticmethod
    def preprocess(state: np.ndarray) -> np.ndarray:
        processed_state = np.array(state, copy=True, dtype=np.float32)

        # process the speed bar
        for i in range(88, 94): processed_state[i, 0:12, :] = processed_state[i, 12, :]

        # process grass colour
        # from [102, 229, 102] to [102, 204, 102]
        processed_state[:, :, 1][processed_state[:, :, 1] == 229] = 204

        # process track colour
        # from [102, 105, 102] and [102, 107, 102] to [102, 102, 102]
        processed_state[:, :, 1][(processed_state[:, :, 1] == 105) | (processed_state[:, :, 1] == 107)] = 102

        # scale
        processed_state /= np.max(processed_state)

        return processed_state

    def policy(self, state) -> tuple[np.ndarray, np.ndarray]:
        """Returns the next action in two formats"""
        # get the best action according to the actor network and add noise to it
        train_action = tf.squeeze(self.actor(state)).numpy() + self.noise()

        # clip the actions, convert to a format accepted by the environment and divide by 4 to improve steering
        env_action = np.array([train_action[0].clip(-1, 1), train_action[1].clip(0, 1), -train_action[1].clip(-1, 0)]) * 0.25

        return env_action, train_action

    @tf.function
    def update_gradients(self, batches: tf.tuple):
        """Training and updating the networks"""
        state_batch, action_batch, reward_batch, next_state_batch = batches

        # gradient descent for critic network
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.G * self.target_critic([next_state_batch, target_actions], training=True)
            critic_value = self.critic([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        # gradient ascent for actor network
        with tf.GradientTape() as tape:
            actions = self.actor(state_batch, training=True)
            critic_value = self.critic([state_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

    @tf.function
    def update_targets(self):
        """Updating target networks"""
        for (a, b) in zip(self.target_actor.variables, self.actor.variables): a.assign(b * self.T + a * (1 - self.T))
        for (a, b) in zip(self.target_critic.variables, self.critic.variables): a.assign(b * self.T + a * (1 - self.T))

    def save_weights(self):
        """Save the weights"""
        self.actor.save_weights(f"{self.saving_dir}/actor.h5")
        self.critic.save_weights(f"{self.saving_dir}/critic.h5")
        self.target_actor.save_weights(f"{self.saving_dir}/target_actor.h5")
        self.target_critic.save_weights(f"{self.saving_dir}/target_critic.h5")

    def main_loop(self, episodes, train=True, verbose=False):
        reward_list = []
        avg_reward_list = []
        for e in range(episodes):
            utils.blockPrint()
            prev_state = self.preprocess(self.env.reset())
            utils.enablePrint()
            episodic_reward = 0
            break_counter = 0
            with utils.Timer() as time:
                while True:
                    if verbose: self.env.render()
                    tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                    action, train_action = self.policy(tf_prev_state)
                    state, reward, done, info = self.env.step(action)
                    state = self.preprocess(state)  # simplify the state image
                    episodic_reward += reward
                    if train:
                        self.buffer.record((prev_state, train_action, reward, state))
                        self.update_gradients(self.buffer.get_batches())
                        self.update_targets()

                    if reward < 0: break_counter += 1
                    else: break_counter = 0
                    # terminate episode if agent earns no reward after 50 steps
                    if done or break_counter == 50: break
                    prev_state = state

            reward_list.append(episodic_reward)
            avg_reward = np.mean(reward_list[-30:])
            avg_reward_list.append(avg_reward)
            print(f"Episode {e + 1}; reward: {episodic_reward:.3f}; average reward: {avg_reward:.3f}; time: {time.getAbsoluteInterval():.3f}s")

        if train: self.save_weights()
        plt.plot(avg_reward_list)
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.show()

    def train(self, verbose=False):
        """Main function for training the agents"""
        if verbose:
            self.actor.summary()
            self.critic.summary()

        self.main_loop(self.episodes, train=True, verbose=verbose)

    def evaluate(self, episodes, models_dir=None):
        """Evaluate function for presentation"""
        if models_dir: self.actor.load_weights(f"{models_dir}/actor.h5")

        self.noise.disable()  # disabling noise for evaluation
        self.main_loop(episodes=episodes, train=False, verbose=True)


if __name__ == '__main__':
    tf.compat.v1.ConfigProto().gpu_options.allow_growth = True
    env = gym.make("CarRacing-v0")

    # set parameters
    ACTOR_LR = 0.00001
    CRITIC_LR = 0.02
    EPISODES = 200
    GAMMA = 0.99  # discount factor for past rewards
    TAU = 0.005  # discount factor for future rewards
    STEERING_NOISE = 0.1  # small noise for steering
    PEDALS_NOISE = 0.8  # big noise for acceleration and braking
    TESTID = 'test2'

    ou_noise = ActionNoise(steering=STEERING_NOISE, pedals=PEDALS_NOISE)
    ddpg = DDPG(env, noise=ou_noise, learning_rates=(ACTOR_LR, CRITIC_LR), episodes=EPISODES, gamma=GAMMA, tau=TAU, saving_dir=TESTID)
    ddpg.train(verbose=True)
    ddpg.evaluate(episodes=10, models_dir='test2')

    # todo test logs of speed and braking
