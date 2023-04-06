import os
from shutil import rmtree
import utils
import gym
import numpy as np
from csv import writer
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, Concatenate, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.engine.keras_tensor import KerasTensor
from keras.callbacks import TensorBoard


class ActionNoise:
    def __init__(self, sigma: np.ndarray, mean=np.zeros(2, dtype=np.float32), theta=0.3, dt=5e-2, x_initial=None):
        """Based on Ornstein-Uhlenbeck process"""
        self.std_dev = sigma
        self.mean = mean
        self.theta = theta
        self.dt = dt
        self.x_initial = x_initial
        self.noise = self.x_initial if self.x_initial else np.zeros(2, dtype=np.float32)
        self.train = True

    def __call__(self) -> np.ndarray:
        """Iterate according to the Ornstein-Uhlenbeck formula"""
        if self.train:
            self.noise = self.noise + self.theta * (self.mean - self.noise) * self.dt + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
            return self.noise
        else: return np.zeros(2)

    def reset(self, sigma: np.ndarray, mean: np.ndarray):
        """Reset noise object to its initial state and reassign standard deviation"""
        self.noise = self.x_initial if self.x_initial else np.zeros_like(self.mean)
        self.std_dev = sigma
        self.mean = mean


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
    def __init__(self, env: gym.envs.registration, noise: ActionNoise, learning_rates: tuple,
                 episodes: int, gamma: float, tau: float, phi: float, saving_dir: str):
        self.env = env
        self.noise = noise
        self.episodes = episodes
        self.G = gamma
        self.T = tau
        self.PHI = phi

        if not os.path.exists(saving_dir): os.mkdir(saving_dir)
        self.saving_dir = saving_dir

        actor_lr, critic_lr = learning_rates
        self.actor_optimizer = Adam(actor_lr)
        self.critic_optimizer = Adam(critic_lr)

        self.state_space = env.observation_space.shape
        self.action_space = 2,  # we're combining acceleration and braking into one axis

        # initialise the buffer
        self.buffer = Buffer(self.state_space, self.action_space, buffer_capacity=60000, batch_size=64)

        # initialise the networks
        self.actor = self.actor_network()
        self.critic = self.critic_network()
        self.target_actor = self.actor_network()
        self.target_critic = self.critic_network()
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    @staticmethod
    def Conv2DWithBatchNorm(filters, kernel_size, strides, name, parent: KerasTensor) -> KerasTensor:
        """Conv2D -> BatchNorm -> ReLu"""
        return Activation('relu')(BatchNormalization()(
                Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, activity_regularizer='l2', use_bias=False, name=name)(parent)))

    @staticmethod
    def DenseWithBatchNorm(size, name, parent: KerasTensor) -> KerasTensor:
        """Dense -> BatchNorm -> ReLu"""
        return Activation('relu')(BatchNormalization()(Dense(size, activity_regularizer='l2', use_bias=False, name=name)(parent)))

    @staticmethod
    def OutputWithBatchNorm(parent: KerasTensor) -> KerasTensor:
        """Dense -> BatchNorm -> Tanh"""
        # acceleration and braking are in one axis
        return Activation('tanh')(BatchNormalization()(
                Dense(2, kernel_initializer='glorot_normal', activity_regularizer='l2', use_bias=False, name='output')(parent)))

    def actor_network(self, D_RATE=0.15) -> Model:
        """Defining the actor network"""
        inputs = Input(shape=self.state_space, name='state_input')
        outputs = self.Conv2DWithBatchNorm(filters=32, kernel_size=8, strides=4, name='conv2d_1', parent=inputs)
        outputs = Dropout(D_RATE)(outputs)
        outputs = self.Conv2DWithBatchNorm(filters=64, kernel_size=4, strides=3, name='conv2d_2', parent=outputs)
        outputs = Dropout(D_RATE)(outputs)
        outputs = self.Conv2DWithBatchNorm(filters=64, kernel_size=3, strides=1, name='conv2d_3', parent=outputs)
        outputs = Dropout(D_RATE)(outputs)
        outputs = Flatten(name='flatten')(outputs)
        outputs = self.DenseWithBatchNorm(size=512, name='dense_1', parent=outputs)
        outputs = self.OutputWithBatchNorm(outputs)
        model = Model(inputs, outputs, name='actor')
        return model

    def critic_network(self, D_RATE=0.15) -> Model:
        """Defining the critic network"""
        # state branch
        state_inputs = Input(shape=self.state_space, name='state_input')
        state_outputs = self.Conv2DWithBatchNorm(filters=32, kernel_size=8, strides=4, name='state_conv2d_1', parent=state_inputs)
        state_outputs = Dropout(D_RATE)(state_outputs)
        state_outputs = self.Conv2DWithBatchNorm(filters=64, kernel_size=4, strides=3, name='state_conv2d_2', parent=state_outputs)
        state_outputs = Dropout(D_RATE)(state_outputs)
        state_outputs = self.Conv2DWithBatchNorm(filters=64, kernel_size=3, strides=1, name='state_conv2d_3', parent=state_outputs)
        state_outputs = Dropout(D_RATE)(state_outputs)
        state_outputs = Flatten(name='state_flatten')(state_outputs)

        # action branch
        action_inputs = Input(shape=2, name='action_input')
        action_outputs = self.DenseWithBatchNorm(size=400, name='action_dense_1', parent=action_inputs)
        action_outputs = Dropout(D_RATE)(action_outputs)

        # joint branch
        concat = Concatenate(name='concat')([state_outputs, action_outputs])
        outputs = self.DenseWithBatchNorm(size=512, name='concat_dense_1', parent=concat)
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
        env_action = np.array([train_action[0].clip(-1, 1), train_action[1].clip(0, 1), -train_action[1].clip(-1, 0)]) * self.PHI

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

    def save_weights(self, agentId=None):
        """Save the weights"""
        if agentId is not None: path = f"{self.saving_dir}/{agentId}"; os.mkdir(path)
        else: path = self.saving_dir
        self.actor.save_weights(f"{path}/actor.h5")
        self.critic.save_weights(f"{path}/critic.h5")
        self.target_actor.save_weights(f"{path}/target_actor.h5")
        self.target_critic.save_weights(f"{path}/target_critic.h5")

    def delete_weights(self, agentId): rmtree(path=f"{self.saving_dir}/{agentId}")

    def load_weights(self, agentId=None):
        """Load the weights"""
        path = f"{self.saving_dir}/{agentId}" if agentId else self.saving_dir
        self.actor.load_weights(f"{path}/actor.h5")
        # self.critic.load_weights(f"{path}/critic.h5")
        # self.target_actor.load_weights(f"{path}/target_actor.h5")
        # self.target_critic.load_weights(f"{path}/target_critic.h5")

    def save_metrics(self, init=False, **kwargs):
        with open(f"{self.saving_dir}/metrics.csv", 'a', newline='') as file:
            if init: writer(file).writerow(kwargs.keys())
            writer(file).writerow(kwargs.values())

    def main_loop(self, episodes, train=True, verbose=True):
        avg_reward = 0
        top_10_agents = []
        rewards = []
        for agentId in range(episodes):
            # env.reset is annoying because it prints to console, so I'm disabling print around it using system settings
            with utils.BlockPrint(): prev_state = self.preprocess(self.env.reset())
            episodic_reward = 0
            break_counter = 0
            step = 0
            while True:
                step += 1
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
                # terminate episode if agent earns no reward after 150 steps
                if done or break_counter == 150: break
                prev_state = state

            rewards.append(episodic_reward)
            avg_reward = np.mean(rewards[-50:])
            print(f"Agent {agentId}; reward: {episodic_reward:.3f}; average reward: {avg_reward:.3f}; time steps: {step}")
            if train:
                self.save_metrics(init=agentId == 0, mov_avg_rewards_50=avg_reward, rewards=episodic_reward, timesteps=step)

                # save top 10 agents for evaluation
                # agent must have minimum 300 points to qualify for evaluation
                if episodic_reward > 300:
                    top_agents_rewards = np.array([rewards[i] for i in top_10_agents])
                    if len(top_10_agents) < 10:
                        top_10_agents.append(agentId)
                        self.save_weights(agentId)
                    elif episodic_reward > np.min(top_agents_rewards):
                        dropout_agent = top_10_agents[np.argmin(top_agents_rewards)]
                        self.delete_weights(dropout_agent)  # delete dropout agents weights
                        top_10_agents[top_10_agents.index(dropout_agent)] = agentId
                        self.save_weights(agentId)  # save new weights
        if not train: return avg_reward

    def train(self, verbose=True):
        """Main function for training the agents"""
        if verbose:
            self.actor.summary()
            self.critic.summary()

        self.main_loop(self.episodes, train=True, verbose=verbose)

    def evaluate(self, episodes, sigma, mean, models_dir, multiple_agents=True):
        """Evaluate function for presentation. Supports evaluation of multiple agents"""
        self.noise.reset(sigma=sigma, mean=mean)
        if multiple_agents:
            agentIds = [f.path.split('\\')[1] for f in os.scandir(models_dir) if f.is_dir()]
            for agentId in agentIds:
                self.load_weights(agentId)
                avg_reward = self.main_loop(episodes=episodes, train=False, verbose=True)
                print('********************************************************')
                print(f"Average reward for agent {agentId}: {avg_reward}")
            print('********************************************************')
        else:
            # this supports evaluation of older tests
            avg_reward = self.main_loop(episodes=episodes, train=False, verbose=True)
            print(f"Average reward for agent: {avg_reward}")


if __name__ == '__main__':
    tf.compat.v1.ConfigProto().gpu_options.allow_growth = True
    env = gym.make("CarRacing-v0")

    # set parameters
    ACTOR_LR = 0.00001
    CRITIC_LR = 0.02
    EPISODES = 1500
    GAMMA = 0.99  # discount factor for past rewards
    TAU = 0.005  # discount factor for future rewards
    PHI = 0.25  # reducing severity of actor's actions
    STD_DEV = np.array([0.1, 0.8])
    MEAN = np.array([0.0, 0.0])
    TESTID = 'nn_test5'

    ou_noise = ActionNoise(sigma=STD_DEV, mean=MEAN)
    ddpg = DDPG(env, noise=ou_noise, learning_rates=(ACTOR_LR, CRITIC_LR), episodes=EPISODES, gamma=GAMMA, tau=TAU, phi=PHI, saving_dir=TESTID)
    ddpg.train(verbose=True)

    STD_DEV = np.array([0.0, 0.08])
    MEAN = np.array([0.0, -0.1])
    ddpg.evaluate(episodes=10, sigma=STD_DEV, mean=MEAN, models_dir=TESTID)

    # TODO TEST:
    #  get loss metric
    #  kernel vs activity regularizer
    #  test batchnorm before and after activation
