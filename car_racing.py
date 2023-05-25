import os
from shutil import rmtree
import utils
import gym
import numpy as np
from csv import writer
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, Concatenate, BatchNormalization
from keras.optimizers import Adam
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay
from keras.utils import plot_model


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
    def __init__(self, env: gym.envs.registration, noise: ActionNoise, episodes,
                 lr, gradClip, gamma, tau, phi, dropout, breakCount, runId, initModelPath=None):
        # initialise params
        self.env = env
        self.noise = noise
        self.episodes = episodes
        self.G = gamma
        self.T = tau
        self.PHI = phi
        self.D_RATE = dropout
        self.BREAK = breakCount

        actorLR, criticLR = lr
        self.actorAdam = Adam(learning_rate=actorLR, clipnorm=gradClip)
        self.criticAdam = Adam(learning_rate=criticLR, clipnorm=gradClip)

        self.stateSpace = 96, 96, 3
        self.actionSpace = 2,  # we're combining acceleration and braking into one axis

        # create a directory for models
        if not os.path.exists(runId): os.mkdir(runId)
        self.savingDir = runId

        # create a directory for logs
        logDir = 'log/' + runId
        self.logger = tf.summary.create_file_writer(logDir)

        # initialise the buffer
        self.buffer = Buffer(self.stateSpace, self.actionSpace, buffer_capacity=60000, batch_size=64)

        # initialise the networks
        self.actor = self.actor_network()
        self.critic = self.critic_network()
        self.targetActor = self.actor_network()
        self.targetCritic = self.critic_network()
        if initModelPath is not None:
            self.load_weights(modelPath=initModelPath)
        else:
            self.targetActor.set_weights(self.actor.get_weights())
            self.targetCritic.set_weights(self.critic.get_weights())

    def actor_network(self) -> Model:
        """Defining the actor network"""
        inputs = Input(shape=self.stateSpace, name='state_input')
        outputs = Conv2D(32, kernel_size=8, strides=4, activation='relu', activity_regularizer='l2', use_bias=False)(inputs)
        outputs = BatchNormalization()(outputs)
        outputs = Dropout(self.D_RATE)(outputs)
        outputs = Conv2D(64, kernel_size=4, strides=3, activation='relu', activity_regularizer='l2', use_bias=False)(outputs)
        outputs = BatchNormalization()(outputs)
        outputs = Dropout(self.D_RATE)(outputs)
        outputs = Conv2D(64, kernel_size=3, strides=1, activation='relu', activity_regularizer='l2', use_bias=False)(outputs)
        outputs = BatchNormalization()(outputs)
        outputs = Dropout(self.D_RATE)(outputs)
        outputs = Flatten(name='flatten')(outputs)
        outputs = Dense(512, activation='relu', activity_regularizer='l2', use_bias=False)(outputs)
        outputs = BatchNormalization()(outputs)
        outputs = Dropout(self.D_RATE)(outputs)
        outputs = Dense(64, activation='relu', activity_regularizer='l2', use_bias=False)(outputs)
        outputs = BatchNormalization()(outputs)
        outputs = Dense(2, activation='tanh', kernel_initializer='glorot_normal', activity_regularizer='l2', use_bias=False, name='output')(outputs)
        model = Model(inputs, outputs, name='actor')
        # plot_model(model, to_file='actor.png')
        return model

    def critic_network(self) -> Model:
        """Defining the critic network"""
        # state branch
        state_inputs = Input(shape=self.stateSpace, name='state_input')
        state_outputs = Conv2D(32, kernel_size=8, strides=4, activation='relu', activity_regularizer='l2')(state_inputs)
        state_outputs = BatchNormalization()(state_outputs)
        state_outputs = Dropout(self.D_RATE)(state_outputs)
        state_outputs = Conv2D(64, kernel_size=4, strides=3, activation='relu', activity_regularizer='l2')(state_outputs)
        state_outputs = BatchNormalization()(state_outputs)
        state_outputs = Dropout(self.D_RATE)(state_outputs)
        state_outputs = Conv2D(64, kernel_size=3, strides=1, activation='relu', activity_regularizer='l2')(state_outputs)
        state_outputs = BatchNormalization()(state_outputs)
        state_outputs = Dropout(self.D_RATE)(state_outputs)
        state_outputs = Flatten(name='flatten')(state_outputs)
        state_outputs = Dense(512, activation='relu', activity_regularizer='l2')(state_outputs)

        # action branch
        action_inputs = Input(shape=2, name='action_input')
        action_outputs = Dense(16, activation='relu', activity_regularizer='l2')(action_inputs)

        # common branch
        concat = Concatenate(name='concat')([state_outputs, action_outputs])
        outputs = BatchNormalization()(concat)
        outputs = Dropout(self.D_RATE)(outputs)
        outputs = Dense(128, activation='relu', activity_regularizer='l2')(outputs)
        outputs = BatchNormalization()(outputs)
        outputs = Dropout(self.D_RATE)(outputs)
        outputs = Dense(128, activation='relu', activity_regularizer='l2', use_bias=False)(outputs)
        outputs = Dense(1, name='action_value_output', use_bias=False)(outputs)

        # outputs single value for state-action
        model = Model([state_inputs, action_inputs], outputs, name='critic')
        # plot_model(model, to_file='critic.png')
        return model

    @staticmethod
    def preprocess(state: np.ndarray) -> np.ndarray:
        """Preprocessing the state representation image"""
        processed_state = np.array(state, copy=True, dtype=np.float32)

        # enlarge the speed bar
        for i in range(88, 94): processed_state[i, 0:12, :] = processed_state[i, 12, :]

        # unify grass colour
        # from [102, 229, 102] to [102, 204, 102]
        processed_state[:, :, 1][processed_state[:, :, 1] == 229] = 204

        # unify track colour
        # from [102, 105, 102] and [102, 107, 102] to [102, 102, 102]
        processed_state[:, :, 1][(processed_state[:, :, 1] == 105) | (processed_state[:, :, 1] == 107)] = 102

        # normalise
        processed_state /= np.max(processed_state)

        return processed_state

    def policy(self, state) -> tuple[np.ndarray, np.ndarray]:
        """Returns the next action in two formats"""
        # get the best action according to the actor network and add noise to it
        train_action = tf.squeeze(self.actor(state)).numpy() + self.noise()

        # reduce the action range to improve training, clip the actions and convert to a format accepted by the environment
        env_action = train_action * self.PHI
        env_action = np.array([env_action[0].clip(-1, 1), env_action[1].clip(0, 1), -env_action[1].clip(-1, 0)])

        return env_action, train_action

    @tf.function
    def update_gradients(self, batches: tf.tuple):
        """Training and updating the networks"""
        state_batch, action_batch, reward_batch, next_state_batch = batches

        # gradient descent for critic network
        with tf.GradientTape() as tape:
            target_actions = self.targetActor(next_state_batch, training=True)
            y = reward_batch + self.G * self.targetCritic([next_state_batch, target_actions], training=True)
            critic_value = self.critic([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.criticAdam.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        # gradient ascent for actor network
        with tf.GradientTape() as tape:
            actions = self.actor(state_batch, training=True)
            critic_value = self.critic([state_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actorAdam.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

    @tf.function
    def update_targets(self):
        """Updating target networks"""
        for (a, b) in zip(self.targetActor.variables, self.actor.variables): a.assign(b * self.T + a * (1 - self.T))
        for (a, b) in zip(self.targetCritic.variables, self.critic.variables): a.assign(b * self.T + a * (1 - self.T))

    def save_weights(self, agentId=None):
        """Save the weights"""
        if agentId is not None: path = f"{self.savingDir}/{agentId}"; os.mkdir(path)
        else: path = self.savingDir
        self.actor.save_weights(f"{path}/actor.h5")
        self.critic.save_weights(f"{path}/critic.h5")
        self.targetActor.save_weights(f"{path}/target_actor.h5")
        self.targetCritic.save_weights(f"{path}/target_critic.h5")

    def delete_weights(self, agentId): rmtree(path=f"{self.savingDir}/{agentId}")

    def load_weights(self, modelPath=None, modelId=None):
        """Load the weights"""
        path = modelPath if modelPath else f"{self.savingDir}/{modelId}" if modelId else self.savingDir
        self.actor.load_weights(f"{path}/actor.h5")
        self.critic.load_weights(f"{path}/critic.h5")
        self.targetActor.load_weights(f"{path}/target_actor.h5")
        self.targetCritic.load_weights(f"{path}/target_critic.h5")

    def main_loop(self, episodes, train=True, verbose=True):
        avgReward = 0
        envStep = 0
        top10Models = []
        rewards = []
        for ep in range(episodes):
            # env.reset is annoying because it prints to console, so I'm disabling print around it using system settings
            with utils.BlockPrint(): prevState = self.preprocess(self.env.reset())
            # init variables
            epReward, breakCounter, episodicStep = 0, 0, 0
            while True:
                episodicStep += 1
                if verbose: self.env.render()
                action, trainAction = self.policy(tf.expand_dims(tf.convert_to_tensor(prevState), 0))  # get new action based on a current state
                state, reward, done, info = self.env.step(action)  # perform an action
                state = self.preprocess(state)  # simplify the state image
                epReward += reward  # append the reward
                if train:
                    self.buffer.record((prevState, trainAction, reward, state))  # save the transition in the buffer
                    self.update_gradients(self.buffer.get_batches())  # update gradients
                    self.update_targets()  # update targets

                # increment the break counter if there was no reward and reset if there was
                if reward < 0: breakCounter += 1
                else: breakCounter = 0
                # terminate episode if agent earns no reward after n steps
                if done or breakCounter == self.BREAK: break
                prevState = state

            envStep += episodicStep
            rewards.append(epReward)
            avgReward = np.mean(rewards[-50:])

            # write logs
            print(f"Episode {ep}; reward: {epReward:.3f}; average reward: {avgReward:.3f}; time steps: {episodicStep}")
            with self.logger.as_default():
                tf.summary.scalar(name="50AvgReward", data=avgReward, step=ep)
                tf.summary.scalar(name="Reward", data=epReward, step=ep)
                tf.summary.scalar(name="TimeSteps", data=episodicStep, step=ep)
                tf.summary.scalar(name="TimeStepsTotal", data=envStep, step=ep)

            if train:
                # save top 10 best scoring models for evaluation
                # model must have minimum 700 points to qualify for evaluation
                # logic below overwrites the lowest-scoring model if there are more 10 models saved
                if epReward > 700:
                    bestModelsRewards = np.array([rewards[i] for i in top10Models])
                    if len(top10Models) < 10:
                        top10Models.append(ep)
                        self.save_weights(ep)
                    elif epReward > np.min(bestModelsRewards):
                        dropout_agent = top10Models[np.argmin(bestModelsRewards)]
                        self.delete_weights(dropout_agent)  # delete dropout agents weights
                        top10Models[top10Models.index(dropout_agent)] = ep
                        self.save_weights(ep)  # save new weights

        if not train: return avgReward

    def train(self, verbose=True):
        """Main function for training the agents"""
        if verbose:
            self.actor.summary()
            self.critic.summary()

        self.main_loop(self.episodes, train=True, verbose=verbose)

    def evaluate(self, episodes, noise: ActionNoise, breakCount, modelsDir=None, modelPath=None):
        """Evaluate function for presentation. Supports evaluation of multiple agents"""
        self.noise = noise
        self.BREAK = breakCount
        if modelsDir is not None:
            modelIds = [f.path.split('\\')[1] for f in os.scandir(modelsDir) if f.is_dir()]
            for modelId in modelIds:
                self.load_weights(modelId=modelId)
                print(f"\nEVALUATING MODEL {modelId}")
                avg_reward = self.main_loop(episodes=episodes, train=False, verbose=True)
                print(f"AVERAGE REWARD FOR MODEL {modelId}: {avg_reward}\n")
        elif modelPath is not None:
            # this supports evaluation of singular models
            self.load_weights(modelPath=modelPath)
            avg_reward = self.main_loop(episodes=episodes, train=False, verbose=True)
            print(f"Average reward for agent: {avg_reward}")


if __name__ == '__main__':
    tf.compat.v1.ConfigProto().gpu_options.allow_growth = True
    env = gym.make("CarRacing-v0")

    # set parameters
    EPISODES = 400
    ACTOR_LR = ExponentialDecay(initial_learning_rate=2e-5, decay_steps=1000, decay_rate=0.96)
    CRITIC_LR = ExponentialDecay(initial_learning_rate=4e-2, decay_steps=1000, decay_rate=0.96)
    GAMMA = 0.99  # discount factor for past rewards
    TAU = 0.005  # discount factor for future rewards
    PHI = np.array([0.25, 0.3])  # reducing severity of actor's actions
    STD_DEV = np.array([0.1, 0.8])  # high noise guarantees exploration
    MEAN = np.array([0.0, -0.05])
    DROPOUT = 0.2
    GRADCLIP = 1
    BREAK_COUNT = 150  # terminating the episode after steps without reward
    # MODELPATH = 'final_model'
    TESTID = 'eval'

    ou_noise = ActionNoise(sigma=STD_DEV, mean=MEAN)
    ddpg = DDPG(env, noise=ou_noise, lr=(ACTOR_LR, CRITIC_LR), gradClip=GRADCLIP, episodes=EPISODES,
                gamma=GAMMA, tau=TAU, phi=PHI, dropout=DROPOUT, breakCount=BREAK_COUNT, runId=TESTID)
    # ddpg.train(verbose=True)  # comment this line when just running evaluation

    BREAK_COUNT = 35
    MODELPATH = 'final_model'  # model dir name
    STD_DEV = np.array([0.0, 0.0])  # removing noise for evaluation
    MEAN = np.array([0.0, -0.05])  # leaving the speed factor of the mean to prevent model from dealing with the speed it's not used to
    ou_noise = ActionNoise(sigma=STD_DEV, mean=MEAN)
    ddpg.evaluate(episodes=100, noise=ou_noise, breakCount=BREAK_COUNT, modelPath=MODELPATH)
