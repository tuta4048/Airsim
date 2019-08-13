from AirSimClient import *
from argparse import ArgumentParser
import numpy as np
from cntk.core import Value
from cntk.initializer import he_uniform
from cntk.layers import Sequential, Convolution2D, Dense, default_options
from cntk.layers.typing import Signature, Tensor
from cntk.learners import adam, learning_rate_schedule, momentum_schedule, UnitType
from cntk.logging import TensorBoardProgressWriter
from cntk.ops import abs, argmax, element_select, less, relu, reduce_max, reduce_sum, square
from cntk.ops.functions import CloneMethod, Function
from cntk.train import Trainer
import csv
import time
from matplotlib import pyplot as plt
# from cntk.device import try_set_default_device, gpu
# try_set_default_device(gpu(0))

class RepMem(object):
    def __init__(self, size, sample_shape, history_length=4):
        self._pos = 0
        self._count = 0
        self._max_size = size
        self._history_length = max(1, history_length)
        self._state_shape = sample_shape
        self._states = np.zeros((size,) + sample_shape, dtype=np.float32)
        self._actions = np.zeros(size, dtype=np.uint8)
        self._rewards = np.zeros(size, dtype=np.float32)
        self._terminals = np.zeros(size, dtype=np.float32)

    def __len__(self):
        return self._count

    def append(self, state, action, reward, done):# not if true it return assert error
        # assert state.shape == self._state_shape, \
        #     'Invaild state shape (required: %s, got: %s)' % (self._state_shape, state.shape)

        self._states[self._pos] = state
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._terminals[self._pos] = done

        self._count = max(self._count, self._pos + 1)
        self._pos = (self._pos + 1) % self._max_size

    def sample(self, size):
        count, pos, history_len, terminals = self._count - 1, self._pos, \
                                             self._history_length, self._terminals
        indexes = []

        while len(indexes) < size:
            index = np.random.randint(history_len, count)

            if index not in indexes:

                if not (index >= pos > index - history_len):
                    if not terminals[(index - history_len):index].any():
                        indexes.append(index)

        return indexes

    def minibatch(self, size):

        indexes = self.sample(size)

        pre_states = np.array([self.get_state(index) for index in indexes], dtype=np.float32)
        post_states = np.array([self.get_state(index + 1) for index in indexes], dtype=np.float32)
        actions = self._actions[indexes]
        rewards = self._rewards[indexes]
        dones = self._terminals[indexes]

        return pre_states, actions, post_states, rewards, dones

    def get_state(self, index):

        if self._count == 0:
            raise IndexError('Empty Memory') # occur error when ture

        index %= self._count
        history_length = self._history_length

        if index >= history_length:
            return self._states[(index - (history_length - 1)):index + 1, ...]
        else:
            indexes = np.arange(index - history_length + 1, index + 1)
            return self._states.take(indexes, mode='wrap', axis=0)

class History(object):
    """docstring for History"""
    def __init__(self, shape):
        self._buffer = np.zeros(shape, dtype=np.float32)

    @property
    def value(self):
        return self._buffer

    def append(self, state):
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1] = state

    def reset(self):
        self._buffer.fill(0)   # fill all the arange number as 0

class LinearEpsilonAnnealingExplorer(object):
    """docstring for LinearEpsilonAnnealingExplorer"""
    def __init__(self, start, end, steps):
        self._start = start
        self._stop = end
        self._steps = steps

        self._step_size = (end - start) / steps

    def __call__(self, num_actions):
        
        return np.random.choice(num_actions)    #random action choicer

    def _epsilon(self, step):
        if step < 0:
            return self._start
        elif step > self._steps:
            return self._stop
        else:
            return self._step_size * step + self._start

    def is_exploring(self, step):
        return np.random.rand() < self._epsilon(step)

def huber_loss(y, y_hat, delta):
    half_delta_squared = 0.5 * delta * delta
    error = y - y_hat
    abs_error = abs(error)

    less_than = 0.5 * square(error)
    more_than = (delta * abs_error) - half_delta_squared
    loss_per_sample = element_select(less(abs_error, delta), less_than, more_than)

    return reduce_sum(loss_per_sample, name='loss')

class DQAgent(object):
    """docstring for DQAgent"""                                                             ############should modify @!@!
    def __init__(self, input_shape, nb_actions,
                 gamma=0.99, explorer=LinearEpsilonAnnealingExplorer(1, 0.1, 1000000),
                 learning_rate=0.00025, momentum=0.95, minibatch_size=32,
                 memory_size=500000, train_after=10000, train_interval=4,
                 target_update_interval=10000, monitor=True):
        self.input_shape = input_shape
        self.nb_actions = nb_actions
        self.gamma = gamma
        self._train_after = train_after
        self._train_interval = train_interval
        self._target_update_interval = target_update_interval
        self._explorer = explorer
        self._minibatch_size = minibatch_size
        self._history = History(input_shape)
        self._memory = RepMem(memory_size, input_shape[1:], 4)
        self._num_actions_taken = 0
        self._episode_rewards, self._episode_q_means, self._episode_q_stddev = [], [], []

        with default_options(activation=relu, init=he_uniform()):
            self._action_value_net = Sequential([
                Dense(input_shape, init=he_uniform(scale=0.01)),
                Dense(input_shape),
                Dense(nb_actions, activation=None, init=he_uniform(scale=0.01))])

        self._action_value_net.update_signature(Tensor[input_shape])

        self._target_net = self._action_value_net.clone(CloneMethod.freeze)


        @Function
        @Signature(post_states=Tensor[input_shape], rewards=Tensor[()], terminals=Tensor[()])
        def compute_q_targets(post_states, rewards, terminals):
            return element_select(
                terminals,
                rewards,
                gamma * reduce_max(self._target_net(post_states), axis=0) + rewards,
            )

        @Function
        @Signature(pre_states=Tensor[input_shape], actions=Tensor[nb_actions],
                   post_states=Tensor[input_shape], rewards=Tensor[()], terminals=Tensor[()])
        def criterion(pre_states, actions, post_states, rewards, terminals):
            q_targets = compute_q_targets(post_states, rewards, terminals)

            q_acted = reduce_sum(self._action_value_net(pre_states) * actions, axis=0)

            return huber_loss(q_targets, q_acted, 1.0)

        lr_schedule = learning_rate_schedule(learning_rate, UnitType.minibatch)
        m_schedule = momentum_schedule(momentum)
        vm_schedule = momentum_schedule(0.999)
        l_sgd = adam(self._action_value_net.parameters, lr_schedule,
                     momentum=m_schedule, variance_momentum=vm_schedule)

        self._metrics_writer = TensorBoardProgressWriter(freq=1, log_dir='metrics', model=criterion) if monitor else None
        self._learner = l_sgd
        self._trainer = Trainer(criterion, (criterion, None), l_sgd, self._metrics_writer)

    def act(self, state):
        self._history.append(state)

        if self._explorer.is_exploring(self._num_actions_taken):
            action = self._explorer(self.nb_actions)
        else:
            env_with_history = self._history.value
            q_values = self._action_value_net.eval(
                env_with_history.reshape((1,) + env_with_history.shape)
            )

            self._episode_q_means.append(np.mean(q_values))
            self._episode_q_stddev.append(np.std(q_values))
            action = q_values.argmax()

        self._num_actions_taken += 1
        return action

    def observe(self, old_state, action, reward, done):
        self._episode_rewards.append(reward)

        if done:
            if self._metrics_writer is not None:
                self._plot_metrics()
            self._episode_rewards, self._episode_q_means, self._episode_q_stddev = [], [], []

            self._history.reset()

        self._memory.append(old_state, action, reward, done)

    def train(self):

        agent_step = self._num_actions_taken

        if agent_step >= self._train_after:
            if (agent_step % self._train_interval) == 0:
                pre_states, actions, post_states, rewards, terminals = self._memory.minibatch(self._minibatch_size)

                self._trainer.train_minibatch(
                    self._trainer.loss_function.argument_map(
                        pre_states=pre_states,
                        actions=Value.one_hot(actions.reshape(-1,1).tolist(), self.nb_actions),
                        post_states=post_states,
                        rewards=rewards,
                        terminals=terminals
                    )
                )

            if (agent_step % self._target_update_interval) == 0:
                self._target_net = self._action_value_net.clone(CloneMethod.freeze)
                filename = "model\model%d" % agent_step # save ???? not good at using %d
                self._trainer.save_checkpoint(filename)

    def _plot_metrics(self):

        if len(self._episode_q_means) > 0:
            mean_q = np.asscalar(np.mean(self._episode_q_means))
            self._metrics_writer.write_value('Mean Q per ep.', mean_q, self._num_actions_taken)

        if len(self._episode_q_stddev) > 0:
            std_q = np.asscalar(np.mean(self._episode_q_stddev))
            self._metrics_writer.write_value('Mean Std Q per ep.', std_q, self._num_actions_taken)

        self._metrics_writer.write_value('Sum rewards per ep', sum(self._episode_rewards), self._num_actions_taken)


def reset_quad(rp_init):
    print('Reseting Quad')
    client.reset()
    client.confirmConnection()
    client.enableApiControl(True)
    time.sleep(0.5)
    rp = [rp_init, rp_init, rp_init, rp_init]
    client.moveByRotorSpeed(rp[0],rp[1],rp[2],rp[3], 2)
    return rp

# get input layer value
def get_current_state():
    input_state = []
    orien = client.getMultirotorState().kinematics_estimated.orientation
    an_vel = client.getMultirotorState().kinematics_estimated.angular_velocity
    lin_vel = client.getMultirotorState().kinematics_estimated.linear_velocity
    posit = client.getMultirotorState().kinematics_estimated.position
    input_state = orien.x_val*(180/3.141592), orien.y_val*(180/3.141592), orien.z_val*(180/3.141592), an_vel.x_val, an_vel.y_val, an_vel.z_val, lin_vel.x_val, lin_vel.y_val, lin_vel.z_val, posit.x_val, posit.y_val, posit.z_val 
    
    return input_state
    
def interpret_action(action):           #concering output actions
    scaling_factor = 0.1
 
    if action == 0:
        client.moveByRotorSpeed(rp[0], rp[1], rp[2], rp[3], 0.001)
    elif action == 1:
        client.moveByRotorSpeed(rp[0]+scaling_factor, rp[1], rp[2], rp[3], 0.001)
        rp[0] += scaling_factor
    elif action == 2:
        client.moveByRotorSpeed(rp[0], rp[1]+scaling_factor, rp[2], rp[3], 0.001)
        rp[1] += scaling_factor
    elif action == 3:
        client.moveByRotorSpeed(rp[0], rp[1], rp[2]+scaling_factor, rp[3], 0.001)
        rp[2] += scaling_factor
    elif action == 4:
        client.moveByRotorSpeed(rp[0], rp[1], rp[2], rp[3]+scaling_factor, 0.001)
        rp[3] += scaling_factor
    elif action == 5:
        client.moveByRotorSpeed(rp[0]-scaling_factor, rp[1], rp[2], rp[3], 0.001)
        rp[0] -= scaling_factor
    elif action == 6:
        client.moveByRotorSpeed(rp[0], rp[1]-scaling_factor, rp[2], rp[3], 0.001)
        rp[1] -= scaling_factor
    elif action == 7:
        client.moveByRotorSpeed(rp[0], rp[1], rp[2]-scaling_factor, rp[3], 0.001)
        rp[2] -= scaling_factor
    elif action == 8:
        client.moveByRotorSpeed(rp[0], rp[1], rp[2], rp[3]-scaling_factor, 0.001)
        rp[3] -= scaling_factor
    return rp

def compute_reward(input_state, quad_state, rp_init, rp):
    er_lim = 5

    pit_er = np.absolute(quad_state[0])
    # print(pit_er)
    ro_er = np.absolute(quad_state[1])
    ya_er = np.absolute(quad_state[2])
    # po_er = np.array([input_state[10:]])-np.array([quad_state[10:]])

    if collision_info.has_collided:
        reward = -10
        rp = reset_quad(rp_init)

    if pit_er and ro_er < er_lim:
        reward = 1
    else:
        reward = -1

    return rp, reward

def isDone(reward):
    done = 0
    if reward <= -100:
        done = 1 
    return done

#  Connection for Airsim
client = MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)


# initial takeoff
rp_init = 500
rp=[rp_init, rp_init, rp_init, rp_init]
client.moveByRotorSpeed(rp[0],rp[1],rp[2],rp[3], 2)   # client.takeoff()
time.sleep(0.5)


# Make reinforcement learning agent
SizeRows = 1
SizeCols = 12
NumActions = 9            # number of actions!!
reward = 0
rewsum = 0
current_step = 0
agent = DQAgent((SizeRows, SizeCols), NumActions, monitor=True)    

# Training

epoch = 100
current_step = 0
max_steps = epoch * 250000

#input variable ******
input_state = get_current_state()  #RC input change!!!!!!!

plt.show()

while True:
    action = agent.act(input_state)
    quad_offset = interpret_action(action)
    time.sleep(0.5)

    quad_state = get_current_state()           # improtant part!!!!
    collision_info = client.getCollisionInfo()
    rp, reward = compute_reward(input_state, quad_state, rp_init, rp)
    rewsum += reward
    done = isDone(rewsum)
    print('Action, Rewards, Done, Current step:', action, rewsum, done, current_step)
    print('Rotor Speed:', rp)
    print('quad_state:', quad_state[:3])
    agent.observe(quad_state, action, rewsum, done)     
    agent.train()
    t=time.clock()
    plt.scatter(t,rewsum, color='b', zorder=2)
    plt.pause(0.0001)

    if done:                   # Commend to reset
        rp = reset_quad(rp_init)
        rewsum = 0      
        current_step += 1

    input_state = get_current_state()





