# -*- coding: utf-8 -*-
import sys
import csv
import time

from AirSimClient import *
from argparse import ArgumentParser  ###

import numpy as np
import random ##

from matplotlib import pyplot as plt ##
import pylab

from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential


EPISODES = 300


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.load_model = False

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000

        # 리플레이 메모리, 최대 크기 2000
        self.memory = deque(maxlen=2000)

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 타깃 모델 초기화
        self.update_target_model()

        if self.load_model:
            self.model.load_weights("./save_model/stability.h5")

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Get input layer value
    def get_state(self):
        orien = client.getMultirotorState().kinematics_estimated.orientation
        an_vel = client.getMultirotorState().kinematics_estimated.angular_velocity
        lin_vel = client.getMultirotorState().kinematics_estimated.linear_velocity
        posit = client.getMultirotorState().kinematics_estimated.position
        state = np.array([orien.x_val*(180/3.141592), orien.y_val*(180/3.141592), orien.z_val*(180/3.141592), an_vel.x_val, an_vel.y_val, an_vel.z_val, lin_vel.x_val, lin_vel.y_val, lin_vel.z_val, posit.x_val, posit.y_val, posit.z_val], np.float64)
        return state

    # 입실론 탐욕 정책으로 행동 선택                    ###################33
    def get_action(self, state):
        
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def set_action(self, action):           #concering output actions
        scaling_factor = 0.01
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
        
    def compute_reward(self, rp, state):
        collision_info = client.getCollisionInfo()
        info = collision_info.has_collided
        er_lim = 5

        pit_er = np.absolute(state[0, 0])
        ro_er = np.absolute(state[0, 1])
        ya_er = np.absolute(state[0, 2])
        # po_er = np.array([input_state[10:]])-np.array([quad_state[10:]])

        if info:
            reward = -10.0
            # rp = reset_quad(rp_init)

        if pit_er and ro_er < er_lim:
            reward = 1.0
        else:
            reward = -1.0

        return reward, info

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []


        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])
        
        # 현재 상태에 대한 모델의 큐함수
        # 다음 상태에 대한 타깃 모델의 큐함수
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)
       
        # 벨만 최적 방정식을 이용한 업데이트 타깃                ##########check plz!!!
        for i in range(self.batch_size):
            for j in range(self.action_size):
                if dones[i]:
                    target[i][actions[j]] = rewards[i]
                else:
                    target[i][[j]] = rewards[i] + self.discount_factor * (np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

def reset_quad(rp_init):
    print('Reseting Quad')
    client.reset()
    client.enableApiControl(True)
    rp = [rp_init, rp_init, rp_init, rp_init]
    client.moveByRotorSpeed(rp[0],rp[1],rp[2],rp[3], 2)
    return rp

if __name__ == "__main__":
    #  Connection for Airsim
    client = MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    rp_init = 500
    rp=[rp_init, rp_init, rp_init, rp_init]
    client.moveByRotorSpeed(rp[0],rp[1],rp[2],rp[3], 2)   # client.takeoff()
    time.sleep(0.5)

    state_size = 12
    action_size = 9
    # DQN 에이전트 생성
    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        # env 초기화
        reset_quad(rp_init)
        state = agent.get_state()   
        state = np.reshape(state, [1, state_size])

        while not done:
            # 현재 상태로 행동을 선택
            act = agent.get_action(state)
            action = agent.set_action(act)
            # reward , done, info requires
            reward, done = agent.compute_reward(rp, state)
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            
            next_state = agent.get_state()    #<- next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # 에피소드가 중간에 끝나면 -100 보상
            reward = reward if not done or score == 499 else -100

            # 리플레이 메모리에 샘플 <s, a, r, s'> 저장
            agent.append_sample(state, act, reward, next_state, done)
            # 매 타임스텝마다 학습
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            score += reward
            state = next_state

            if done:
                # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                agent.update_target_model()

                score = score if score == 500 else score + 100
                # 에피소드마다 학습 결과 출력
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/stability.png")
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)

                # 이전 10개 에피소드의 점수 평균이 490보다 크면 학습 중단
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    agent.model.save_weights("./save_model/stability.h5")
                    sys.exit()
