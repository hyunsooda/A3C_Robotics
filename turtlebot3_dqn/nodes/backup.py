#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import os
import json
import numpy as np
import random
import time
import sys
import threading
from multiprocessing import Process
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
from src.turtlebot3_dqn.environment_stage_1 import Env
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Activation, Input 
from keras.models import Model
from keras import backend as K


EPISODES = 3000
rospy.init_node('turtlebot3_dqn_stage_1')
pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
result = Float32MultiArray()
get_action = Float32MultiArray()


class A3CAgent:
    def __init__(self, state_size, action_size):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/save_model/stage_1_')
        self.result = Float32MultiArray()

        self.load_model = False
        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 2000
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 64
        self.states, self.actions, self.rewards = [], [], []
        self.threads = 1

        self.actor, self.critic = self.buildModel()
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        if self.load_model:
            self.loadModel()

    def loadModel(self):
        self.actor.load_weights(self.dirPath + "_actor.h5")
        self.critic.load_weights(self.dirPath + "_critic.h5")

    def saveModel(self):
        self.actor.save_weights(self.dirPath + "_actor.h5")
        self.critic.save_weights(self.dirPath + "_critic.h5")

    def buildModel(self):
        input = Input(shape=(self.state_size,))
        dropout = 0.2
        fc1 = Dense(64, activation='relu', kernel_initializer='lecun_uniform')(input)
        fc2 = Dense(64, activation='relu', kernel_initializer='lecun_uniform')(fc1)
        fc3 = Dropout(dropout)(fc2)
        
        policy = Dense(self.action_size, activation='softmax')(fc3)
        value = Dense(1, activation='linear')(fc3)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()
        
        return actor, critic

    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        policy = self.actor.output

        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob + 1e-10) * advantages
        cross_entropy = -K.sum(cross_entropy)

        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        loss = cross_entropy + 0.01 * entropy

        optimizer = RMSprop(lr=self.learning_rate, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.actor.trainable_weights, [],loss)
        train = K.function([self.actor.input, action, advantages],
                           [loss], updates=updates)
        return train

    def critic_optimizer(self):
        discounted_prediction = K.placeholder(shape=(None,))

        value = self.critic.output

        loss = K.mean(K.square(discounted_prediction - value))

        optimizer = RMSprop(lr=self.learning_rate, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.critic.trainable_weights, [],loss)
        train = K.function([self.critic.input, discounted_prediction],
                           [loss], updates=updates)
        return train

    def train(self):
        agents = [Agent(self.action_size, self.state_size,
                        [self.actor, self.critic], 
                        self.optimizer, self.discount_factor, self.dirPath, "goal" + str(i), i)
                  for i in range(self.threads)]
        ags = []

        for agent in agents:
            time.sleep(1)
            # os.putenv('ROS_MASTER_URI','http://localhost:1151' + str(agent.thread_num))
            # os.environ["ROS_MASTER_URI"] = "http://localhost:11513"
            agent.start()
        #     ag = Process(target=agent.run)
        #     ags.append(ag)
        #     ag.start()
        
        # for ag in ags:
        #     ag.join()

        while True:
            time.sleep(60 * 5) # for 5 miniute
            self.saveModel()

class Agent(threading.Thread):
    def __init__(self, action_size, state_size, model, optimizer, discount_factor, dirPath, model_name, thread_num):
        threading.Thread.__init__(self)

        self.action_size = action_size
        self.state_size = state_size
        self.actor, self.critic = model
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.dirPath = dirPath
        self.model_name = model_name
        self.states, self.actions, self.rewards = [], [], []
        self.local_actor, self.local_critic = self.build_local_model()
        self.thread_num = thread_num

        self.t_max = 20
        self.t = 1

    def discounted_prediction(self, rewards, done):
        discounted_prediction = np.zeros_like(rewards)
        running_add = 0

        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_prediction[t] = running_add
        return discounted_prediction

    def train_model(self, done):
        discounted_prediction = self.discounted_prediction(self.rewards, done)

        states = np.asarray(self.states)
        # print("value")
        # print(states)
        # print(self.states.reshape(1, len(self.state)))
        # values = self.critic.predict(self.states.reshape(1, len(self.state)))
        values = self.critic.predict(states)
        values = np.reshape(values, len(values))

        advantages = discounted_prediction - values

        self.optimizer[0]([states, self.actions, advantages])
        self.optimizer[1]([states, discounted_prediction])
        self.states, self.actions, self.rewards = [], [], []

    def build_local_model(self):
        input = Input(shape=(self.state_size,))
        dropout = 0.2
        fc = Dense(64, activation='relu', kernel_initializer='lecun_uniform')(input)
        fc = Dense(64, activation='relu', kernel_initializer='lecun_uniform')(fc)
        fc = Dropout(dropout)(fc)
        
        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        local_actor = Model(inputs=input, outputs=policy)
        local_critic = Model(inputs=input, outputs=value)

        local_actor._make_predict_function()
        local_critic._make_predict_function()

        local_actor.set_weights(self.actor.get_weights())
        local_critic.set_weights(self.critic.get_weights())
        
        local_actor.summary()
        local_critic.summary()

        return local_actor, local_critic
    
    def update_local_model(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())
        
    def getAction(self, state):
        # print("actor")
        # print(state.reshape(1, len(state)))
        policy = self.local_actor.predict(state.reshape(1, len(state)))[0]
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    def append_sample(self, state, action, reward):
        self.states.append(state)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    def run(self):
        # os.putenv('ROS_MASTER_URI','http://localhost:1151' + str(self.thread_num))
        os.environ["ROS_MASTER_URI"] = "http://localhost:1151" + str(self.thread_num)
        agent = Process(target=self.agentTrain)
        agent.start()
        agent.join()

    def agentTrain(self):
        # os.putenv('ROS_MASTER_URI','http://localhost:1151' + str(self.thread_num))
        os.environ["ROS_MASTER_URI"] = "http://localhost:1151" + str(self.thread_num)
        print(os.environ["ROS_MASTER_URI"])
        print(os.popen('echo $ROS_MASTER_URI').read())
        env = Env(self.action_size, model_name=self.model_name)

        scores, episodes = [], []
        global_step = 0
        start_time = time.time()

        for e in range(0, EPISODES):
            # os.putenv('ROS_MASTER_URI','http://localhost:1151' + str(self.thread_num))
            done = False
            print("step?")
            state = env.reset()
            print("www?")
            score = 0
            print("episode : ", e)
            for t in range(6000):
                # os.putenv('ROS_MASTER_URI','http://localhost:1151' + str(self.thread_num))
                action, policy = self.getAction(state)

                next_state, reward, done = env.step(action)

                self.append_sample(state, action, reward)

                if self.t >= self.t_max or done:
                    self.train_model(done)
                    self.update_local_model()
                    self.t = 1

                self.t += 1
                score += reward
                state = next_state
                get_action.data = [action, score, reward]
                pub_get_action.publish(get_action)
            
                if t >= 500:
                    rospy.loginfo("Time out!!")
                    done = True

                if done:
                    result.data = [score, np.max(policy)]
                    pub_result.publish(result)
                    scores.append(score)
                    episodes.append(e)
                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)

                    rospy.loginfo('Ep: %d score: %.2f time: %d:%02d:%02d',
                                e, score,  h, m, s)
                    break

                global_step += 1

if __name__ == '__main__':
    global_agent = A3CAgent(state_size=26, action_size=5)
    global_agent.train()