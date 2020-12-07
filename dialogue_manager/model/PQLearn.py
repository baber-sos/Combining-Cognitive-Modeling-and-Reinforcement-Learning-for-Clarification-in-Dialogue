import numpy as np
import random
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import math
import argparse
import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.distributions.categorical import Categorical

import torch
import torch.nn as nn

from model.model import model
import pickle
from state.state import INTENTS


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'is_terminal'))

class DQN(nn.Module):
    def __init__(self, obs_params, outputs, num_layers):
        super(DQN, self).__init__()
        self.net_layers = []
        linear_input_size = 1
        for i in obs_params:
            linear_input_size *= i

        layer_sizes = [linear_input_size, outputs]

        index = 1
        for i in range(num_layers - 1):
            new_layer_size = int((2/3) * layer_sizes[index - 1]) + int(layer_sizes[index])
            # new_layer_size = int(math.sqrt(layer_sizes[index - 1] * layer_sizes[index]))
            # new_layer_size = 10
            layer_sizes.insert(index, new_layer_size)
            index += 1
        print('These are layer sizes: %s' % str(layer_sizes)) 
        for i in range(num_layers):
            self.net_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            torch.nn.init.xavier_uniform_(self.net_layers[-1].weight)
            #torch.nn.init.kaiming_uniform_(self.net_layers[-1].weight, nonlinearity='relu')
        self.net_layers = nn.ModuleList(self.net_layers)

        # self.dim_reduce = linear_input_size 
        # output_size = int(os.getenv('REDUCED_DIM')) if os.getenv('REDUCED_DIM') != '-1' else linear_input_size
        # #self.dim_reduce
        # for i in range(num_layers):
        #     if i == num_layers - 1:
        #         output_size = outputs
        #     self.net_layers.append(nn.Linear(self.dim_reduce, output_size))
        #     # print('Before', self.net_layers[-1].weight)
        #     torch.nn.init.xavier_uniform_(self.net_layers[-1].weight)
        #     # print('After', self.net_layers[-1].weight)
        #     # exit()
        #     self.dim_reduce = output_size

    def forward(self, x):        
        reduced_out = x
        for i in range(len(self.net_layers) - 1):
            reduced_out = F.relu(self.net_layers[i](reduced_out))
            # reduced_out = F.leaky_relu(self.net_layers[i](reduced_out))
            # print('Activation')
            # reduced_out = torch.tanh(self.net_layers[i](reduced_out))
        # return torch.softmax(self.net_layers[-1](reduced_out), dim=1)
        # return torch.sigmoid(self.net_layers[-1](reduced_out))
        #return torch.tanh(self.net_layers[-1](reduced_out))
        return self.net_layers[-1](reduced_out)

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.prev_pos = -1
    
    def push(self, *args):
        if (len(self.memory) < self.capacity):
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.prev_pos = self.position
        self.position = (self.position + 1) % (self.capacity)
    
    def sample(self, batch_size):
        return random.sample(self.memory[:self.prev_pos] + \
            self.memory[self.prev_pos + 1 :], batch_size-1) + [self.memory[self.prev_pos]]
        # return random.sample(self.memory, batch_size)[:-1] + [self.memory[self.prev_pos]]
    def __len__(self):
        return len(self.memory)
    
    def empty(self):
        self.memory = []
        self.position = 0

class PQLearn(model):
    def __init__(self, num_actions):
        # print('Starting')
        super(PQLearn, self).__init__(num_actions)
        # model.__init__(num_actions)
        self.MAX_CONV_LEN = int(os.getenv('MAX_CONV_LEN'))
        if os.getenv('FEATURE_REP') == 'history':
            self.vector_dim = ((self.MAX_CONV_LEN + 1) * len(INTENTS)) + 3 + 1
        elif os.getenv('FEATURE_REP') == 'actions':
            self.vector_dim = 3 + (len(INTENTS)**2) + 1 + 1
            if os.getenv('FEATURE_TRANSFORM') == 'fft':
                self.vector_dim = int((int(self.vector_dim/2) + 1) * 2)
                print('This is the input dimension: %d' % self.vector_dim)
        self.num_actions = num_actions
        print('Number of actions: %d' % (num_actions,))

        self.EPS_END = float(os.getenv('EPS_END'))
        self.EPS_START = float(os.getenv('EPS_START'))
        self.steps_done = 0
        # self.steps_done = 204132
        self.EPS_DECAY = float(os.getenv('EPS_DECAY'))
        self.BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
        
        self.REPLAY_CAPACITY = int(os.getenv('REPLAY_CAPACITY'))
        
        self.test_memory = ReplayMemory(1)
        self.train_memory = ReplayMemory(self.REPLAY_CAPACITY)
        self.memory = self.train_memory
        self.NUM_LAYERS = int(os.getenv('NUM_LAYERS'))
        self.target_trail = int(os.getenv('TARGET_TRAIL'))
        self.device = os.getenv('DEVICE')

        self.GAMMA = float(os.getenv('GAMMA'))
        self.policy_net = DQN((self.vector_dim,), num_actions, self.NUM_LAYERS)
        self.target_net = DQN((self.vector_dim,), num_actions, self.NUM_LAYERS)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # self.optimizer = optim.Adam(self.policy_net.parameters(), \
        #     lr=float(os.getenv('LEARN_RATE')), weight_decay=(10**(-6)))
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), \
            lr=eval(os.getenv('LEARN_RATE')), weight_decay=eval(os.getenv('WEIGHT_DECAY')))
        # self.optimizer = optim.SGD(self.policy_net.parameters(), \
        #     lr=eval(os.getenv('LEARN_RATE')), weight_decay=eval(os.getenv('WEIGHT_DECAY')), momentum=0.9)
        
        # self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=4000)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, float(os.getenv('LR_DECAY')))
        # self.lr_scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, \
        #     max_lr=eval(os.getenv('LEARN_RATE')), epochs=eval(os.getenv('NUM_EPOCHS')), \
        #     steps_per_epoch=eval(os.getenv('ITEMS_PER_EPOCH')))

        ##statistics tracking so decisions can be made

        #keeping track for train phase
        self.train_loss_track = []
        self.train_reward_track = []
        self.train_successes = []
        self.train_action_choices = []
        self.train_learn_rate = []
        #keeping track for test phase
        self.test_loss_track = []
        self.test_reward_track = []
        self.test_successes = []
        self.test_action_choices = []

        #number of successes in test mode => imprecision_threshold : [success_count, failure_count]
        self.successes = [0, 0]

        self.action_choices = []

        self.loss_track = []
        self.reward_track = []

        self.temp_reward = 0.0
        self.is_test = False
        
        self.temp_start = float(os.getenv('temp_start'))
        self.temp_end = float(os.getenv('temp_end'))
        self.temp = self.temp_start #this is temperature and temperature difference
        max_exp = -1 * math.log(self.temp_end/self.temp_start)
        self.incr = max_exp/self.EPS_DECAY
        self.temp_diff = (self.temp_start - self.temp_end)/self.EPS_DECAY
        self.start_exp = 0

        if os.getenv('LOAD_MODEL') != 'False':
            print('Loading Model: %s' % (os.getenv('LOAD_MODEL')))
            model_path = os.getenv('MODELF') + '/trained_model' + os.getenv('LOAD_MODEL')
            model_state_dict = torch.load(model_path, map_location=torch.device(self.device))
            self.policy_net.load_state_dict(model_state_dict)
            self.target_net.load_state_dict(model_state_dict)

            # names = ['train_loss_track', 'train_reward_track', 'train_successes', 'test_loss_track', \
            # 'test_reward_track', 'test_successes', 'test_action_choices']
            # for name in names:
            #     file_name = os.path.dirname(__file__) + '/training_parameters/' + name + \
            #         os.getenv('MATCHER_STRAT') + os.getenv('SIMULATOR') + os.getenv('COLOR_MODEL') + \
            #         os.getenv('SIMU_IMPREC_THRESH') + '.pkl'
            #     cur_params = pickle.load(open(file_name, 'rb'))
            #     exec('self.' + name + '= cur_params')

            if os.getenv('TRIAL_MODE') == 'True':
                self.set_test_phase(2)

    def get_next_action(self, dialogue_state):
        state = dialogue_state.get_vector_representation()
        probs = np.array(state[0][:3].clone().detach().cpu())
        incomplete = state[0][-1]
        if os.getenv('FEATURE_TRANSFORM') == 'fft':
            state = torch.cat(tuple(*torch.rfft(state, len(state.shape), normalized=True))).view(1, -1)
        sample = random.random()
        if os.getenv('ACTION_SAMPLE_MODE') == 'boltzmann' and (not self.is_test):
            with torch.no_grad():
                if len(dialogue_state) == 1 and self.temp > self.temp_end:
                    self.start_exp += self.incr
                    self.temp = self.temp_start * math.exp(-1 * self.start_exp) 
                    #self.temp -= self.temp_diff
                print(self.policy_net(state))
                print('Temperature: %f Length: %d' %(self.temp, len(dialogue_state)))
                action_q_vals = self.policy_net(state).view(-1)
                action_softmax = F.softmax(action_q_vals/self.temp, dim=0)
                action = int(Categorical(action_softmax).sample())
                print('Botlzmann Softmax: %s Action Taken: %d' % (str(action_softmax), action))
                print('-----------------------------')
                return action

        if self.EPS_DECAY > 0:
            eps_thresh = self.EPS_END + (self.EPS_START - self.EPS_END) *\
                math.exp(-1. * self.steps_done/self.EPS_DECAY)
        self.steps_done += 1

        if self.is_test or (sample > eps_thresh):
            print('Following Policy: ', self.steps_done)
            print('Epsilon Value: %.2f, Sampled Value: %.2f' % (eps_thresh,sample))
            print('-----------------------------------')
            with torch.no_grad():
                q_values = self.policy_net(state)
                print(q_values)
                action = int(q_values.max(1)[1].view(1))

                # action = 2 if action == 0 else 4
                action = [2, 4, 6, 7][action]

                prev_action = dialogue_state._cur_state.intent
                conv_len = len(dialogue_state)
                #incomplete = state[0][-1]
                if self.is_test:
                    #probs = np.array(state[0][:3].clone().detach().cpu())
                    self.action_choices.append((tuple(probs), int(action), int(prev_action), int(conv_len), int(incomplete)))
                # print('Following Policy: ', self.steps_done, action)
                return action

        else:
            rand_actions = [2,4,6,7]
            #rand_actions = [i for i in range(self.num_actions)]
            # rand_actions = [2, 4]
            return int(torch.tensor(random.choice(rand_actions), \
                dtype=torch.long).view(1))
            # return int(torch.tensor(random.randrange(self.num_actions), \
            #     dtype=torch.long).view(1))

    def check_terminal(self, dh, state_rep):
        if len(dh) == (int(os.getenv('MAX_CONV_LEN')) + 1):
            return True
        return int(state_rep.view(-1)[3 + (4 * len(INTENTS)) + 4]) == 1

    def update_model(self, dialogue_history, state, action, next_state, reward):
        if action == 2:
            action = 0
        elif action == 4:
            action = 1
        elif action == 6:
            action = 2
        elif action == 7:
            action = 3
        
        if self.is_test == 2:
            return
        is_terminal = self.check_terminal(dialogue_history, next_state)
        #print('Terminal: %s, Length: %d, Action: %d' % (is_terminal, len(dialogue_history), action))
        with torch.set_grad_enabled(not self.is_test):
            empty_next = False
            if os.getenv('FEATURE_TRANSFORM') == 'fft':
                state = torch.cat(tuple(*torch.rfft(state, len(state.shape), normalized=True))).view(1, -1)
                #print('The shape of input state: %s' % str(state.shape))
                next_state = torch.cat(tuple(*torch.rfft(next_state, len(next_state.shape), normalized=True))).view(1, -1)
                
            reward = torch.tensor([reward], device=self.device, dtype=torch.float)
            action = torch.tensor(action, dtype=torch.long).view(1, 1)

            self.memory.push(state, action, next_state, reward, is_terminal)

            if len(self.memory) < self.BATCH_SIZE:
                return
            
            transitions = self.memory.sample(self.BATCH_SIZE)
            batch = Transition(*zip(*transitions))

            #non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
            #                                batch.next_state)), device=self.device, dtype=torch.bool)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not True,
                                            batch.is_terminal)), device=self.device, dtype=torch.bool)
            #non_final_next_states = torch.cat([s for s in batch.next_state
            #                                        if s is not None]).to(self.device)
            if not (self.is_test and batch.is_terminal[0]):
                #print(self.is_test[0], batch.is_terminal)
                non_final_next_states = torch.cat([s for i, s in enumerate(batch.next_state)
                                                        if batch.is_terminal[i] is not True]).to(self.device)

            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            all_values = self.policy_net(state_batch)
            state_action_values = all_values.gather(1, action_batch)

            next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
            if not (self.is_test and batch.is_terminal[0]):
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].to(self.device).detach()

            expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

            # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
            loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
            
            self.loss_track.append(float(loss.data))
            if os.getenv('MODE') == 'LR':
                self.train_learn_rate.append(float(self.optimizer.param_groups[0]['lr']))

            self.temp_reward += float(reward.view(1).data[0])

            # if abs(reward.view(1).data) > 0.1:
            if action.view(1).data[0] == 1 and reward.view(1).data[0] >= 0.3:
                # print('Positive Result')
                self.successes[0] += 1
                self.reward_track.append(self.temp_reward)
                self.temp_reward = 0.0
            elif action.view(1).data[0] == 1:
                # print('Negative Result', reward.view(1).data, action)
                self.successes[1] += 1
                self.reward_track.append(self.temp_reward)
                self.temp_reward = 0.0
            elif len(dialogue_history) == 2 and self.temp_reward != float(reward.view(1).data[0]):
                temp = float(reward.view(1).data[0])
                self.successes[1] += 1
                self.reward_track.append(self.temp_reward - temp)
                self.temp_reward = temp
            
            # print('(((((((((((((((((())))))))))))))))))')
            # print('This is the reward:', reward.view(1).data[0], 'IsTest:', self.is_test, \
            #         'Action Choice:', int(action.data[0]), 'Total Reward till now:', self.temp_reward)
            # print('(((((((((((((((((())))))))))))))))))')

            #signifies that a simulation is over
            
            if not self.is_test:
                # print('BACKPROPAGATING WEIGHTS!')
                self.optimizer.zero_grad()
                
                loss.backward()
                for param in self.policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()

                if os.getenv('MODE') == 'LR':
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] + (8.3325*10**(-9))
                else:
                    # for param_group in self.optimizer.param_groups:
                    #     param_group['lr'] = param_group['lr'] - (2.72**(-9))
                    pass
                    # self.lr_scheduler.step()

                if self.steps_done % self.target_trail == 0:
                    new_state_dict = dict()
                    for key in self.target_net.state_dict():
                        new_state_dict[key] = (self.target_net.state_dict()[key].clone().detach() * \
                            (1 - float(os.getenv('ALPHA')))) + (self.policy_net.state_dict()[key].clone().detach() * \
                            float(os.getenv('ALPHA')))
                    # new_state_dict = (self.target_net.state_dict() * (1 - float(os.getenv('ALPHA')))) + \
                        # (self.policy_net.state_dict() * float(os.getenv('ALPHA')))
                    self.target_net.load_state_dict(new_state_dict)
                    self.target_net.eval()
                    self.policy_net.train()
    
    def is_simulation(self):
        return False
    
    def set_test_phase(self, flag, change=True):
        if not flag and change:
            self.lr_scheduler.step()
        self.is_test = flag
        if len(self.loss_track) > 0:
            print('Average Loss: ', sum(self.loss_track)/len(self.loss_track))
        #self.memory.empty()
        if flag:
            #putting in test phase so coming from train phase
            #self.memory.capacity = 1
            self.BATCH_SIZE = 1
            self.memory = self.test_memory
            # print('Length of Train reward track: ', len(self.reward_track))
            self.train_loss_track.append(self.loss_track)
            self.train_reward_track.append(self.reward_track)
            self.train_successes.append(self.successes)
        else:
            #putting in train phase so coming from test phase
            #self.memory.capacity = int(os.getenv('REPLAY_CAPACITY'))
            self.memory = self.train_memory
            self.BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
            # print('Length of Test reward track: ', len(self.reward_track))
            self.test_loss_track.append(self.loss_track)
            self.test_reward_track.append(self.reward_track)
            self.test_successes.append(self.successes)
            self.test_action_choices.append(self.action_choices)
        
        self.loss_track = []
        self.reward_track = []
        self.successes = [0, 0]
        self.action_choices = []

    
    def save_model(self):
        model_file = os.getenv('MODELF') + '/trained_model' + os.getenv('MATCHER_STRAT') + \
            os.getenv('SIMULATOR') + os.getenv('COLOR_MODEL') + \
            os.getenv('SIMU_IMPREC_THRESH') + os.getenv('EVAL_MODE') + '.pt'
        torch.save(self.policy_net.state_dict(), model_file)
        #save the reward logs
        names = ['train_loss_track', 'train_reward_track', 'train_successes', 'test_loss_track', \
            'test_reward_track', 'test_successes', 'test_action_choices', 'train_learn_rate']
        for name in names:
            file_name = os.path.dirname(__file__) + '/training_parameters/' + name + \
                os.getenv('MATCHER_STRAT') + os.getenv('SIMULATOR') + \
                os.getenv('COLOR_MODEL') + os.getenv('SIMU_IMPREC_THRESH') + \
                os.getenv('EVAL_MODE') + '.pkl'
            pickle.dump(eval('self.' + name), open(file_name, 'wb'))
    
    def load_model(self):
        path = os.getenv('MODELF')
        if os.path.exists(path):
            self.policy_net.load_state_dict(torch.load(path))
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def plot_statistics(self):
        print('STARTING PLOTTING')
        avg_interval=int(os.getenv('AVG_ITERATIONS'))

        #plot everything here
        #plot the train loss
        cur_loss_track = []
        for ele in self.train_loss_track:
            cur_loss_track += ele
        # [cur_loss_track += ele for ele in self.train_loss_track]
        fig1, axes = plt.subplots()
        fig1.suptitle('Training Loss')
        axes.set_xlabel('Number of Iterations')
        axes.set_ylabel('Average TD Loss Per ' + str(avg_interval) + ' Iterations') 

        count = 0
        if len(cur_loss_track) >= avg_interval:
            loss_plot = np.zeros(len(cur_loss_track) - avg_interval + 1)
            for i in range(0, len(cur_loss_track) - avg_interval + 1):
                loss_plot[count] = sum(cur_loss_track[i : i + avg_interval])/avg_interval
                count += 1
            axes.plot(loss_plot, '*')
        
        #plot the test loss
        cur_loss_track = []
        for ele in self.test_loss_track:
            cur_loss_track += ele
        fig2, axes2 = plt.subplots()
        fig2.suptitle('Testing Loss')
        axes2.set_xlabel('Number of Iterations')
        axes2.set_ylabel('Average TD Loss Per ' + str(avg_interval) + ' Iterations') 
        count = 0
        if len(cur_loss_track) >= avg_interval:
            loss_plot = np.zeros(len(cur_loss_track) - avg_interval + 1)
            for i in range(0, len(cur_loss_track) - avg_interval + 1):
                loss_plot[count] = sum(cur_loss_track[i : i + avg_interval])/avg_interval
                count += 1
            axes2.plot(loss_plot, '*')

        #Action Plots
        # fig3, axes3 = plt.subplots(projection='3d')
        fig3 = plt.figure()
        axes3 = fig3.add_subplot(projection='3d')
        fig3.suptitle('Action Decisions During Testing Phase')
        axes3.set_xlabel('Probability of Patch 1 Being the Referent')
        axes3.set_ylabel('Probability of Patch 2 Being the Referent')
        axes3.set_zlabel('Probability of Patch 3 Being the Referent')
        action_choices = self.test_action_choices[-1]
        x = []; y = []
        markers = ['o', '^', '*', 'x', '+', '+', 'v']
        labels = ['NONE', 'IDENITFY', 'CLARIFICATION', 'CONFIRMATION', 'SELECTION', 'REJECTION']
        used = set()
        for diff, act, pa, cl, inc in action_choices:
            if labels[act] in used:
                axes3.scatter(*diff, marker=markers[act])
            else:
                axes3.scatter(*diff, marker=markers[act], label=labels[act])
                used.add(labels[act])
        axes3.legend()
        #successes and failures
        # print('Train Completions:', self.train_successes)
        # print('Test Completions:', self.test_successes)

        fig4, axes4 = plt.subplots()
        fig4.suptitle('Task Success vs Failures durign Testing Phase')
        width = 0.3
        x_intervals = np.arange(len(self.test_successes))
        axes4.set_xlabel('Success Rate Over Test Data For 4 Epochs')
        axes4.set_ylabel('Number of Successes')
        axes4.bar(x_intervals - width/2, [x[0] for x in self.test_successes], \
            width, label='task success')
        axes4.bar(x_intervals + width/2, [x[1] for x in self.test_successes], \
            width, label='task failure')
        axes4.legend()

        #reward plots
        fig5, axes5 = plt.subplots()
        fig5.suptitle('Reward Change During Training Phase')
        axes5.set_xlabel('Number of Simulations')
        axes5.set_ylabel('Average Training Reward Over ' + str(avg_interval) + ' Simulations')
        for j in range(len(self.train_reward_track)):
            reward_track = self.train_reward_track[j]
            print('Reward Info:', sum(reward_track), len(reward_track), len(reward_track) - avg_interval)
            reward_plot = np.zeros(len(reward_track) - avg_interval + 1)
            count = 0
            for i in range(len(reward_plot)):
                reward_plot[i] = sum(reward_track[i : i + avg_interval])/avg_interval
            axes5.plot(reward_plot, label='Training Reward After ' + \
                str(j+1) + 'th Epoch')
        axes5.legend()

        fig6, axes6 = plt.subplots()
        fig6.suptitle('Reward Change During Testing Phase')
        axes6.set_xlabel('Number of Simulations')
        axes6.set_ylabel('Average Testing Reward Over ' + str(avg_interval) + ' Simulations')
        for j in range(len(self.test_reward_track)):
            reward_track = self.test_reward_track[j]
            print('Reward Info:', sum(reward_track), len(reward_track), len(reward_track) - avg_interval)
            reward_plot = np.zeros(len(reward_track) - avg_interval + 1)
            count = 0
            for i in range(len(reward_plot)):
                reward_plot[i] = sum(reward_track[i : i + avg_interval])/avg_interval
            axes6.plot(reward_plot, label='Testing Simulation Reward After ' + \
                str(j+1) + 'th Epoch')
        axes6.legend()

        plot_files = ['Training_Loss', 'Testing_Loss', 'Action_Choices', 'Test_Success', \
            'Training_Reward', 'Testing_Reward']
        for i in range(6):
            eval('fig' + str(i+1)).savefig(os.path.dirname(__file__) + '/plots/' + \
                plot_files[i] + os.getenv('MATCHER_STRAT') + os.getenv('SIMULATOR') + os.getenv('COLOR_MODEL') + \
                os.getenv('SIMU_IMPREC_THRESH') + '.png')
        #start with the train loss
        # cur_loss_track = []
        # [cur_loss_track += ele for ele in self.train_loss_track]
        # fig, axes = plt.subplots(len(self.train_loss_track))
        # fig.suptitle('Training Loss')
        # axes[-1].set_xlabel('Number of Iterations')
        # axes[-1].set_ylabel('Average TD Loss Per ' + str(avg_interval) + ' Iterations') 
        # for i, ax in enumerate(axes):
        #     count = 0
        #     cur_loss_track = self.train_loss_track[i]
        #     if len(cur_loss_track) >= avg_interval:
        #         loss_plot = np.zeros(len(cur_loss_track) - avg_interval + 1)
        #         for i in range(0, len(cur_loss_track) - avg_interval + 1):
        #             loss_plot[count] = sum(cur_loss_track[i : i + avg_interval])/avg_interval
        #             count += 1
        #         ax.plot(loss_plot, '*')

        #now with test loss
        # fig2, axes2 = plt.subplots(len(self.test_loss_track))
        # fig2.suptitle('Testing Loss')
        # axes2[-1].set_xlabel('Number of Iterations')
        # axes2[-1].set_ylabel('Average TD Loss Per ' + str(avg_interval) + ' Iterations') 
        # for i, ax in enumerate(axes2):
        #     count = 0
        #     cur_loss_track = self.test_loss_track[i]
        #     if len(cur_loss_track) >= avg_interval:
        #         loss_plot = np.zeros(len(cur_loss_track) - avg_interval + 1)
        #         for i in range(0, len(cur_loss_track) - avg_interval + 1):
        #             loss_plot[count] = sum(cur_loss_track[i : i + avg_interval])/avg_interval
        #             count += 1
        #         ax.plot(loss_plot, '*')
        
        # names = ['train_loss_track', 'train_reward_track', 'train_successes', 'test_loss_track', \
        #     'test_reward_track', 'test_successes', 'test_action_choices']
        # for name in names:
        #     file_name = os.path.dirname(__file__) + '/training_parameters/' + name + \
        #         os.getenv('SIMULATOR') + os.getenv('COLOR_MODEL') + os.getenv('SIMU_IMPREC_THRESH') + '.pkl'
        #     pickle.dump(eval('self.' + name), open(file_name, 'wb'))
        
        plt.show()

