import copy
import time

import utils
from env_wrappers import *
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import hashlib
import pickle
from rl_algorithms import TD
from schedules import LinearSchedule
from buffers import BatchBuffer,Graph_buffer

class DQN_Agent():
    def __init__(self,env,net,args_dict):
        self.game_env = env
        self.args_dict = args_dict
        self.action_space = self.game_env.action_space
        self.action_space_set = set(range(self.action_space))
        self.state_space = self.game_env.observation_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        self.net = net(self.action_space,self.state_space,args_dict['atari_name']).to(self.device)
        self.target_net = net(self.action_space,self.state_space,args_dict['atari_name']).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())

        self.exploration_decay = LinearSchedule(schedule_timesteps=args_dict['decay_step'],final_p=args_dict['exploration_final_eps'],initial_p=args_dict['exploration_initial_eps'])
        self.lr_decay = LinearSchedule(schedule_timesteps=args_dict['final_step'], final_p=0.)
        self.update = TD(self.net,self.target_net, self.lr_decay,self.device,args_dict)

        self.graph_buffer = Graph_buffer(args_dict,action_space=self.action_space) # CEER
        self.batch_buffer = BatchBuffer(args_dict) # DQN
        self.current_episode = [[] for _ in range(args_dict['number_env'])]

        self.max_q_mean = 0
        self.all_q_mean = 0
        self.density = 0

    def save_model(self,path):
        torch.save(self.net.state_dict(), path)
        # torch.save(self.net.node_dict(),path,_use_new_zipfile_serialization=False)

    def load_model(self,path):
        self.net.load_state_dict(torch.load(path))

    def act(self,states,rewards,dones,infos,train,current_step):
        # print([(s.dtype,s.shape) for s in states])
        states_tensor = torch.from_numpy(np.array(states)).to(self.device).float()
        # print(states)
        # print(states_tensor.shape)
        with torch.no_grad():
            q_values = self.net(states_tensor)
        q_values = q_values.detach().cpu().numpy()
        # print('q_values :',q_values.shape)
        actions = []
        if train:
            epsilon = self.exploration_decay.value(current_step)
            exploration_list = np.random.random(self.args_dict['number_env']) < epsilon
            for i in range(self.args_dict['number_env']):
                # print('number :', i)
                # print(args.number_env, q_values.shape, q_values[i], states_tensor.shape)
                if exploration_list[i]:
                    actions.append(np.random.randint(self.action_space))
                else:
                    actions.append(np.argmax(q_values[i]))

            self.train(states,actions,rewards,dones,infos,current_step)
        else:
            exploration_list = np.random.random(self.args_dict.number_env) < 0.01  # 0.05
            for i in range(self.args_dict['number_env']):
                if exploration_list[i]:
                    actions.append(np.random.randint(self.action_space))
                else:
                    actions.append(np.argmax(q_values[i]))
                # actions = np.argmax(q_values,axis=1)
        # print(q_values)
        return actions

    def train(self,states,actions,rewards,dones,infos,current_step):
        if self.args_dict.sample_method != 'uniform':
            if rewards is None:
                self.s_t = states
                self.a_t = actions
            else:
                s_t_key_list = []
                for i in range(self.args_dict['number_env']):
                    if dones[i]:
                        s_t_key = hashlib.md5(pickle.dumps(self.s_t[i])).hexdigest() + str(False)
                        s_t1_key = hashlib.md5(pickle.dumps(infos[i]['terminal_state'])).hexdigest()+str(True)
                        self.graph_buffer.add_data(self.s_t[i], self.a_t[i], rewards[i],dones[i],
                                                   infos[i]['terminal_state'],s_t_key,s_t1_key)
                        self.current_episode[i].reverse()
                        self.graph_buffer.update_node(self.args_dict.batch_size,self.current_episode[i])
                        self.current_episode[i] = []
                    else:
                        s_t_key = hashlib.md5(pickle.dumps(self.s_t[i])).hexdigest()+str(False)
                        s_t1_key = hashlib.md5(pickle.dumps(states[i])).hexdigest()+str(False)
                        self.graph_buffer.add_data(self.s_t[i], self.a_t[i], rewards[i],dones[i],states[i],
                                                   s_t_key,s_t1_key)
                        self.current_episode[i].append(s_t_key)
                    s_t_key_list.append(s_t_key)

                self.s_t = states
                self.a_t = actions
        else:
            if rewards is None and dones is None:
                for i in range(self.batch_buffer.buffer_num):
                    self.batch_buffer.buffer_list[i].add_data(state_t=states[i],action_t=actions[i])
            else:
                for i in range(self.batch_buffer.buffer_num):
                    self.batch_buffer.buffer_list[i].add_data(
                        state_t=states[i],
                        action_t=actions[i],
                        reward_t=rewards[i],
                        terminal_t=dones[i])

        if current_step % self.args_dict['target_update_interval'] == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        if current_step >= self.args_dict['learning_starts']:
            # print(np.shape(self.batch_buffer.buffer_list))
            if self.args_dict['sample_method'] != 'uniform':
                for _ in range(self.args_dict['batch_num']):
                    s_t, a_t, r_t, t_t, s_t1, target_q_t, updated_t1,\
                    all_target_q_t,not_exist_action_value = self.graph_buffer.sample_batch(self.args_dict.batch_size)

                    s_t, one_hot_a_t, index, r_t, t_t, s_t1 = self.update.np2torch(
                        self.args_dict.batch_size, self.action_space, s_t, a_t, r_t, t_t, s_t1)

                    max_q_mean,all_q_mean,density = self.update.learn(self.args_dict.sample_method,
                                                                      self.graph_buffer,self.args_dict.batch_size,self.action_space,
                                                                      s_t, one_hot_a_t, r_t, t_t, s_t1,target_q_t,updated_t1,
                                                                      all_target_q_t,not_exist_action_value,self.args_dict.policy_loss_para)

                self.max_q_mean = max_q_mean
                self.all_q_mean = all_q_mean
                self.density = density
            else:
                for _ in range(self.args_dict.batch_num):
                    n = int(self.args_dict.batch_size / self.args_dict.number_env)
                    s_t, a_t, r_t, t_t, s_t1 = self.batch_buffer.sample_batch(current_step,n)
                    # print('state:',s_t)
                    s_t,one_hot_a_t,index,r_t,t_t,s_t1 = self.update.np2torch(
                        self.args_dict.batch_size,self.action_space,s_t, a_t, r_t, t_t, s_t1)
                    self.update.learn(self.args_dict.sample_method,None,self.args_dict.batch_size,self.action_space, s_t,one_hot_a_t, r_t, t_t, s_t1)
                    # print('data shape:', s_t.shape, a_t.shape, ret.shape, v.shape,logp.shape, adv.shape)
                    # print('data type:', s.dtype, a.dtype, ret.dtype, v.dtype,logp.dtype, adv.dtype)
