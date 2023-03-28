import numpy as np
from collections import deque,OrderedDict,defaultdict
import copy
import sys
import utils
import random

class TreeNode(object):
    def __init__(self, parent,state,state_key):
        self.parent = parent # {}
        self.state = state # array or feature vector
        self.s_key = state_key # s_key is combination of state and terminal, str(state)+str(terminal)

        self.node_visited_time = 0
        self.edges = {}  # a map from action to TreeNode, key:edge, value: node
        self.edges_info = {} # key: edge, value: [a,r,t,visited_time]
        self.q = defaultdict(float) # key: a, value: q value
        self.a_visited_time = defaultdict(int) # key:a, value: visited time
        self.value = 0 # state value, is also q value for q learning
        self.value_updated_time = 0

    def expand(self,edge,node,a,r,t):
        # edge is combination of action, reward and terminal, str(action)+str(reward)+str(terminal)
        self.edges[edge] = node
        self.edges_info[edge] = [a,r,t,1]

    def print_info(self):
        print('edges:',self.node_visited_time,self.edges.keys())
        print('edges_info:',self.edges_info)
        print('q:',self.q,self.a_visited_time,self.value,self.value_updated_time)


class Graph_buffer():
    # for CEER
    def __init__(self,args_dict, action_space):
        self.buffer_size = args_dict.buffer_size
        self.current_buffer_length = 0
        self.gamma = args_dict.gamma
        self.tau = args_dict.tau
        self.action_space = action_space
        self.s_key = OrderedDict() # state key, save s_key as set, search complexity is O(1)
        self.s_key_without_terminal_s_list = [] # state key
        self.terminal_s_key = set()
        self.node_dict = {}
        self.s_key_list_for_uniform_sample = deque(maxlen=self.buffer_size) # for uniform sample
        self.total_value_updated_time = 0 # should initialized as 0, but as 1 in case zero division error
        self.total_edges = 0

    def add_data(self,s_t,a_t,r_t,t_t,s_t_1,s_t_key,s_t_1_key): # todo: multiple children
        edge = s_t_key + '_' + str(a_t) + '_' + str(r_t) + '_' + str(t_t) + '_' + s_t_1_key
        # todo: what if s_t == s_t_1?
        if s_t_key not in self.s_key and s_t_1_key not in self.s_key:
            # print('none!')
            # node
            self.node_dict[s_t_key] = TreeNode(parent={},state=s_t,state_key=s_t_key)
            self.node_dict[s_t_1_key] = TreeNode(parent={}, state=s_t_1,state_key=s_t_1_key)
            # edge
            self.node_dict[s_t_key].expand(edge,self.node_dict[s_t_1_key],a_t,r_t,t_t)
            self.node_dict[s_t_1_key].parent[edge] = self.node_dict[s_t_key]
            # record s key
            self.s_key_without_terminal_s_list.append(s_t_key) # s_t_key must not be terminal state
            if s_t_key == s_t_1_key:
                assert self.node_dict[s_t_key] is self.node_dict[s_t_1_key]
                if self.current_buffer_length < self.buffer_size:
                    tree_idx_s_t = self.current_buffer_length + self.buffer_size - 1
                    self.s_key[s_t_key] = tree_idx_s_t
                    self.current_buffer_length += 1
                else:
                    del_k, del_v = self.s_key.popitem(last=False)  # k:s_key, v: tree index
                    self.del_node(del_k)
                    self.s_key[s_t_key] = del_v
            else:
                if self.current_buffer_length < self.buffer_size:
                    # add s_t
                    tree_idx_s_t = self.current_buffer_length + self.buffer_size - 1
                    self.s_key[s_t_key] = tree_idx_s_t
                    self.current_buffer_length += 1
                    # add s_t_1
                    if self.current_buffer_length < self.buffer_size:
                        tree_idx_s_t_1 = self.current_buffer_length + self.buffer_size - 1
                        self.s_key[s_t_1_key] = tree_idx_s_t_1
                        self.current_buffer_length += 1
                    else:
                        del_k, del_v = self.s_key.popitem(last=False)  # k:s_key, v: tree index
                        self.del_node(del_k)
                        # print('+++', del_k, del_v)
                        self.s_key[s_t_1_key] = del_v
                else:
                    # add s_t
                    del_k, del_v = self.s_key.popitem(last=False)  # k:s_key, v: tree index
                    self.del_node(del_k)
                    # print('xxx', del_k, del_v)
                    self.s_key[s_t_key] = del_v
                    # add s_t_1
                    del_k, del_v = self.s_key.popitem(last=False)  # k:s_key, v: tree index
                    self.del_node(del_k)
                    # print('---', del_k, del_v)
                    self.s_key[s_t_1_key] = del_v

            if not t_t and s_t_1_key != s_t_key:
                self.s_key_without_terminal_s_list.append(s_t_1_key)

            assert self.node_dict[s_t_key] is self.node_dict[s_t_1_key].parent[edge]
            assert self.node_dict[s_t_key].edges[edge] is self.node_dict[s_t_1_key]

        elif s_t_key in self.s_key and s_t_1_key not in self.s_key:
            # print('s_t in!')
            # node
            self.node_dict[s_t_1_key] = TreeNode(parent={}, state=s_t_1,state_key=s_t_1_key)
            #edge
            self.node_dict[s_t_key].expand(edge,self.node_dict[s_t_1_key],a_t,r_t,t_t)
            self.node_dict[s_t_1_key].parent[edge] = self.node_dict[s_t_key]
            # keep state order
            self.s_key.move_to_end(s_t_key)
            # add to sum tree
            if self.current_buffer_length < self.buffer_size:
                tree_idx_s_t_1 = self.current_buffer_length + self.buffer_size - 1
                self.s_key[s_t_1_key] = tree_idx_s_t_1
                self.current_buffer_length += 1
            else:
                del_k, del_v = self.s_key.popitem(last=False)  # k:s_key, v: tree index
                self.del_node(del_k)
                # print('***', del_k, del_v)
                self.s_key[s_t_1_key] = del_v
            # record s key, don't care the order
            if not t_t:
                self.s_key_without_terminal_s_list.append(s_t_1_key)

            assert self.node_dict[s_t_key] is self.node_dict[s_t_1_key].parent[edge]
            assert self.node_dict[s_t_key].edges[edge] is self.node_dict[s_t_1_key]

        elif s_t_key not in self.s_key and s_t_1_key in self.s_key:
            # print('s_t_1 in!')
            # node
            self.node_dict[s_t_key] = TreeNode({},state=s_t,state_key=s_t_key)
            # edge
            self.node_dict[s_t_key].expand(edge,self.node_dict[s_t_1_key],a_t,r_t,t_t)
            assert edge not in self.node_dict[s_t_1_key].parent
            self.node_dict[s_t_1_key].parent[edge] = self.node_dict[s_t_key]
            # keep state order
            self.s_key.move_to_end(s_t_1_key)
            # add to sum tree
            if self.current_buffer_length < self.buffer_size:
                tree_idx_s_t = self.current_buffer_length + self.buffer_size - 1
                self.s_key[s_t_key] = tree_idx_s_t
                self.current_buffer_length += 1
            else:
                del_k, del_v = self.s_key.popitem(last=False)  # k:s_key, v: tree index
                self.del_node(del_k)
                # print('@@@', del_k, del_v)
                self.s_key[s_t_key] = del_v
            # record s key, don't care the order
            self.s_key_without_terminal_s_list.append(s_t_key)

            assert self.node_dict[s_t_key] is self.node_dict[s_t_1_key].parent[edge]
            assert self.node_dict[s_t_key].edges[edge] is self.node_dict[s_t_1_key]

        else:
            # print('both in!')
            # edge
            if edge in self.node_dict[s_t_key].edges:
                assert edge in self.node_dict[s_t_1_key].parent
                assert self.node_dict[s_t_key] is self.node_dict[s_t_1_key].parent[edge],\
                    [edge,self.node_dict[s_t_1_key].parent,np.swapaxes(self.node_dict[s_t_key].state,0,2)[0],
                     np.swapaxes(self.node_dict[s_t_1_key].state, 0, 2)[0],
                     np.swapaxes(self.node_dict[s_t_1_key].parent[edge].state,0,2)[0]]
                assert self.node_dict[s_t_key].edges[edge] is self.node_dict[s_t_1_key]
                self.node_dict[s_t_key].edges_info[edge][3] += 1 # update edge visited time

            else:
                self.node_dict[s_t_key].expand(edge,self.node_dict[s_t_1_key],a_t,r_t,t_t)
                self.node_dict[s_t_1_key].parent[edge] =self.node_dict[s_t_key]

                if s_t_key == s_t_1_key:
                    assert self.node_dict[s_t_key] is self.node_dict[s_t_1_key]

                assert self.node_dict[s_t_key] is self.node_dict[s_t_1_key].parent[edge]
                assert self.node_dict[s_t_key].edges[edge] is self.node_dict[s_t_1_key]
            # keep state order
            self.s_key.move_to_end(s_t_key)
            self.s_key.move_to_end(s_t_1_key)

        # update state visited time
        self.node_dict[s_t_key].node_visited_time += 1
        assert len(self.node_dict) == len(self.s_key)
        assert edge in self.node_dict[s_t_1_key].parent and edge in self.node_dict[s_t_1_key].parent[edge].edges, \
            [edge, self.node_dict[s_t_key].edges, self.node_dict[s_t_1_key].parent[edge].edges]

        if t_t:
            self.terminal_s_key.add(s_t_1_key)
        assert len(self.terminal_s_key) + len(self.s_key_without_terminal_s_list) == len(self.s_key),[len(self.terminal_s_key),len(self.s_key_without_terminal_s_list),len(self.s_key)]

        self.s_key_list_for_uniform_sample.append([s_t_key,edge])

    def update_node(self,n,current_s_t_key_list = None): # todo: set a flag to reduce computation, if no value are changed, we don't need to update so frequently
        if current_s_t_key_list:
            len_current_s_t_key_list = len(current_s_t_key_list)
            if n > len_current_s_t_key_list:
                index_list = np.random.randint(len(self.s_key_without_terminal_s_list), size=n - len_current_s_t_key_list)
                for index in index_list:
                    assert self.s_key_without_terminal_s_list[index] in self.s_key, index
                    assert self.s_key_without_terminal_s_list[index] in self.node_dict, index
                    self.up_node_(self.s_key_without_terminal_s_list[index])
            for s_key in current_s_t_key_list:
                # assert s_key in self.s_key, s_key
                # assert s_key in self.node_dict, s_key
                if s_key not in self.node_dict:
                    continue
                self.up_node_(s_key)
        else:
            index_list = np.random.randint(len(self.s_key_without_terminal_s_list), size=n)
            for index in index_list:
                assert self.s_key_without_terminal_s_list[index] in self.s_key, index
                assert self.s_key_without_terminal_s_list[index] in self.node_dict, index
                self.up_node_(self.s_key_without_terminal_s_list[index])

    def up_node_(self,s_key):
        if self.node_dict[s_key].edges:
            # update q
            # reset q
            old_edges_num = len(self.node_dict[s_key].q)
            self.total_edges -= old_edges_num
            self.node_dict[s_key].q = defaultdict(float)
            self.node_dict[s_key].a_visited_time = defaultdict(int)
            for e in self.node_dict[s_key].edges:
                # r + (1-t) * gamma * next_q
                # edges_info # key: edge, value: [a,r,t,visited_time]
                r = self.node_dict[s_key].edges_info[e][1] # r
                t = self.node_dict[s_key].edges_info[e][2] # t
                next_q = self.node_dict[s_key].edges[e].value # next q value
                visited_time = self.node_dict[s_key].edges_info[e][3] # visited time
                unnormalized_q = (r + (1-t) * self.gamma * next_q) * visited_time

                self.node_dict[s_key].q[self.node_dict[s_key].edges_info[e][0]] += unnormalized_q
                self.node_dict[s_key].a_visited_time[self.node_dict[s_key].edges_info[e][0]] += visited_time
            for a in self.node_dict[s_key].a_visited_time:
                self.node_dict[s_key].q[a] /= self.node_dict[s_key].a_visited_time[a]
            # update v
            # print(self.node_dict[s_key].q)
            q_max = max(self.node_dict[s_key].q.values())
            # print(q_max)
            if self.node_dict[s_key].value != q_max:
                self.node_dict[s_key].value = q_max
                self.node_dict[s_key].value_updated_time += 1
                self.total_value_updated_time += 1
                # self.changed_count += 1
            new_edges_num = len(self.node_dict[s_key].q)
            self.total_edges += new_edges_num
            assert self.total_edges >= 0
            # assert new_edges_num >= old_edges_num, [new_edges_num,old_edges_num]

    def del_node(self,state_key):
        self.total_edges -= len(self.node_dict[state_key].q)
        # del pointer from children
        for e in list(self.node_dict[state_key].edges.keys()):
            # print(self.node_dict[state_key].children[c].parent)
            # print('delete children pointer',c,index)
            # print('---')
            del self.node_dict[state_key].edges[e].parent[e]

        # del pointer from parent
        for e in list(self.node_dict[state_key].parent.keys()):
            # print(self.node_dict[state_key].parent[index].children)
            # print('delete parent pointer',index,c)
            # print('---')
            del self.node_dict[state_key].parent[e].edges[e]
        # substrct value updated time
        self.total_value_updated_time -= self.node_dict[state_key].value_updated_time
        assert self.total_value_updated_time >= 0
        # delete node
        del self.node_dict[state_key]
        if state_key in self.terminal_s_key:
            self.terminal_s_key.remove(state_key)
        else:
            self.s_key_without_terminal_s_list.remove(state_key)

    def get_edge(self,edges,len_edges,index):
        stat = [[0.]*len_edges for _ in range(6)]
        # each row means: 0:edge visited time, 1:a, 2:r, 3:t, 4:true target q, 5:value_updated_time
        for i in range(len_edges):
            # edges_info key: edge, value: [a,r,t,visited_time]
            stat[0][i] = self.node_dict[index].edges_info[edges[i]][3] # to compute edge probs
        total_visited_time = sum(stat[0])
        if total_visited_time == 0:
            return total_visited_time,stat
        else:
            self.up_node_(index)
        for i in range(len_edges):
            stat[1][i] = self.node_dict[index].edges_info[edges[i]][0]  # a
            stat[2][i] = self.node_dict[index].edges_info[edges[i]][1]  # r
            stat[3][i] = self.node_dict[index].edges_info[edges[i]][2]  # t
            stat[4][i] = self.node_dict[index].q[self.node_dict[index].edges_info[edges[i]][0]]  # true target q
            stat[5][i] = self.node_dict[index].value_updated_time  # up
        return total_visited_time,stat

    def sample_batch(self,n):
        s_t = []
        a_t = []
        r_t = []
        t_t = []
        s_t1 = []
        target_q_t = []
        updated_t1 = []
        all_target_q_t = []
        not_exist_action_value = []

        # index_list = self.get_s_key(n)
        length = len(self.s_key_list_for_uniform_sample) # [s_t_key,edge]
        s_key_index_list = np.random.randint(length,size=n)
        for i in s_key_index_list:
            index = self.s_key_list_for_uniform_sample[i][0] # s_t_key
            edges = list(self.node_dict[index].edges.keys())
            len_edges = len(edges)
            one_hot_index = [0] * self.action_space
            if edges:
                total_visited_time,stat = self.get_edge(edges,len_edges,index)
                e_index = edges.index(self.s_key_list_for_uniform_sample[i][-1])

                s_t.append(self.node_dict[index].state)
                a_t.append(stat[1][e_index])
                r_t.append(stat[2][e_index])
                t_t.append(stat[3][e_index])
                s_t1.append(self.node_dict[index].edges[edges[e_index]].state)
                target_q_t.append(stat[4][e_index])
                if stat[5][e_index] > 0:
                    updated_t1.append(1.)
                else:
                    updated_t1.append(0.)

                tmp_q = []
                tmp_not_exist_action_value = []
                for a in range(self.action_space):
                    if a in self.node_dict[index].q:
                        tmp_q.append(self.node_dict[index].q[a])
                        tmp_not_exist_action_value.append(0.)
                    else:
                        tmp_q.append(-np.inf)
                        tmp_not_exist_action_value.append(-np.inf) # give minimum value for not existing value
                all_target_q_t.append(tmp_q)
                not_exist_action_value.append(tmp_not_exist_action_value)

                all_q = list(self.node_dict[index].q.values()) # all q key: action, value: tabular q value
                all_max_q = round(max(all_q),10)
                all_q.remove(self.node_dict[index].q[stat[1][e_index]]) # remove current q | s,a,r,s'
                all_q.append(-np.inf) # in case only one action in q values

                for k in self.node_dict[index].q.keys():
                    if round(self.node_dict[index].q[k],10) == all_max_q:
                        one_hot_index[k] = 1.
            else:
                print('!!!',index)
                self.node_dict[index].print_info()
                print(self.node_dict[index].parent.keys())
                print([self.node_dict[index].parent[k].edges_info[k] for k in self.node_dict[index].parent.keys()])
                assert edges, edges
        all_target_q_t = np.array(all_target_q_t)
        not_exist_action_value = np.array(not_exist_action_value)

        assert len(a_t) == n
        # change True/False to 1,0 by +0.
        return np.array(s_t),np.array(a_t),np.array(r_t),np.array(t_t)+0.,np.array(s_t1),\
               np.array(target_q_t),np.array(updated_t1),\
               all_target_q_t,not_exist_action_value


class Buffer():
    # for DQN
    def __init__(self,buffer_size):
        self.initialize_buffer(buffer_size)

    def initialize_buffer(self,buffer_size):
        self.state_list = deque(maxlen=buffer_size+1)
        self.action_list = deque(maxlen=buffer_size+1)
        self.clone_state_list = deque(maxlen=buffer_size+1)
        self.reward_list = deque(maxlen=buffer_size)
        self.terminal_list = deque(maxlen=buffer_size)

    def add_data(self,state_t=None,action_t=None,reward_t=None,terminal_t=None,clone_state_t=None):
        if state_t is not None:
            self.state_list.append(state_t)
        if action_t is not None:
            self.action_list.append(action_t)
        if reward_t is not None:
            self.reward_list.append(reward_t)
        if terminal_t is not None:
            self.terminal_list.append(terminal_t)
        if clone_state_t is not None:
            self.clone_state_list.append(clone_state_t)

class BatchBuffer():
    def __init__(self,args_dict):
        self.args_dict = args_dict
        self.buffer_num = args_dict.number_env
        self.buffer_size = int(args_dict['buffer_size'] / args_dict['number_env'])
        self.buffer_list = [Buffer(self.buffer_size) for _ in range(self.buffer_num)]
        self.model_list = deque(maxlen=self.buffer_size+1)
        self.gamma = args_dict.gamma
        # print(self.model_list.maxlen,self.buffer_list[0].state_list.maxlen)

    def sample_batch(self,current_step,n):
        max_index = min(int(self.args_dict.buffer_size/self.args_dict.number_env), int(current_step / self.args_dict.number_env))
        index = np.random.randint(max_index,size = n)
        s_t = []
        a_t = []
        r_t = []
        t_t = []
        s_t1 = []
        for buffer in self.buffer_list:
            for i in index:
                # print('buffer.state_list[i] :',buffer.state_list[i].dtype)
                s_t.append(buffer.state_list[i])
                a_t.append(buffer.action_list[i])
                r_t.append(np.float32(buffer.reward_list[i]))
                t_t.append(np.float32(buffer.terminal_list[i]))
                s_t1.append(buffer.state_list[i+1])
        # print(cs_t[0].shape)
        # print(np.array(s_t).shape,np.array(a_t).shape,np.array(r_t).shape,np.array(t_t).shape,np.array(s_t1).shape,np.array(cs_t).shape)
        return np.array(s_t),np.array(a_t),np.array(r_t),np.array(t_t),np.array(s_t1)