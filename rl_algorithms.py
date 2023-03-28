import torch
import torch.nn.functional as F
import numpy as np

class TD():
    def __init__(self, net,target_net, decay,device,args_dict):
        self.net = net
        self.target_net = target_net
        self.decay = decay
        self.device = device
        # parameters
        self.args_dict = args_dict
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args_dict.learning_rate)
        # print('self.net.parameters():',self.net.parameters())

    def np2torch(self,batch_size,action_space,s_t, a_t, r_t, t_t, s_t1):
        s_t = torch.from_numpy(s_t).to(self.device).float()
        one_hot_a_t = torch.zeros(batch_size,action_space).to(self.device)
        index = torch.LongTensor(a_t).view(-1,1).to(self.device)
        one_hot_a_t.scatter_(dim=1,index=index,value=1.).to(self.device)
        r_t = torch.from_numpy(r_t).to(self.device)
        t_t = torch.from_numpy(t_t).to(self.device)
        s_t1 = torch.from_numpy(s_t1).to(self.device).float()

        return s_t,one_hot_a_t,index,r_t,t_t,s_t1

    def update(self,q_values,target_q,soft_q_loss = None,value_loss=True,policy_loss = False):
        # print('!!!',soft_q_loss)
        loss = torch.tensor(0.,requires_grad=True).to(self.device).float()
        if value_loss:
            loss = torch.nn.functional.smooth_l1_loss(q_values, target_q)
        if policy_loss:
            loss = loss + soft_q_loss
        # print('loss:',loss.cpu().item())
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.net.parameters(), args.max_grad_norm)
        self.optimizer.step()

    def compute_target(self,batch_size,action_space,s_t, r_t, t_t, s_t1,one_hot_a_t):
        all_q_values = self.net(s_t)  # shape [batch_size, action_space]
        q_values = (all_q_values * one_hot_a_t).sum(dim=1)
        if self.args_dict.double_dqn:
            with torch.no_grad():
                one_hot_max_actions = torch.zeros(batch_size, action_space).to(self.device)
                max_actions_index = self.net(s_t1).argmax(dim=1).view(-1, 1)
                one_hot_max_actions.scatter_(dim=1, index=max_actions_index, value=1.).to(self.device)

                next_q_values = self.target_net(s_t1)
                next_q_values = (next_q_values*one_hot_max_actions).sum(dim=1)
                target_q = r_t + (1.-t_t)*self.args_dict.gamma * next_q_values
        else:
            with torch.no_grad():
                next_q = self.target_net(s_t1)
                next_q, indices = next_q.max(dim=1)
                target_q = r_t + (1.-t_t)*self.args_dict.gamma * next_q
        return all_q_values, q_values, target_q

    def learn(self,sample_method,graph_buffer,batch_size,action_space, s_t,one_hot_a_t,r_t, t_t, s_t1,
              target_q_t=None,updated_t1=None,all_target_q_t=None,
              not_exist_action_value=None,policy_loss_para=None):

        if sample_method == 'kl':
            real_target_q = torch.from_numpy(target_q_t).to(self.device).float()
            updated_t1 = torch.from_numpy(updated_t1).to(self.device)
            all_target_q_t = torch.from_numpy(all_target_q_t).to(self.device)
            not_exist_action_value = torch.from_numpy(not_exist_action_value).to(self.device)

            all_q_values = self.net(s_t)  # shape [batch_size, action_space]. flatten and not flatten work both
            with torch.no_grad():
                next_q = self.target_net(s_t1)
                next_q, indices = next_q.max(dim=1)
                bootstrap_target_q = r_t + (1-t_t)*self.args_dict.gamma * next_q
                updated_real_target_q = bootstrap_target_q*(1-updated_t1) + real_target_q*updated_t1
                target_q = updated_real_target_q * self.args_dict.sample_method_para + bootstrap_target_q * (1 - self.args_dict.sample_method_para)
            # print(target_q)
            q_values = (all_q_values * one_hot_a_t).sum(dim=1)
            masked_q = all_q_values + not_exist_action_value
            # print(masked_q)
            log_soft_q = F.log_softmax(input=masked_q, dim=1)
            buffer_policy = F.softmax(all_target_q_t/self.args_dict.tau,dim=1)
            soft_q_loss = - torch.mean(torch.sum(buffer_policy * log_soft_q,dim=1)) * policy_loss_para
            self.update(q_values, target_q,soft_q_loss,value_loss=True,policy_loss=True)
            # self.update(q_values, target_q, q_loss)
            all_q_values_np = all_q_values.detach().cpu().numpy()
            max_q_mean = np.mean(np.max(all_q_values_np,axis=1))
            all_q_mean = np.mean(all_q_values_np)
            # density = graph_buffer.total_edges / len(graph_buffer.s_key)
            density = graph_buffer.total_edges / (len(graph_buffer.s_key) - len(graph_buffer.terminal_s_key))
            return max_q_mean,all_q_mean,density
        else:
            # print('state:', s_t)
            all_q_values, q_values, target_q = self.compute_target(batch_size,action_space,s_t, r_t, t_t, s_t1,one_hot_a_t)
            self.update(q_values = q_values,target_q = target_q)
