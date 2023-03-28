import copy
import arguments as args
from agents import DQN_Agent
import pandas as pd
import torch
import random
import itertools
import time
import numpy as np
import utils
import sys
import os
import hashlib
import pickle
import multiprocessing
from multiprocessing import Pool
import optimal_lr
from networks import QNetwork as cnn


def run_atari(a_n, net_type, sd, exp_final_eps, b_size, b_num, ddqn, up_time,method, method_para, p_loss, p_loss_para,
              tau, train = True):
    args_dict = {'CUDA_VISIBLE_DEVICES': args.CUDA_VISIBLE_DEVICES,
                 'number_env': args.number_env, 'learning_rate': args.learning_rate, 'final_step': args.final_step,
                 'buffer_size': args.buffer_size, 'learning_starts': args.learning_starts, 'gamma': args.gamma,
                 'target_update_interval': args.target_update_interval, 'decay_step': args.decay_step,
                 'exploration_initial_eps': args.exploration_initial_eps, 'max_grad_norm': args.max_grad_norm,
                 'test_num': args.test_num, 'FullyObs_minigrid': args.FullyObs_minigrid,
                 'deterministic': args.deterministic,
                 'fix_difficulty': args.fix_difficulty,
                 'atari_name': a_n, 'network_type': net_type, 'seed': sd,
                 'exploration_final_eps': exp_final_eps, 'batch_size': b_size,
                 'batch_num': b_num, 'double_dqn': ddqn, 'update_time': up_time,
                 'sample_method': method,
                 'sample_method_para': method_para, 'policy_loss': p_loss,
                 'policy_loss_para': p_loss_para, 'tau':tau}
    args_dict = utils.dict_to_object(args_dict)
    args_dict.path = str(args_dict.double_dqn) + '_' + str(args_dict.FullyObs_minigrid) + '/' + args_dict.atari_name + '/' + args_dict.network_type + '_' + str(args_dict.seed) + '_' + str(args_dict.learning_rate)
    if args_dict.sample_method == 'kl':
        args_dict.sample_method_para = optimal_lr.optimal_para_kl_dict[args_dict.atari_name][0]
        args_dict.policy_loss_para = optimal_lr.optimal_para_kl_dict[args_dict.atari_name][1]
        args_dict.tau = optimal_lr.optimal_para_kl_dict[args_dict.atari_name][2]
    elif args_dict.sample_method == 'uniform':
        args_dict.learning_rate = optimal_lr.optimal_lr_dict['dqn'][args_dict.atari_name]
        args_dict.path = str(args_dict.double_dqn) + '_' + str(args_dict.FullyObs_minigrid) + '/' + args_dict.atari_name + '/' + args_dict.network_type + '_' + str(args_dict.seed) + '_' + str(args_dict.learning_rate)
        # print('learning_rate:',args_dict.learning_rate)
    print('Training ...')
    setup_seed(args_dict.seed)
    if args_dict.sample_method != 'uniform':
        args_dict.name_form = str(args_dict.batch_size) + '_' + str(args_dict.batch_num) + '_' + str(args_dict.update_time) + '_' + str(args_dict.exploration_final_eps).split('.')[-1] + '_' + args_dict.sample_method + '_' + str(args_dict.sample_method_para) + '_' + str(args_dict.policy_loss) + '_' + str(args_dict.policy_loss_para) + '_' + str(args_dict.tau)
    else:
        args_dict.name_form = str(args_dict['batch_size']) + '_' + str(args_dict['batch_num']) +'_' + str(args_dict['update_time']) + '_' + str(args_dict['exploration_final_eps']).split('.')[-1] + '_' + args_dict['sample_method']
    print('log path:',args_dict.path)
    print('file name:',args_dict.name_form)
    if 'MinAtar' in args_dict['atari_name']:
        from env_wrappers.minatar_wrappers import Baselines_DummyVecEnv
    elif 'Sokoban' in args_dict['atari_name']:
        from env_wrappers.sokoban_wrappers import Baselines_DummyVecEnv
    elif 'MiniGrid' in args_dict['atari_name']:
        from env_wrappers.minigrid_wrappers import Baselines_DummyVecEnv

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_dict['CUDA_VISIBLE_DEVICES'])
    # print('CUDA_VISIBLE_DEVICES:',args_dict['CUDA_VISIBLE_DEVICES'])
    batch_env = Baselines_DummyVecEnv(env_id=args_dict['atari_name'], num_env=args_dict['number_env'], array_obs=(args_dict['network_type'] != 'mlp'))
    agent = DQN_Agent(batch_env,cnn,args_dict)
    # print('agent initialized!')

    states = batch_env.reset()
    max_q_mean_list = []
    all_q_mean_list = []
    density_list = []
    time_use_list = []
    print('game:',args_dict['atari_name'],'| action space:',batch_env.action_space,
          '|obs space:',batch_env.observation_space,'|method:',args_dict['sample_method'])
    print('---')

    rewards, dones, info = None,None,None
    current_step = 0
    tstart = time.time()
    net_time = 0
    while current_step <= args_dict.final_step:
        net_tstart = time.time()
        # print(states)
        actions = agent.act(states,rewards, dones, info, train,current_step)
        # print(actions)
        net_time += time.time() - net_tstart
        next_states, rewards, dones, info = batch_env.step(actions)
        # print(rewards)
        states = next_states
        # print('states :', states[0].dtype)

        if current_step % int(args_dict['final_step']/20/2) == 0: # 200000 10000 50000
            tnow = time.time() + 0.001
            time_use_list.append(tnow - tstart)
            print('{}, seed: {}, sample_method_para: {}, policy_loss_para: {}, tau: {}, step: {:.2e}, reward (mean,max): {}, length: {}'.format(
                args_dict.atari_name, args_dict.seed, args_dict.sample_method_para, args_dict.policy_loss_para, args_dict.tau, current_step,[batch_env.get_episode_rewmean(),batch_env.get_episode_rewmax()],
                batch_env.get_episode_lenmean()))

        if current_step % int(args_dict['final_step']/20) == 0: # 1000000 100000
            model_path = 'model/' + args_dict['path']
            try:
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
            except:
                pass
            model_name = '/' + str(current_step) + '_' + args_dict['name_form'] + ".pth"
            agent.save_model(model_path + model_name)

            max_q_mean_list.append(agent.max_q_mean)
            all_q_mean_list.append(agent.all_q_mean)
            density_list.append(agent.density)

        current_step += batch_env.get_num_of_envs()
        # print('current_step :',current_step,actions,dones,agent.terminal_list,agent.action_list)

    log_path = 'log/' + args_dict['path']
    try:
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    except:
        pass

    np.savez_compressed(log_path + '/statistics_' + args_dict['name_form'],
                        max_q_mean = max_q_mean_list,
                        all_q_mean = all_q_mean_list,
                        density = density_list,
                        action_space = [batch_env.action_space]*21,
                        time_use = time_use_list)
    # this file is much smaller than np.save, f.write, pickle.dump

    print('Testing ... ')
    test_atari(args_dict)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def test_atari(args_dict):
    tempM = []  # temp. mean score
    # mean_score = []
    setup_seed(args_dict.seed*1000) # test using different seed
    tstart = time.time()
    path = args_dict.path
    name_form = args_dict.name_form
    if 'MinAtar' in args_dict.atari_name:
        from env_wrappers.minatar_wrappers import Baselines_DummyVecEnv
    elif 'Sokoban' in args_dict.atari_name:
        from env_wrappers.sokoban_wrappers import Baselines_DummyVecEnv
    elif 'MiniGrid' in args_dict.atari_name:
        from env_wrappers.minigrid_wrappers import Baselines_DummyVecEnv

    for index in range(0, int(args_dict.final_step) + 1, int(args_dict.final_step/20)):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args_dict.CUDA_VISIBLE_DEVICES)
        test_env = Baselines_DummyVecEnv(env_id=args_dict.atari_name, num_env=args_dict.number_env, array_obs=(args_dict.network_type != 'mlp'))

        test_states = test_env.reset()

        test_rewards, test_dones, test_info = None, None, None
        agent = DQN_Agent(test_env, cnn, args_dict)

        model_path = 'model/' + path
        model_name = '/' + str(index) + '_' + name_form + ".pth"
        agent.load_model(model_path + model_name)

        # print(model_path+model_name,'loaded!')

        while len(test_env.epinfobuf) < args_dict.test_num:
            actions = agent.act(test_states, test_rewards, test_dones, test_info, False,0)
            test_next_states, test_rewards, test_dones, test_info = test_env.step(actions)
            test_states = test_next_states
            # input("next action:")

        tempM.append(test_env.get_episode_rewmean())
    print(tempM,test_env.get_episode_lenmean())

    # mean_score.append(tempM)
    mean_score = {'steps':range(0, int(args_dict.final_step) + 1, int(args_dict.final_step / 20)),'mean_score':tempM}
    mean_score = pd.DataFrame(mean_score)
    # mean_score = pd.DataFrame(mean_score).melt(var_name='iteration', value_name='mean_score')

    log_path = 'log/' + path
    try:
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    except:
        pass
    mean_score.to_csv(log_path + "/mean_score_" + name_form + ".csv", index=False)

    print("Save mean_score successfully. Time: {:.2f}".format((time.time()-tstart)/60))
    print(log_path + "/mean_score_" + name_form + ".csv")

if __name__ == '__main__':
    total_start_time = time.time()
    # print(os.getcwd())
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--lr", type=float, default=1e-4)
    para = parser.parse_args()

    experiments = list(itertools.product(*args.para_list_dict.values()))
    # print(len(experiments))

    # for single process
    for a_n, net_type, sd, exp_final_eps, b_size, b_num, ddqn, up_time, \
        method, method_para, p_loss, p_loss_para, tau in experiments:
        run_atari(a_n, net_type, sd, exp_final_eps, b_size, b_num, ddqn, up_time,method, method_para, p_loss, p_loss_para, tau)

    # ctx = multiprocessing.get_context('spawn')
    # p = ctx.Pool(4)
    # for atari_name, network_type, seed, exploration_final_eps, batch_size, batch_num, double_dqn, update_time, \
    #     sample_method, sample_method_para, policy_loss, policy_loss_para,tau in experiments:
    #     p.apply_async(run_atari, args=(atari_name, network_type, seed, exploration_final_eps, batch_size, batch_num, double_dqn, update_time,
    #                                    sample_method, sample_method_para, policy_loss, policy_loss_para,tau,),error_callback=print_error)
    # p.close()
    # p.join()
    print('total time used: {:.2f}'.format((time.time()-total_start_time)/3600))
