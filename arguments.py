# import pynvml
# pynvml.nvmlInit()

# gpu_num = pynvml.nvmlDeviceGetCount()
# if gpu_num:
#     CUDA_VISIBLE_DEVICES = gpu_num - 1 # use the last gpu
    # print('CUDA_VISIBLE_DEVICES:', CUDA_VISIBLE_DEVICES)
CUDA_VISIBLE_DEVICES=0
number_env = 8

para_list_dict = {
    'atari_name_list': ['Sokoban-Push_5x5_1_120'],
    # 'atari_name_list': ['MinAtar/Asterix-v0','MinAtar/Breakout-v0','MinAtar/Freeway-v0','MinAtar/Seaquest-v0','MinAtar/SpaceInvaders-v0',
    #                     'MiniGrid-DoorKey-6x6-v0','MiniGrid-Unlock-v0','MiniGrid-RedBlueDoors-6x6-v0','MiniGrid-SimpleCrossingS9N1-v0',
    #                     'MiniGrid-SimpleCrossingS9N2-v0','MiniGrid-LavaCrossingS9N1-v0','MiniGrid-LavaCrossingS9N2-v0',
    #                     'Sokoban-Push_5x5_1_120','Sokoban-Push_6x6_1_120','Sokoban-Push_7x7_1_120','Sokoban-Push_6x6_3_120',
    #                     'Sokoban-Push_5x5_2_120','Sokoban-Push_6x6_2_120','Sokoban-Push_7x7_2_120'],
    'network_type_list': ['large'],  # 'larger','large','medium','small','mlp'
    'seed_list': [0], #  list(range(21))
    'exploration_final_eps_list': [0.01],  # 0.1
    'batch_size_list': [32],  # 32, 64, 128,256,512
    'batch_num_list': [2], # replay ratio, 0.25, int(number_env*0.25)
    'double_dqn_list': [False], # True, False
    'update_time_list': [1], # number of updates for each batch
    'sample_method_list': ['kl'], # uniform, kl
    'sample_method_para_list':[0.], # [0.,0.2,0.5,0.8,1.]
    'policy_loss_list':[True], # True, False
    'policy_loss_para_list':[0.], # [0.,0.01,0.1,1.,2.,5.]
    'tau_list': [1.],#  temperature for softmax 1.,0.1,0.01
                  }

final_step = 2e6 # 2e6
learning_rate = 1e-4 # 3e-3,1e-3,3e-4,1e-4,3e-5,1e-5
buffer_size = int(final_step/20) # int(5e4) # int(final_step/20) # int(1e5) # 1_000_000,1e6,1e5 80 int(1e5/number_env)
learning_starts = final_step*0.005 # 100 10000
gamma = 0.99
target_update_interval= 1000
decay_step = final_step/2 # fraction of entire training period over which the exploration rate is reduced
exploration_initial_eps = 1.0
max_grad_norm = 10.
test_num = 100

FullyObs_minigrid = True
deterministic = False
fix_difficulty = False