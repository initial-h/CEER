
optimal_lr_dict = {
    'dqn':
        {'Sokoban-Push_5x5_1_120':1e-3, 'Sokoban-Push_5x5_2_120':1e-3, 'Sokoban-Push_6x6_1_120':1e-4,
         'Sokoban-Push_6x6_2_120':1e-4, 'Sokoban-Push_6x6_3_120':3e-4, 'Sokoban-Push_7x7_1_120':1e-4,
         'Sokoban-Push_7x7_2_120':1e-4,
         'MiniGrid-DoorKey-6x6-v0':3e-4, 'MiniGrid-Unlock-v0':3e-4,'MiniGrid-RedBlueDoors-6x6-v0':3e-4,
         'MiniGrid-SimpleCrossingS9N1-v0':1e-4,'MiniGrid-SimpleCrossingS9N2-v0':3e-4,
         'MiniGrid-LavaCrossingS9N1-v0':3e-4,'MiniGrid-LavaCrossingS9N2-v0':1e-4,
         'MinAtar/Asterix-v0':3e-5, 'MinAtar/Breakout-v0':3e-4,'MinAtar/Freeway-v0':3e-4,
         'MinAtar/Seaquest-v0':1e-4, 'MinAtar/SpaceInvaders-v0':3e-4,'cliffworld':1e-4}
    ,
    'DDQN-EBU':
        {'Sokoban-Push_5x5_1_120':3e-5, 'Sokoban-Push_5x5_2_120':1e-4, 'Sokoban-Push_6x6_1_120':1e-5,
         'Sokoban-Push_6x6_2_120':1e-5, 'Sokoban-Push_6x6_3_120':1e-5, 'Sokoban-Push_7x7_1_120':1e-5,
         'Sokoban-Push_7x7_2_120':1e-5,
         'MiniGrid-DoorKey-6x6-v0':1e-5, 'MiniGrid-Unlock-v0':1e-5, 'MiniGrid-RedBlueDoors-6x6-v0':1e-5,
         'MiniGrid-SimpleCrossingS9N1-v0':1e-4, 'MiniGrid-SimpleCrossingS9N2-v0':1e-4,
         'MiniGrid-LavaCrossingS9N1-v0':1e-4, 'MiniGrid-LavaCrossingS9N2-v0':1e-5,
         'MinAtar/Asterix-v0':3e-5, 'MinAtar/Breakout-v0':3e-4, 'MinAtar/Freeway-v0':1e-4,
         'MinAtar/Seaquest-v0':1e-4, 'MinAtar/SpaceInvaders-v0':1e-4}
    ,
    'DISCOR-DDQN-UER':
        {'Sokoban-Push_5x5_1_120':3e-4, 'Sokoban-Push_5x5_2_120':1e-4, 'Sokoban-Push_6x6_1_120':3e-4,
         'Sokoban-Push_6x6_2_120':1e-4, 'Sokoban-Push_6x6_3_120':3e-4, 'Sokoban-Push_7x7_1_120':1e-4,
         'Sokoban-Push_7x7_2_120':1e-4,
         'MiniGrid-DoorKey-6x6-v0':1e-4, 'MiniGrid-Unlock-v0':3e-5, 'MiniGrid-RedBlueDoors-6x6-v0':3e-4,
         'MiniGrid-SimpleCrossingS9N1-v0':1e-4, 'MiniGrid-SimpleCrossingS9N2-v0':3e-4,
         'MiniGrid-LavaCrossingS9N1-v0':1e-4, 'MiniGrid-LavaCrossingS9N2-v0':1e-4,
         'MinAtar/Asterix-v0':3e-5, 'MinAtar/Breakout-v0':1e-3, 'MinAtar/Freeway-v0':1e-4,
         'MinAtar/Seaquest-v0':1e-4, 'MinAtar/SpaceInvaders-v0':3e-4}
    ,
    'DDQN-UER':
        {'Sokoban-Push_5x5_1_120':3e-4, 'Sokoban-Push_5x5_2_120':1e-4, 'Sokoban-Push_6x6_1_120':3e-4,
         'Sokoban-Push_6x6_2_120':1e-4, 'Sokoban-Push_6x6_3_120':1e-4, 'Sokoban-Push_7x7_1_120':1e-5,
         'Sokoban-Push_7x7_2_120':3e-4,
         'MiniGrid-DoorKey-6x6-v0':3e-4, 'MiniGrid-Unlock-v0':1e-4, 'MiniGrid-RedBlueDoors-6x6-v0':3e-4,
         'MiniGrid-SimpleCrossingS9N1-v0':1e-4, 'MiniGrid-SimpleCrossingS9N2-v0':1e-4,
         'MiniGrid-LavaCrossingS9N1-v0':1e-4, 'MiniGrid-LavaCrossingS9N2-v0':1e-4,
         'MinAtar/Asterix-v0':1e-4, 'MinAtar/Breakout-v0':3e-4, 'MinAtar/Freeway-v0':1e-4,
         'MinAtar/Seaquest-v0':1e-4, 'MinAtar/SpaceInvaders-v0':3e-4}
    ,
    'DDQN-PER':
        {'Sokoban-Push_5x5_1_120':3e-4, 'Sokoban-Push_5x5_2_120':3e-4, 'Sokoban-Push_6x6_1_120':3e-4,
         'Sokoban-Push_6x6_2_120':3e-5, 'Sokoban-Push_6x6_3_120':1e-4, 'Sokoban-Push_7x7_1_120':1e-4,
         'Sokoban-Push_7x7_2_120':1e-4,
         'MiniGrid-DoorKey-6x6-v0':1e-4, 'MiniGrid-Unlock-v0':1e-4, 'MiniGrid-RedBlueDoors-6x6-v0':1e-4,
         'MiniGrid-SimpleCrossingS9N1-v0':1e-4, 'MiniGrid-SimpleCrossingS9N2-v0':3e-4,
         'MiniGrid-LavaCrossingS9N1-v0':1e-4, 'MiniGrid-LavaCrossingS9N2-v0':1e-4,
         'MinAtar/Asterix-v0':3e-5, 'MinAtar/Breakout-v0':3e-4, 'MinAtar/Freeway-v0':3e-5,
         'MinAtar/Seaquest-v0':1e-4, 'MinAtar/SpaceInvaders-v0':1e-4}
    ,
    'DDQN-TER':
        {'Sokoban-Push_5x5_1_120':1e-4, 'Sokoban-Push_5x5_2_120':1e-3, 'Sokoban-Push_6x6_1_120':3e-4,
         'Sokoban-Push_6x6_2_120':1e-4, 'Sokoban-Push_6x6_3_120':3e-4, 'Sokoban-Push_7x7_1_120':3e-4,
         'Sokoban-Push_7x7_2_120':1e-4,
         'MiniGrid-DoorKey-6x6-v0':3e-4, 'MiniGrid-Unlock-v0':3e-4, 'MiniGrid-RedBlueDoors-6x6-v0':3e-4,
         'MiniGrid-SimpleCrossingS9N1-v0':3e-4, 'MiniGrid-SimpleCrossingS9N2-v0':3e-4,
         'MiniGrid-LavaCrossingS9N1-v0':3e-4, 'MiniGrid-LavaCrossingS9N2-v0':3e-4,
         'MinAtar/Asterix-v0':3e-4, 'MinAtar/Breakout-v0':3e-4, 'MinAtar/Freeway-v0':3e-4,
         'MinAtar/Seaquest-v0':3e-4, 'MinAtar/SpaceInvaders-v0':3e-4}
}

optimal_para_kl_dict = {
    'Sokoban-Push_5x5_1_120': [0.8, 1.0, 0.1],
    'Sokoban-Push_6x6_1_120': [0.8, 2.0, 0.01],
    'Sokoban-Push_7x7_1_120': [0.5,2.0,0.01],
    'Sokoban-Push_6x6_3_120': [0.0, 2.0, 0.1],
    'Sokoban-Push_5x5_2_120': [0.8, 5.0, 0.01],
    'Sokoban-Push_6x6_2_120': [0.5, 5.0, 0.1],
    'Sokoban-Push_7x7_2_120': [0.8, 5.0, 0.01],
    'MiniGrid-DoorKey-6x6-v0': [0.5,2.0,0.01],
    'MiniGrid-Unlock-v0': [0.5,5.0,0.01],
    'MiniGrid-RedBlueDoors-6x6-v0': [0.5,2.0,0.01],
    'MiniGrid-SimpleCrossingS9N1-v0': [0.8, 2.0, 0.01],
    'MiniGrid-SimpleCrossingS9N2-v0': [0.2, 5.0, 0.001],
    'MiniGrid-LavaCrossingS9N1-v0': [0.5, 2.0, 0.01],
    'MiniGrid-LavaCrossingS9N2-v0': [0.2, 5.0, 0.01],
    'MinAtar/Asterix-v0': [0.0, 0.01, 0.1],
    'MinAtar/Breakout-v0': [0.5,2.0,0.01],
    'MinAtar/Freeway-v0': [0.0,0.1,0.01],
    'MinAtar/Seaquest-v0': [0.0, 0.01, 1.0],
    'MinAtar/SpaceInvaders-v0': [0.0,2.0,1.0],
    'PongNoFrameskip-v4': [0.2,5.0,1.0],
} # sample_method_para, policy_loss_para, tau
