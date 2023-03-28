import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as f


class QNetwork(nn.Module):
    def __init__(self, action_space,state_space,atari_name):
        super(QNetwork, self).__init__()
        self.atari_name = atari_name

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        padding = 0
        self.conv_1 = nn.Conv2d(state_space[-1], 16, kernel_size=3, stride=1,padding=padding)
        torch.nn.init.kaiming_normal_(self.conv_1.weight,nonlinearity='relu')
        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(state_space[0]+padding*2) * size_linear_unit(state_space[1]+padding*2) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)
        torch.nn.init.kaiming_normal_(self.fc_hidden.weight, nonlinearity='relu')
        # Output layer:
        self.output = nn.Linear(in_features=128, out_features=action_space)
        torch.nn.init.kaiming_normal_(self.output.weight, nonlinearity='relu')
    def forward(self, x):
        # print('state:',x)
        if 'Sokoban' in self.atari_name:
            x = x / 255.  # scale
            # print('divided 255!')

        x = torch.transpose(x, 1, 3)  # NHWC -> NCHW
        # Rectified output from the first conv layer
        x = f.relu(self.conv_1(x))

        # Rectified output from the final hidden layer
        x = f.relu(self.fc_hidden(x.view(x.size(0), -1)))

        # Returns the output from the fully-connected linear layer
        return self.output(x)
