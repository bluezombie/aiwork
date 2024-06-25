import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN_CNN_Model(nn.Module):
    def __init__(self,  env_inputs, n_actions):
        super(DQN_CNN_Model, self).__init__()

        self.input_shape = env_inputs
        self.num_actions = n_actions
        
        self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate the size of the linear layer input
        self.fc_input_dim = self._get_conv_output(self.input_shape)
        
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def _get_conv_output(self, shape):
        # Create a dummy input tensor and pass it through the conv layers
        # to calculate the output size of the conv layers
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            output = self.conv1(input)
            output = self.conv2(output)
            output = self.conv3(output)
            return int(np.prod(output.size()))

        
    def forward(self, env_input):
        result = F.relu(self.conv1(env_input))
        result = F.relu(self.conv2(result))
        result = F.relu(self.conv3(result))
        result = result.view(result.size(0), -1)  # Flatten the tensor
        result = F.relu(self.fc1(result))
        result = self.fc2(result)
        return result