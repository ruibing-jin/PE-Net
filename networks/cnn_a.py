import torch.nn as nn
import torch
from collections import OrderedDict

class CNN_A(nn.Module):
    def __init__(self, num_hidden, input_dim, aux_dim):
        super(CNN_A, self).__init__()
        self.num_hidden = num_hidden
        self.input_dim = input_dim
        self.aux_dim = aux_dim
        print('Load the CNN-A...')

        self.cnn_basic = nn.Sequential(OrderedDict([
                                ('cnn1', nn.Conv1d(in_channels = 14, out_channels = 16, kernel_size = 5, stride=2, padding=2)),
                                ('bn1', nn.BatchNorm1d(16)),
                                ('relu1', nn.ReLU(inplace=True)),
                                ('cnn2_1', nn.Conv1d(16, 64, 3, padding=1)),
                                ('bn2_1', nn.BatchNorm1d(64)),
                                ('relu2_1', nn.ReLU(inplace=True)),
                                ]))

        self.fc = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(960, 256)),
                                ('relu1', nn.ReLU(inplace=True)),
                                ('dropout1', nn.Dropout(p=0.2)),
                                ('fc2', nn.Linear(256, 256)),
                                ('relu2', nn.ReLU(inplace=True)),
                                ('dropout2', nn.Dropout(p=0.2))
                                ]))

        self.reg = nn.Linear(256, 1)

    # Defining the forward pass
    def forward(self, x):
        # x shape: (N, L, H_in)

        x_cnn = torch.transpose(x, 1, 2)
        x_cnn = self.cnn_basic(x_cnn)

        x_cnn = x_cnn.view(x_cnn.shape[0],-1)
        x_cnn = self.fc(x_cnn)

        out = self.reg(x_cnn)

        return out