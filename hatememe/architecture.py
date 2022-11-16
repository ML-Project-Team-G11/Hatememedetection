import torch.nn as nn
import torch.nn.functional as F

class HMMLP(nn.Module):

    def __init__(self, n_in=512*2, n_out=1, ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(n_in, n_out)
        # self.fc2 = nn.Linear(256, 64)
        # self.fc3 = nn.Linear(64, 16)
        # self.fc4 = nn.Linear(16, n_out)

    def forward(self, x):
        
        x = self.fc1(x)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)

        return x