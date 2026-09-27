import torch
import torch.nn as nn
import torch.nn.functional as F



class ChannelDropout(nn.Module):
    def __init__(self, prob = 0.1):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if not self.training:
            return x
        col_number = int(self.prob * x.size(1))    
        col_indexes = torch.randperm(x.size(1))[:col_number]
        x = x.clone()
        x[:, col_indexes] = 0
        actual_prob = actual_prob = col_number / x.size(1)
        x = x / (1 - actual_prob)
        return x

class CDLM(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(CDLM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
  

        self.norm1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = ChannelDropout(0.5)
        self.linear1 = nn.Linear(input_dim, 1024)
        self.dropout2 = ChannelDropout(0.5)

        self.norm2 = nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1024, 512)
        self.dropout3 = ChannelDropout(0.5)
    

        self.norm3 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, 256)
        self.dropout4 = ChannelDropout(0.5)
        

        self.norm4 = nn.BatchNorm1d(256)
        self.linear4 = nn.Linear(256, 256)
        self.dropout5 = ChannelDropout(0.5)

        self.output = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.dropout2(F.relu(self.linear1(self.dropout1(self.norm1(x)))))
        
        x = self.dropout3(F.relu(self.linear2(self.norm2(x))))
   
        x = self.dropout4(F.relu(self.linear3(self.norm3(x))))

        x = self.dropout5(F.relu(self.linear4(self.norm4(x))))
  
        output = self.output(x)
        return output
