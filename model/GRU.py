# 定义GRU网络
import torch
from torch import nn
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size,device):
        #(self, input_size, hidden_size, num_layers, output_size, batch_size, device)
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.batch_size = batch_size
        self.device = device
        # feature_size为特征维度，就是每个时间点对应的特征数量，这里为1
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        # GRU 没有细胞状态
        # c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        # print(input_seq.size())
        # input(batch_size, seq_len, input_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.gru(input_seq, h_0)
        pred = self.fc(output)  # pred(batch_size, seq_len, output_size)
        pred = pred[:, -1, :]

        return pred
