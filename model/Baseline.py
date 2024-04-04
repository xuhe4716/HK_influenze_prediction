import torch.nn as nn

class Baseline(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        # Store initialization parameters, although the Baseline model doesn't actually use parameters other than output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        #self.device = device

    def forward(self, input_seq):
        # Baseline will extract the first feature of the last time step of the input sequence as the predicted value
        #print(input_seq)
        pred = input_seq[:, -1, 0].unsqueeze(-1)
        if self.output_size > 1:
            pred = pred.repeat(1, self.output_size)
        return pred
