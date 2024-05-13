import torch
import torch.nn as nn


class IEEM(nn.Module):

    def __init__(self):
        super(IEEM, self).__init__()

    def forward(self, out, act):
        out[out < -2.095122] = -2.09
        loss = (3.819922*torch.log(2.095122 + act) - 3.819922*torch.log(2.095122 + out)) ** 2

        # g(x) = 3.819922*ln(2.095122+x)-0.1140096

        # category 1 : (0, 35]
        # real vlaue = 35
        # predict value =25 45

        return torch.mean(loss)
