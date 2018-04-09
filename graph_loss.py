import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
class GraphLoss(nn.Module):
    # format of the penalizing class: distance = (i, j, similarity), size = number_of_pair * 3
    def __init__(self, loss_weight=1e-8, use_cuda=False):
        super(GraphLoss, self).__init__()
        self.loss_weight = loss_weight
        self.use_cuda = use_cuda

    def forward(self, y, att):
        # Generate W matrix
        norm1 = torch.norm(att, 2, 1, True)
        norm2 = torch.norm(att.t(), 2, 0, True)
        norm = torch.mm(norm1, norm2)
        W0 = torch.div(torch.mm(att, att.t()), norm)
        # set the threshold to 0.7
        W1 = torch.mul(W0, (W0 > 0.7).float())
        # W1 = 2 * torch.mul(W0, (W0 > 0.8).float()) - 1
        W = W1 - Variable(torch.eye(W1.shape[0], W1.shape[1]))

        # Generate D matrix
        D = torch.diag(torch.sum(W, 1))

        # Generate L matrix
        L = D - W

        # Regularize Loss
        temp = torch.mm(torch.mm(y.t(), L), y)
        loss = torch.trace(temp)

        return self.loss_weight * loss

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.

        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device_id))


def main():

    y = Variable(torch.randn((64, 1024)))
    att = Variable(torch.randn((64, 102)))
    ct = GraphLoss()

    out = ct(y, att)
    out.backward()

if __name__ == '__main__':
    main()
