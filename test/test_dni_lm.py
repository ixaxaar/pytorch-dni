# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import pytest
# import numpy as np

# import torch.nn as nn
# import torch as T
# from torch.autograd import Variable as var
# import torch.nn.functional as F
# from torch.nn.utils import clip_grad_norm
# import torch.optim as optim
# import numpy as np

# import sys
# import os
# import math
# import time
# sys.path.insert(0, '.')

# import functools

# from dni import DNI
# from test_lm import RNNModel

# def test_dni():
#   T.manual_seed(1111)

#   rnn_type = 'LSTM'
#   ntoken = 100
#   ninp = 3
#   nhid = 7
#   nlength = 8
#   nlayers = 2
#   dropout = 0.2
#   batch_size = 5

#   net = RNNModel(rnn_type, ntoken, ninp, nhid, nlayers, dropout)
#   hx = net.init_hidden(batch_size)
#   input = T.arange(batch_size * nlength).view(batch_size, nlength).long()

#   net = DNI(net)
#   out = net(var(input), None)
#   # net.backward(0)

