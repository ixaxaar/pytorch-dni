#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


def format(o, obj):
  class_name = obj.__class__.__name__

  if class_name == 'Embedding':
    return o
  elif class_name == 'LSTM':
    return o[0]
  elif class_name == 'Linear':
    return o
  else:
    raise Exception('Unknown output format, please define a custom one for ' + str(obj))

