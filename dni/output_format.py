#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


class Formatter(object):
  def __init__(self):
    pass

  def format(self, o, obj):
    class_name = obj.__class__.__name__

    if class_name == 'LSTM':
      return o[0]
    elif class_name == 'GRU':
      return o[0]
    elif class_name == 'RNN':
      return o[0]
    else:
      return o

def format(o, obj):
  class_name = obj.__class__.__name__

  if class_name == 'LSTM':
    return o[0]
  elif class_name == 'GRU':
    return o[0]
  elif class_name == 'RNN':
    return o[0]
  else:
    return o

