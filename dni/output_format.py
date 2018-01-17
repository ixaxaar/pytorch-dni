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

def detach_all(o):
  if type(o) is list:
    return [ detach_all(x) for x in o ]
  elif type(o) is tuple:
    return tuple([ detach_all(x) for x in o ])
  elif type(o) is dict:
    return { k: detach_all(v) for k,v in o.items() }
  elif type(o) is var:
    return o.detach()
  else:
    return o
