#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch as T
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
import torch.optim as optim
import numpy as np

import logging
import inspect
from collections import OrderedDict
from math import floor


δ = 1e-6

log = logging.getLogger('dni')
log.setLevel(logging.INFO)
# create file handler which logs even debug messages
fh = logging.FileHandler('dni.log')
fh.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s: %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
log.addHandler(fh)
log.addHandler(ch)


def detach_all(o):
  if type(o) is list:
    return [detach_all(x) for x in o]
  elif type(o) is tuple:
    return tuple([detach_all(x) for x in o])
  elif type(o) is dict:
    return {k: detach_all(v) for k, v in o.items()}
  elif type(o) is set:
    return set([ detach_all(x) for x in o ])
  elif type(o) is Variable:
    return o.detach()
  elif type(o) is OrderedDict:
    return OrderedDict({ k: detach_all(v) for k,v in o.items() })
  else:
    return o


def format(o, obj):
  class_name = obj.__class__.__name__

  if class_name == 'LSTM' or class_name == 'LSTMCell':
    return o[0].float()
  elif class_name == 'GRU' or class_name == 'GRUCell':
    return o[0].float()
  elif class_name == 'RNN' or class_name == 'RNNCell':
    return o[0].float()
  elif class_name == 'Linear':
    return o.float()
  else:
    return o


def is_leaf(module):
  l = 0
  for x in module.children():
    l += 1
  p = sum([1 for x in module.parameters()])

  return l == 0 and p > 0

def for_all_leaves(net):
  for module in net.modules():
    if is_leaf(module):
      yield module

def monkeypatch_forwards(net, callback, *args, **kwargs):
  for module in for_all_leaves(net):
    log.debug('Monkeypatching forward for ' + str(module))
    cb = callback(module.forward, *args, **kwargs)
    setattr(module, 'forward', cb)

def as_type(var1, ref):
  dtype = ref.data.__class__.__name__

  if type(var1) is not Variable:
    return var1
  if dtype == 'FloatTensor':
    return var1.float()
  elif dtype == 'DoubleTensor':
    return var1.double()
  elif dtype == 'HalfTensor':
    return var1.half()
  elif dtype == 'ByteTensor':
    return var1.byte()
  elif dtype == 'CharTensor':
    return var1.char()
  elif dtype == 'ShortTensor':
    return var1.short()
  elif dtype == 'IntTensor':
    return var1.int()
  elif dtype == 'LongTensor':
    return var1.long()
  else:
    return var1


def register_nan_checks(model):
  def check_grad(module, grad_input, grad_output):
    if any(np.all(np.isnan(gi.data.cpu().numpy())) for gi in grad_input if gi is not None):
      print('NaN gradient in grad_input ' + type(module).__name__)

  model.apply(lambda module: module.register_backward_hook(check_grad))

def check_if_gradients_propagate(model):
  def check_grad(module, grad_input, grad_output):
    print('Gradients propagating in module ', module)

  model.apply(lambda module: module.register_backward_hook(check_grad))


def get_padding(input_dim, kernel_size, dilation, stride):
  padding = floor((input_dim * stride + (kernel_size - 1) * dilation + 1)/2)
  return padding


def cuda(x, grad=False, gpu_id=-1):
  if gpu_id == -1:
    return Variable(x, requires_grad=grad)
  else:
    return Variable(x.pin_memory(), requires_grad=grad).cuda(gpu_id, async=True)

