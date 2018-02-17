#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from .util import *
from .altprop import Altprop
from .monoids import *
import copy


class GlobalInhibition(Altprop):

  def __init__(
      self,
      network,
      inhibitory_network,
      recursive=True,
      op=subM,
      gpu_id=-1
  ):
    super(GlobalInhibition, self).__init__()

    # the parent network
    self.network = network

    # mirror gradient networks (for each module in network)
    self.inhibitory_network = inhibitory_network
    self.inhibitory_network_hx = None

    # whether we should apply triggers recursively
    self.recursive = recursive
    self.op = op

    self.gpu_id = gpu_id

    # monkeypatch forward hooks to all leaf modules in the network
    monkeypatch_forwards(self.network, self._forward_update_hook)
    log.debug(self.network)
    log.debug("=============== Hooks registered =====================")

    self.initial_input = None
    # Set model's methods as our own
    method_list = [m for m in dir(self.network)
                   if callable(getattr(self.network, m)) and not m.startswith("__")
                   and not hasattr(self, m)]
    # for m in method_list:
    #   setattr(self, m, getattr(self.network, m))

  def _forward_update_hook(self, forward):
    def hook(*input, **kwargs):
      module = forward.__self__

      log.debug('Forward called for ' + str(module))
      if self.initial_input is None:
        self.initial_input = input

      # forward through the parent (excitatory) net
      excitatory = forward(*input, **kwargs)
      e = format(excitatory, module)

      # forward through the global inhibitory net
      hx = self.inhibitory_network_hx
      if type(self.initial_input) is tuple and len(self.initial_input) == 2:
        iinput = (self.initial_input[0], hx)
      else:
        iinput = self.initial_input
      inhibitory = self.inhibitory_network(*iinput, **kwargs)
      if type(self.initial_input) is tuple and len(self.initial_input) == 2:
        self.inhibitory_network_hx = detach_all(inhibitory[1])
      i = self.inhibitory_network.output(inhibitory)

      if type(excitatory) is tuple:
        excitatory = list(excitatory)
        excitatory[0] = self.op(e, i)
        excitatory = tuple(excitatory)
      else:
        excitatory = self.op(e, i)

      # print(T.norm(inhibitory[0] if type(inhibitory) is tuple else inhibitory, 2))
      return excitatory

    return hook

  def __format(self, outputs, module):
    return format(outputs, module)

  def forward(self, *args, **kwargs):
    log.debug("=============== Forward pass starting =====================")
    self.initial_input = None

    ret = self.network(*args, **kwargs)
    log.debug("=============== Forward pass done =====================")
    return ret

  def backward(self, *args, **kwargs):
    log.debug("=============== Backward pass starting =====================")
    ret = self.network.backward(*args, **kwargs)
    log.debug("=============== Backward pass done =====================")
    return ret

  def cuda(self, device_id=0):
    self.network.cuda(device_id)
    self.inhibitory_network = self.inhibitory_network.cuda(device_id)
    self.gpu_id = device_id
    return self
