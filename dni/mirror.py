#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from .util import *
from .altprop import Altprop
import copy


class Mirror(Altprop):

  def __init__(
      self,
      network,
      optim=None,
      mirror_optim='adam',
      mirror_lr=0.001,
      λ=0.0,
      recursive=True,
      gpu_id=-1
  ):
    super(Mirror, self).__init__()

    # the network and optimizer for the entire network
    self.network = network
    self.network_optimizer = \
        optim if optim is not None else self.get_optim(self.network.parameters(), mirror_optim, mirror_lr)

    # optim params for the mirror networks' optimizers
    self.mirror_optim = mirror_optim if optim is None else optim.__class__.__name__.lower()
    self.lr = mirror_lr if optim is None else optim.defaults['lr']
    # criterion for the grad estimator loss
    # optim params for the network's per-layer optimizers
    self.optim = str(self.network_optimizer.__class__.__name__).lower()

    # mirror gradient networks (for each module in network)
    self.mirror_networks = {}
    self.mirror_networks_hx = {}

    # lock that prevents the backward and forward hooks to act respectively
    self.backward_lock = False
    self.synthetic_grad_input = {}

    self.λ = λ

    # whether we should apply triggers recursively
    self.recursive = recursive

    self.gpu_id = gpu_id

    log.info('Creating mirror of network \n' + str(self.network) + '\n with optimizer ' +
             self.optim + ', lr ' + str(self.lr))

    # Create mirror nets for every leaf module
    if self.recursive:
      for module in for_all_leaves(self.network):
        self.__create_mirror_nets(module)
    else:
      self.__create_mirror_nets(self.network)

    # register backward hooks to all leaf modules in the network
    monkeypatch_forwards(self.network, self._forward_update_hook)
    log.debug(self.network)
    log.debug("=============== Hooks registered =====================")

    # Set model's methods as our own
    method_list = [m for m in dir(self.network)
                   if callable(getattr(self.network, m)) and not m.startswith("__")
                   and not hasattr(self, m)]
    # for m in method_list:
    #   setattr(self, m, getattr(self.network, m))

  def __create_mirror_nets(self, module):
    log.debug('Creating mirror net for ' + str(module))
    # the mirror network
    self.mirror_networks[id(module)] = copy.deepcopy(module)
    setattr(self, 'MIRROR_' + module.__class__.__name__, self.mirror_networks[id(module)])
    log.debug('Created mirror net: \n' + str(self.mirror_networks[id(module)]))

    self.mirror_networks_hx[id(module)] = None

    if self.gpu_id != -1:
      self.mirror_networks[id(module)] = self.mirror_networks[id(module)].cuda(self.gpu_id)

  def _forward_update_hook(self, forward):
    def hook(*input, **kwargs):
      module = forward.__self__

      log.debug('Forward called for ' + str(module))
      # forward through the parent (excitatory) net
      excitatory = forward(*input, **kwargs)
      e = format(excitatory, module)

      # forward through the mirror (inhibitory) net
      hx = self.mirror_networks_hx[id(module)]
      if type(input) is tuple and len(input) == 2:
        input = (input[0], hx)
      inhibitory = self.mirror_networks[id(module)](*input, **kwargs)
      if type(input) is tuple and len(input) == 2:
        self.mirror_networks_hx[id(module)] = detach_all(inhibitory[1])
      i = format(inhibitory, module)

      if type(excitatory) is tuple:
        excitatory = list(excitatory)
        excitatory[0] = (e / (i + δ))
        excitatory = tuple(excitatory)
      else:
        excitatory = (e / (i + δ))

      # print(T.norm(inhibitory[0] if type(inhibitory) is tuple else inhibitory, 2))
      return excitatory

    return hook

  def __format(self, outputs, module):
    return format(outputs, module)

  def forward(self, *args, **kwargs):
    log.debug("=============== Forward pass starting =====================")

    # self.register_forward(self.network, self._forward_update_hook, recursive=self.recursive)
    ret = self.network(*args, **kwargs)
    # self.unregister_forward()
    log.debug("=============== Forward pass done =====================")
    return ret

  def backward(self, *args, **kwargs):
    log.debug("=============== Backward pass starting =====================")
    ret = self.network.backward(*args, **kwargs)
    log.debug("=============== Backward pass done =====================")
    return ret

  def cuda(self, device_id=0):
    self.network.cuda(device_id)
    self.mirror_networks = { k: v.cuda(device_id) for k,v in self.mirror_networks.items()}
    self.gpu_id = device_id
    return self
