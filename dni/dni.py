#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from .linear_dni import Linear_DNI
from .output_format import *
from .util import *


class DNI(nn.Module):

  def __init__(
      self,
      network,
      dni_network=None,
      optim=None,
      grad_optim='adam',
      grad_lr=0.001,
      hidden_size=None,
      gpu_id=-1
  ):
    super(DNI, self).__init__()

    # the DNI network generator
    self.dni_network = Linear_DNI if dni_network is None else dni_network

    # the network and optimizer for the entire network
    self.network = network
    self.network_optimizer = \
        optim if optim is not None else self.get_optim(self.network.parameters(), grad_optim, grad_lr)

    # optim params for the DNI networks' optimizers
    self.grad_optim = grad_optim if optim is None else optim.__class__.__name__.lower()
    self.lr = grad_lr if optim is None else optim.defaults['lr']
    # criterion for the grad estimator loss
    self.grad_loss = nn.MSELoss()
    # optim params for the network's per-layer optimizers
    self.optim = str(self.network_optimizer.__class__.__name__).lower()

    # DNI synthetic gradient networks (for each module in network)
    self.dni_networks = {}
    self.dni_networks_data = {}

    self.forward_hooks = []
    self.backward_hooks = []
    # lock that prevents the backward and forward hooks to act respectively
    self.backward_lock = False
    self.forward_lock = False

    # hidden size of the DNI network
    self.hidden_size = 10 if hidden_size is None else hidden_size

    self.gpu_id = gpu_id

    # register forward and backward hooks to all leaf modules in the network
    self.register_forward(self.network, self._forward_update_hook)
    self.register_backward(self.network, self._backward_update_hook)
    log.debug(self.network)
    log.debug("=============== Hooks registered =====================")

  def register_forward(self, network, hook):
    for module in network.modules():
      # register hooks only to leaf nodes in the graph with at least 1 learnable Parameter
      l = 0
      for x in module.children():
        l += 1
      p = sum([1 for x in module.parameters()])

      if l == 0 and p > 0:
        # register forward hooks
        h = hook()
        log.debug('Registering forward hooks for ' + str(module))
        module.register_forward_hook(h)
        self.forward_hooks += [{"name": str(module), "id": id(module), "hook": h}]

  def register_backward(self, network, hook):
    for module in network.modules():
      # register hooks only to leaf nodes in the graph with at least 1 learnable Parameter
      l = 0
      for x in module.children():
        l += 1
      p = sum([1 for x in module.parameters()])

      if l == 0 and p > 0:
        # register backward hooks
        h = hook()
        log.debug('Registering backward hooks for ' + str(module))
        module.register_backward_hook(h)
        self.backward_hooks += [{"name": str(module), "id": id(module), "hook": h}]

  def __get_dni_hidden(self, module):
    # get the DNI network's hidden state
    return self.dni_networks_data[id(module)]['hidden'] \
          if 'hidden' in self.dni_networks_data[id(module)] else None

  def _forward_update_hook(self):
    def hook(module, input, output):
      if self.forward_lock:
        log.debug("============= Forward locked for " + str(module))
        return

      log.debug('Forward hook called for ' + str(module))
      output = format(output, module)

      # create DNI networks and optimizers if they dont exist (for this module)
      if id(module) not in list(self.dni_networks.keys()):
        # the DNI network
        self.dni_networks[id(module)] = self.dni_network(
            input_size=output.size(-1),
            hidden_size=self.hidden_size,
            output_size=output.size(-1)
        )

        self.dni_networks_data[id(module)] = {}
        # the gradient module's (DNI network) optimizer
        self.dni_networks_data[id(module)]['grad_optim'] = \
            self.get_optim(self.dni_networks[id(module)].parameters(), otype=self.grad_optim, lr=self.lr)

        # the network module's optimizer
        self.dni_networks_data[id(module)]['optim'] = \
            self.get_optim(module.parameters(), otype=self.optim, lr=self.lr)

        # store the DNI outputs (synthetic gradients) here for calculating loss during backprop
        self.dni_networks_data[id(module)]['output'] = []

        if self.gpu_id != -1:
          self.dni_networks[id(module)] = self.dni_networks[id(module)].cuda(self.gpu_id)

      self.dni_networks_data[id(module)]['optim'].zero_grad()

      # get the grad-detached output for this module
      # this is done so that the entire network does not backprop for every layer
      # TODO: We're actually doing a forward pass again here to rid the module of the input's history
      # this is costly as shit
      input = detach_all(input)
      self.forward_lock = True
      output = module(*input)
      self.forward_lock = False
      output = format(output, module)

      hx = self.__get_dni_hidden(module)
      # pass through the DNI network, get updated gradients for the host network
      self.dni_networks[id(module)].eval()
      grad, hx = self.dni_networks[id(module)](output.detach(), hx if hx is None else hx.detach())
      self.dni_networks[id(module)].train()

      # backprop with generated gradients
      self.backward_lock = True
      output.backward(grad.detach())
      self.backward_lock = False

      # optiimize the module's params
      # TODO: parameter = parameter - grad - try subtractive directly on param weights!
      # can inhibitory neurons be gradient estimators? :O
      self.dni_networks_data[id(module)]['optim'].step()

      # store the hidden state and output
      self.dni_networks_data[id(module)]['hidden'] = hx
      self.dni_networks_data[id(module)]['output'].append(output.detach())

    return hook

  def _backward_update_hook(self):
    def hook(module, grad_input, grad_output):
      if self.backward_lock:
        log.debug("============= Backward locked for " + str(module))
        return

      log.debug('Backward hook called for ' + str(module))
      self.dni_networks_data[id(module)]['grad_optim'].zero_grad()

      # get the network module's output
      output = self.dni_networks_data[id(module)]['output'].pop()
      hx = self.__get_dni_hidden(module)
      # pass through the DNI net
      predicted_grad, hx = self.dni_networks[id(module)](output, hx if hx is None else hx.detach())

      # loss is MSE of the estimated gradient (by the DNI network) and the actual gradient
      loss = self.grad_loss(predicted_grad, grad_output[0].detach())

      # backprop and update the DNI net
      loss.backward()
      self.dni_networks_data[id(module)]['grad_optim'].step()
    return hook

  def cuda(self, device_id):
    self.network.cuda(device_id)
    self.gpu_id = device_id
    return self

  def forward(self, *args, **kwargs):
    log.debug("=============== Forward pass starting =====================")
    ret = self.network(*args, **kwargs)
    log.debug("=============== Forward pass done =====================")
    return ret

  def backward(self, *args, **kwargs):
    log.debug("=============== Backward pass starting =====================")
    ret = self.network.backward(*args, **kwargs)
    log.debug("=============== Backward pass done =====================")
    return ret

  def get_optim(self, parameters, otype="adam", lr=0.001):
    if type(otype) is str:
      if otype == 'adam':
        optimizer = optim.Adam(parameters, lr=lr, eps=1e-9, betas=[0.9, 0.98])
      elif otype == 'adamax':
        optimizer = optim.Adamax(selfparameters, lr=lr, eps=1e-9, betas=[0.9, 0.98])
      elif otype == 'rmsprop':
        optimizer = optim.RMSprop(parameters, lr=lr, momentum=0.9, eps=1e-10)
      elif otype == 'sgd':
        optimizer = optim.SGD(parameters, lr=lr)  # 0.01
      elif otype == 'adagrad':
        optimizer = optim.Adagrad(parameters, lr=lr)
      elif otype == 'adadelta':
        optimizer = optim.Adadelta(parameters, lr=lr)

    return optimizer
