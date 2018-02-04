#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from .util import *
from .altprop import Altprop
from .dni_nets import LinearDNI
import copy


class DNI(Altprop):

  def __init__(
      self,
      network,
      dni_network=None,
      dni_params={},
      optim=None,
      grad_optim='adam',
      grad_lr=0.001,
      hidden_size=None,
      λ=0.5,
      recursive=True,
      gpu_id=-1
  ):
    super(DNI, self).__init__()

    # the DNI network generator
    self.dni_network = LinearDNI if dni_network is None else dni_network
    self.dni_params = dni_params

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

    # lock that prevents the backward and forward hooks to act respectively
    self.backward_lock = False
    self.synthetic_grad_input = {}

    # hidden size of the DNI network
    self.hidden_size = 10 if hidden_size is None else hidden_size

    self.λ = λ

    # whether we should apply triggers recursively
    self.recursive = recursive

    self.gpu_id = gpu_id
    self.ctr = 0
    self.cumulative_grad_losses = 0

    # register backward hooks to all leaf modules in the network
    self.register_backward(self.network, self._backward_update_hook, recursive=self.recursive)
    log.debug(self.network)
    log.debug("=============== Hooks registered =====================")

    log.info('Using DNI to optimize the network \n' + str(self.network) + '\n with optimizer ' +
             self.optim + ', lr ' + str(self.lr) + ' with ' + str(self.dni_network.__name__))

    # Set model's methods as our own
    method_list = [m for m in dir(self.network)
                   if callable(getattr(self.network, m)) and not m.startswith("__")
                   and not hasattr(self, m)]
    for m in method_list:
      setattr(self, m, getattr(self.network, m))

  def __get_dni_hidden(self, module):
    # get the DNI network's hidden state
    return self.dni_networks_data[id(module)]['hidden'] \
        if 'hidden' in self.dni_networks_data[id(module)] else None

  def __create_backward_dni_nets(self, module, output):
    log.debug('Creating DNI net for ' + str(module))
    # the DNI network
    dni_params = { **self.dni_params, **{'module': module} } \
        if hasattr(self.dni_params, 'module') else self.dni_params
    self.dni_networks[id(module)] = self.dni_network(
        input_size=output.size(-1),
        hidden_size=self.hidden_size,
        output_size=output.size(-1),
        **dni_params
    )
    setattr(self, 'dni_net_' + str(id(module)), self.dni_networks[id(module)])
    log.debug('Created DNI net: \n' + str(self.dni_networks[id(module)]))

    self.dni_networks_data[id(module)] = {}
    # the gradient module's (DNI network) optimizer
    self.dni_networks_data[id(module)]['grad_optim'] = \
        self.get_optim(self.dni_networks[id(module)].parameters(), otype=self.grad_optim, lr=self.lr)

    # the network module's optimizer
    self.dni_networks_data[id(module)]['optim'] = \
        self.get_optim(module.parameters(), otype=self.optim, lr=self.lr)

    # store the DNI outputs (synthetic gradients) here for calculating loss during backprop
    self.dni_networks_data[id(module)]['input'] = []
    self.synthetic_grad_input[id(module)] = None

    if self.gpu_id != -1:
      self.dni_networks[id(module)] = self.dni_networks[id(module)].cuda(self.gpu_id)

  def _forward_update_hook(self):
    def hook(module, input, output):
      log.debug('Forward hook called for ' + str(module))
      output = format(output, module)

      # create DNI networks and optimizers if they dont exist (for this module)
      if id(module) not in list(self.dni_networks.keys()):
        self.__create_backward_dni_nets(module, output)

      self.dni_networks_data[id(module)]['input'].append(detach_all(input))

    return hook

  def __save_synthetic_gradient(self, module, grad_input):
    def hook(m, i, o):
      if id(module) == id(m):
        self.synthetic_grad_input[id(module)] = detach_all(i)
    return hook

  def __format(self, outputs, module):
    return format(outputs, module)

  def _backward_update_hook(self):
    def hook(module, grad_input, grad_output):
      if self.backward_lock:
        log.debug("============= Backward locked for " + str(module))
        return

      log.debug('Backward hook called for ' + str(module) + '  ' +
                str(len(self.dni_networks_data[id(module)]['input'])))

      # store the synthetic gradient output during the backward pass
      handle = module.register_backward_hook(self.__save_synthetic_gradient(module, grad_input))

      self.dni_networks_data[id(module)]['optim'].zero_grad()
      self.dni_networks_data[id(module)]['grad_optim'].zero_grad()

      try:
        input = self.dni_networks_data[id(module)]['input'].pop()
      except IndexError:
        log.warning('Trying to search for non existent output ' + str(module))
        return

      # forward pass through the module alone
      outputs = module(*input)
      output = self.__format(outputs, module)

      # pass through the DNI net
      hx = self.__get_dni_hidden(module)
      predicted_grad, hx = \
          self.dni_networks[id(module)](output.detach(), hx if hx is None else detach_all(hx))

      # BP(λ)
      predicted_grad = as_type(predicted_grad, grad_output[0])
      if self.λ > 0:
        grad = (1 - self.λ) * predicted_grad + self.λ * grad_output[0]
      else:
        grad = predicted_grad

      self.backward_lock = True
      output.backward(grad.detach(), retain_graph=True)
      self.backward_lock = False
      handle.remove()

      # loss is MSE of the estimated gradient (by the DNI network) and the actual gradient
      loss = self.grad_loss(predicted_grad, grad_output[0].detach())

      # backprop and update the DNI net
      loss.backward()

      # track gradient losses
      self.ctr += 1
      self.cumulative_grad_losses = self.cumulative_grad_losses + loss.data.cpu().numpy()[0]
      if self.ctr % 1000 == 0:
        log.info('Average gradient loss last 1k steps: ' + str(self.cumulative_grad_losses / self.ctr))
        self.cumulative_grad_losses = 0
        self.ctr = 0

      # update parameters
      self.dni_networks_data[id(module)]['optim'].step()
      self.dni_networks_data[id(module)]['grad_optim'].step()

      # (back)propagate the (mixed) synthetic and original gradients
      if any(x is None for x in grad_input) or \
              any(x is None for x in self.synthetic_grad_input[id(module)]) or \
              self.synthetic_grad_input[id(module)] is None:
        grad_inputs = None
      else:
        zipped = [(as_type(s, a), a)
                  for s, a in zip(self.synthetic_grad_input[id(module)], grad_input)]
        grad_inputs = tuple(((1 - self.λ) * s) + (self.λ * a.detach()) for (s, a) in zipped)
      return grad_inputs

    return hook

  def forward(self, *args, **kwargs):
    log.debug("=============== Forward pass starting =====================")
    # clear out all buffers
    for i, data in self.dni_networks_data.items():
      data['input'] = []

    self.register_forward(self.network, self._forward_update_hook, recursive=self.recursive)
    ret = self.network(*args, **kwargs)
    self.unregister_forward()
    log.debug("=============== Forward pass done =====================")
    return ret

  def backward(self, *args, **kwargs):
    log.debug("=============== Backward pass starting =====================")
    ret = self.network.backward(*args, **kwargs)
    log.debug("=============== Backward pass done =====================")
    return ret

  def cuda(self, device=0):
    self.network.cuda(device_id)
    self.gpu_id = device_id
    return self
