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
      λ=0.5,
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
    self.synthetic_grad_input = {}

    # hidden size of the DNI network
    self.hidden_size = 10 if hidden_size is None else hidden_size

    self.λ = λ

    self.gpu_id = gpu_id

    # register forward and backward hooks to all leaf modules in the network
    # self.register_forward(self.network, self._forward_update_hook)
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
        handle = module.register_forward_hook(h)
        self.forward_hooks += [{"name": str(module), "id": id(module), "hook": handle}]

  def unregister_forward(self):
    for h in self.forward_hooks:
      h['hook'].remove()
    self.forward_hooks = []

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
        self.dni_networks_data[id(module)]['input'] = []
        self.synthetic_grad_input[id(module)] = None

        if self.gpu_id != -1:
          self.dni_networks[id(module)] = self.dni_networks[id(module)].cuda(self.gpu_id)

      self.dni_networks_data[id(module)]['input'].append(detach_all(input))

    return hook

  def _backward_update_hook(self):
    def hook(module, grad_input, grad_output):
      if self.backward_lock:
        log.debug("============= Backward locked for " + str(module))
        # lock valid for only one module
        # TODO: check if these handles are called asynchronously
        self.backward_lock = False
        return

      log.debug('Backward hook called for ' + str(module) + '  ' +
                str(len(self.dni_networks_data[id(module)]['input'])))

      def save_synthetic_gradient(module, grad_input):
        def hook(m, i, o):
          if any(x is None for x in grad_input):
            self.synthetic_grad_input[id(module)] = None
            return
          else:
            # TODO: replace None grad_inputs with zero tensors?
            i = tuple([x if x is not None else T.zeros(grad_input[ctr].size()) for ctr, x in enumerate(i)])

            if id(module) == id(m):
              self.synthetic_grad_input[id(module)] = detach_all(i)
        return hook

      # store the synthetic gradient output during the backward pass
      handle = module.register_backward_hook(save_synthetic_gradient(module, grad_input))

      self.dni_networks_data[id(module)]['optim'].zero_grad()
      self.dni_networks_data[id(module)]['grad_optim'].zero_grad()

      try:
        input = self.dni_networks_data[id(module)]['input'].pop()
      except IndexError:
        log.warning('Trying to search for non existent output ' + str(module))
        return

      # forward pass through the network module
      output = module(*input)
      output = format(output, module)

      # pass through the DNI net
      hx = self.__get_dni_hidden(module)
      predicted_grad, hx = self.dni_networks[id(module)](output.detach(), hx if hx is None else detach_all(hx))

      # BP(λ)
      grad = (1 - self.λ) * predicted_grad + self.λ * grad_output[0]
      self.backward_lock = True
      output.backward(grad.detach())
      self.backward_lock = False
      handle.remove()

      # loss is MSE of the estimated gradient (by the DNI network) and the actual gradient
      loss = self.grad_loss(predicted_grad, grad_output[0].detach())

      # backprop and update the DNI net
      loss.backward()

      # update parameters
      self.dni_networks_data[id(module)]['optim'].step()
      self.dni_networks_data[id(module)]['grad_optim'].step()

      # (back)propagate the (mixed) synthetic and original gradients
      if any(x is None for x in grad_input) or \
              self.synthetic_grad_input[id(module)] is None:
        grad_inputs = None
      else:
        self.synthetic_grad_input[id(module)] = \
            [x if type(x) is var else var(x) for x in self.synthetic_grad_input[id(module)]]

        grad_inputs = tuple(((1 - self.λ) * s) + (self.λ * a.detach())
                            for s, a in zip(self.synthetic_grad_input[id(module)], grad_input))
      return grad_inputs

    return hook

  def cuda(self, device_id):
    self.network.cuda(device_id)
    self.gpu_id = device_id
    return self

  def forward(self, *args, **kwargs):
    log.debug("=============== Forward pass starting =====================")
    self.register_forward(self.network, self._forward_update_hook)
    ret = self.network(*args, **kwargs)
    self.unregister_forward()
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
