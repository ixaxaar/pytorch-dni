#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
import torch.optim as optim
import numpy as np

import logging

log = logging.getLogger('dni')
log.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('dni.log')
fh.setLevel(logging.DEBUG)
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
    return [ detach_all(x) for x in o ]
  elif type(o) is tuple:
    return tuple([ detach_all(x) for x in o ])
  elif type(o) is dict:
    return { k: detach_all(v) for k,v in o.items() }
  elif type(o) is var:
    return o.detach()
  else:
    return o

