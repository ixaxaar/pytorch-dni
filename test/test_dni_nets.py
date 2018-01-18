#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
import torch.optim as optim
import numpy as np

import sys
import os
import math
import time
sys.path.insert(0, '.')

import functools

from dni import *
from dnc import *
from test_utils import generate_data, criterion

def test_rnn_dni():
  T.manual_seed(1111)

  input_size = 100
  hidden_size = 100
  rnn_type = 'lstm'
  num_layers = 3
  num_hidden_layers = 5
  dropout = 0.2
  nr_cells = 200
  cell_size = 17
  read_heads = 2
  sparse_reads = 4
  gpu_id = -1
  debug = True
  lr = 0.001
  sequence_max_length = 10
  batch_size = 10
  cuda = gpu_id
  clip = 20
  length = 13

  rnn = SAM(
      input_size=input_size,
      hidden_size=hidden_size,
      rnn_type=rnn_type,
      num_layers=num_layers,
      num_hidden_layers=num_hidden_layers,
      dropout=dropout,
      nr_cells=nr_cells,
      cell_size=cell_size,
      read_heads=read_heads,
      sparse_reads=sparse_reads,
      gpu_id=gpu_id,
      debug=debug
  )

  optimizer = optim.Adam(rnn.parameters(), lr=lr)
  rnn = DNI(rnn, optim=optimizer, dni_network=RNN_DNI)
  optimizer.zero_grad()

  input_data, target_output = generate_data(batch_size, length, input_size, cuda)
  target_output = target_output.transpose(0, 1).contiguous()

  output, (chx, mhx, rv), v = rnn(input_data, None)
  output = output.transpose(0, 1)

  loss = criterion((output), target_output)
  loss.backward()

  T.nn.utils.clip_grad_norm(rnn.parameters(), clip)
  optimizer.step()

  assert target_output.size() == T.Size([27, 10, 100])
  assert chx[0][0].size() == T.Size([num_hidden_layers,10,100])
  # assert mhx['memory'].size() == T.Size([10,12,17])
  assert rv.size() == T.Size([10, 34])


def test_linear_dni():
  T.manual_seed(1111)

  input_size = 100
  hidden_size = 100
  rnn_type = 'lstm'
  num_layers = 3
  num_hidden_layers = 5
  dropout = 0.2
  nr_cells = 200
  cell_size = 17
  read_heads = 2
  sparse_reads = 4
  gpu_id = -1
  debug = True
  lr = 0.001
  sequence_max_length = 10
  batch_size = 10
  cuda = gpu_id
  clip = 20
  length = 13

  rnn = SAM(
      input_size=input_size,
      hidden_size=hidden_size,
      rnn_type=rnn_type,
      num_layers=num_layers,
      num_hidden_layers=num_hidden_layers,
      dropout=dropout,
      nr_cells=nr_cells,
      cell_size=cell_size,
      read_heads=read_heads,
      sparse_reads=sparse_reads,
      gpu_id=gpu_id,
      debug=debug
  )

  optimizer = optim.Adam(rnn.parameters(), lr=lr)
  rnn = DNI(rnn, optim=optimizer, dni_network=Linear_DNI)
  optimizer.zero_grad()

  input_data, target_output = generate_data(batch_size, length, input_size, cuda)
  target_output = target_output.transpose(0, 1).contiguous()

  output, (chx, mhx, rv), v = rnn(input_data, None)
  output = output.transpose(0, 1)

  loss = criterion((output), target_output)
  loss.backward()

  T.nn.utils.clip_grad_norm(rnn.parameters(), clip)
  optimizer.step()

  assert target_output.size() == T.Size([27, 10, 100])
  assert chx[0][0].size() == T.Size([num_hidden_layers,10,100])
  # assert mhx['memory'].size() == T.Size([10,12,17])
  assert rv.size() == T.Size([10, 34])

def test_linear_sigmoid_dni():
  T.manual_seed(1111)

  input_size = 100
  hidden_size = 100
  rnn_type = 'lstm'
  num_layers = 3
  num_hidden_layers = 5
  dropout = 0.2
  nr_cells = 200
  cell_size = 17
  read_heads = 2
  sparse_reads = 4
  gpu_id = -1
  debug = True
  lr = 0.001
  sequence_max_length = 10
  batch_size = 10
  cuda = gpu_id
  clip = 20
  length = 13

  rnn = SAM(
      input_size=input_size,
      hidden_size=hidden_size,
      rnn_type=rnn_type,
      num_layers=num_layers,
      num_hidden_layers=num_hidden_layers,
      dropout=dropout,
      nr_cells=nr_cells,
      cell_size=cell_size,
      read_heads=read_heads,
      sparse_reads=sparse_reads,
      gpu_id=gpu_id,
      debug=debug
  )

  optimizer = optim.Adam(rnn.parameters(), lr=lr)
  rnn = DNI(rnn, optim=optimizer, dni_network=Linear_Sigmoid_DNI)
  optimizer.zero_grad()

  input_data, target_output = generate_data(batch_size, length, input_size, cuda)
  target_output = target_output.transpose(0, 1).contiguous()

  output, (chx, mhx, rv), v = rnn(input_data, None)
  output = output.transpose(0, 1)

  loss = criterion((output), target_output)
  loss.backward()

  T.nn.utils.clip_grad_norm(rnn.parameters(), clip)
  optimizer.step()

  assert target_output.size() == T.Size([27, 10, 100])
  assert chx[0][0].size() == T.Size([num_hidden_layers,10,100])
  # assert mhx['memory'].size() == T.Size([10,12,17])
  assert rv.size() == T.Size([10, 34])


