#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T

from functools import reduce
from dni.util import *


class Monoid:

  def __init__(self, null, lift, op):
    self.null = null
    self.lift = lift
    self.op = op

  def fold(self, xs):
    if hasattr(xs, "__fold__"):
      return xs.__fold__(self)
    else:
      return reduce(self.op, (self.lift(x) for x in xs), self.null)

  def __call__(self, *args):
    return self.fold(args)

  def star(self):
    return Monoid(self.null, self.lift, self.op)

sumM = Monoid(0, lambda x: x, lambda a, b: a + b)
subM = Monoid(0, lambda x: x, lambda a, b: a - b)
prodM = Monoid(1, lambda x: x, lambda a, b: a * b)
divM = Monoid(1, lambda x: x, lambda a, b: T.div(a, b + δ))
matmulM = Monoid(1, lambda x: x, lambda a, b: T.matmul(a, b))
