#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .altprop import Altprop
from .dni import DNI
from .cdni import CDNI
from .mirror import Mirror
from .global_inhibition import GlobalInhibition
from .dni_monkeypatch import _DNI

from .dni_nets import DNINetwork
from .dni_nets import RNNDNI
from .dni_nets import LinearDNI, LinearBatchNormDNI, LinearSigmoidDNI
from .dni_nets import Conv2dDNI

from .monoids import Monoid
from .monoids import sumM
from .monoids import subM
from .monoids import prodM
from .monoids import divM
from .monoids import matmulM

from .util import detach_all, get_padding
