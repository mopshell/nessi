#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: stacking.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2019 Damien Pageot
# ------------------------------------------------------------------
"""
Data stacking functions.

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np

def stack(object, **options):
    """
    Stack all traces in one.

    :param object: the SU-like data
    :param weight: a 1D array containing weight to apply to each trace.
    :param mean: if ``True`` divide the resulting trace by the number of traces.
    """

    # Get header parameters
    ns = object.header[0]['ns']
    dt = object.header[0]['dt']
    delrt = object.header[0]['delrt']
    trid = object.header[0]['trid']
    ntrac = len(object.header)

    # Get options
    weight = options.get('weight', np.ones(ntrac, dtype=np.float32))
    mean = options.get('mean', False)

    # Stacking traces
    stacktrac = np.zeros(ns, dtype=np.float32)
    for itrac in range(0, ntrac):
        stacktrac[:] += object.traces[itrac,:]*weight[itrac]

    # Mean
    if mean == True:
        stacktrac[:] /= np.sum(weight)

    # Update header
    object.header.resize(1)
    object.header[0]['tracl'] = 1
    object.header[0]['tracr'] = 1
    object.header[0]['tracf'] = 1

    # Update traces
    object.traces.resize(1, ns)
    object.traces = stacktrac[:]
