#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: windowing.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2018 Damien Pageot
# ------------------------------------------------------------------
"""
Data windowing functions.

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np

def time_window(object, **options):
    """
    Window traces in time.

    :param object: input Stream object containing traces to window.
    :param tmin: (optional) minimum time to pass in second (default tmin=0.0).
    :param tmax: (optional) maximum time to pass in second (default tmax=0.0).
    """

    # Get number of time sample and time sampling from header
    ns = object.header[0]['ns']
    dt = object.header[0]['dt']/1000000.
    delrt = object.header[0]['delrt']/1000.

    # Get the number of traces
    if object.traces.ndim == 1:
        ntrac = 1
    if object.traces.ndim == 2:
        ntrac = np.size(object.traces, axis=0)

    # Get options
    tmin = options.get('tmin', 0.)
    tmax = options.get('tmax', 0.)

    # Calculate the index of tmin and tmax and the size of the new data array
    itmin = np.int((tmin-delrt)/dt)
    itmax = np.int((tmax-delrt)/dt)
    nsnew = itmax-itmin
    if tmin < 0:
        delrtnew = int(tmin*1000.)
    else:
        delrtnew = 0

    # Slice the data array and update the header
    if object.traces.ndim == 1:
        object.traces = object.traces[0, itmin:itmax+1]
        object.header[0]['ns'] = nsnew
        object.header[0]['delrt'] = delrtnew
    else:
        object.traces = object.traces[:, itmin:itmax+1]
        object.header[:]['ns'] = nsnew
        object.header[:]['delrt'] = delrtnew

def space_window(object, **options): #dobs, imin=0.0, imax=0.0, axis=0):
    """
    Window traces in space.

    :param dobs: input data to window
    :param imin: (optional) minimum value of trace to pass (=0)
    :param tmax: (optional) maximum value of trace to pass (=0)
    """

    # Get options
    vmin = options.get('vmin', 0)
    vmax = options.get('vmax', len(object.header))

    # Windowing
    object.header = object.header[vmin:vmax+1]
    object.traces = object.traces[vmin:vmax+1, :]
    for itrac in range(0, len(object.header)):
        object.header[itrac]['tracf'] = itrac+1
