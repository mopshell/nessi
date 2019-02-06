#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: test_stream_main.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2019 Damien Pageot
# ------------------------------------------------------------------
"""
Test suite for the main methods of the Stream class.

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np
from nessi.core import Stream


def test_stream_create_1d():
    """
    Test the ``create`` method of the Stream class for 1D data.
    """

    # Create one-dimensionnal data of size 'ns'
    ns = 256
    data = np.ones(ns)


    # Create a new Stream object
    object = Stream()

    # Create SU-like data structure from 'data' without options.
    object.create(data)

    # Testing the stream object members initialization
    np.testing.assert_equal(object.header['trid'], 1)
    np.testing.assert_equal(object.header['tracl'], 1)
    np.testing.assert_equal(object.header['tracr'], 1)
    np.testing.assert_equal(object.header['tracf'], 1)
    np.testing.assert_equal(object.header['ns'], ns)
    np.testing.assert_equal(object.header['dt'], 0.01*1000000.)

def test_stream_create_2d():
    """
    Test the ``create`` method of the Stream class for 2D data.
    """

    # Create two-dimensionnal data of size 'ns'x'nr'
    ns = 256
    nr = 64
    data = np.ones((nr, ns))


    # Create a new Stream object
    object = Stream()

    # Create SU-like data structure from 'data' without options.
    object.create(data)

    # Testing the stream object members initialization
    np.testing.assert_equal(object.header[:]['trid'], np.ones(nr, dtype=np.int16))
    np.testing.assert_equal(object.header[:]['tracl'], np.linspace(1, nr, nr, dtype=np.int32))
    np.testing.assert_equal(object.header[:]['tracr'], np.linspace(1, nr, nr, dtype=np.int32))
    np.testing.assert_equal(object.header[:]['tracf'], np.linspace(1, nr, nr, dtype=np.int32))
    np.testing.assert_equal(object.header[0]['ns'], ns)
    np.testing.assert_equal(object.header[0]['dt'], 0.01*1000000.)

if __name__ == "__main__" :
    np.testing.run_module_suite()
