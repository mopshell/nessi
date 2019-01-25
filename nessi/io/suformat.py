#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: suformat.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2019 Damien Pageot
# ------------------------------------------------------------------

"""
Support of Seismic Unix format.
"""

# Import modules
import numpy as np
import pandas as pd

# 1-Define NESSI data structure => Inspired from SU
# 2-Get data from SU header using pack/unpack (?)

def suread():
    """
    Read Seismic Unix files and store in NESSI data structure.
    """
