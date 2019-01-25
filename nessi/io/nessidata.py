#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: nessidata.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2019 Damien Pageot
# ------------------------------------------------------------------

"""
The NESSI data structure.
"""

# Import modules
import numpy as np
import pandas as pd

class DataStruct():

    def __init__(self):
        """
        Initialize NESSI data structure.
        """

        # Dictionnary header
        dheader = [{'n1':np.nan,   # number of samples in the first dimension
                    'd1':np.nan,   # sampling in the first dimension
                    'n2':np.nan,   # number of sample in the second dimension
                    'd2':np.nan,   # sampling in the second dimension
                    'trid':np.nan, # trace identification code
                    'sx':np.nan,   # source coordinate X
                    'sy':np.nan,   # source coordinate Y
                    'sz':np.nan,   # source coordinate Z (elevation)
                    'gx':np.nan,   # receiver coordinate X
                    'gy':np.nan,   # receiver coordinate Y
                    'gz':np.nan,   # receiver coordinate Z (elevation)
                    },]
