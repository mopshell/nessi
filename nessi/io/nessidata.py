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
        dheader = [{
                    # Identification
                    'id':np.nan,     # data identification code
                    # Timestamp
                    'date':np.nan,   # Date in YYYYMMDD format
                    'time':np.nan,   # Time in HHMMSS format
                    # Time domain data
                    'ns':np.nan,     # Number of time samples
                    'dt':np.nan,     # Time sampling
                    'delrt':np.nan,  # Delta time
                    # Data size if not time domain
                    'n1':np.nan,     # number of samples in the first dimension
                    'd1':np.nan,     # sampling in the first dimension
                    'f1':np.nan,     #
                    'n2':np.nan,     # number of sample in the second dimension
                    'd2':np.nan,     # sampling in the second dimension
                    'f2':np.nan,     #
                    # Acquisition
                    'sx':np.nan,     # source coordinate X
                    'sy':np.nan,     # source coordinate Y
                    'sz':np.nan,     # source coordinate Z (elevation)
                    'gx':np.nan,     # receiver coordinate X
                    'gy':np.nan,     # receiver coordinate Y
                    'gz':np.nan,     # receiver coordinate Z (elevation)
                    },]

        # Create header
        self.header = pd.DataFrame(dheader)

        # Create data array
        self.traces = []
