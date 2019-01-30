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
Module for handling seismic dataset.
"""

# Import modules
import numpy as np

class DataStruct():

    def __init__(self):
        """
        Initialize NESSI data structure.
        """

        # Dictionnary header
        nessidtype = np.dtype([
                    # Identification
                    ('id', np.int16),       # data identification code
                    # Time domain data
                    ('ns', np.int32),       # Number of time samples
                    ('dt', np.float32),     # Time sampling
                    ('delrt', np.float32),  # Delta time
                    # Data size if not time domain
                    ('n1', np.int16),       # number of samples in the first dimension
                    ('d1', np.float32),     # sampling in the first dimension
                    ('f1', np.float32),     #
                    ('n2', np.int16),       # number of sample in the second dimension
                    ('d2', np.float32),     # sampling in the second dimension
                    ('f2', np.float32),     #
                    # Acquisition
                    ('sx', np.float32),     # source coordinate X
                    ('sy', np.float32),     # source coordinate Y
                    ('sz', np.float32),     # source coordinate Z (elevation)
                    ('gx', np.float32),     # receiver coordinate X
                    ('gy', np.float32),     # receiver coordinate Y
                    ('gz', np.float32)      # receiver coordinate Z (elevation)
                    ])

        # Original path if exist
        self.filepath = ' '

        # Original format
        self.fileformat = ' '

        # Processing history
        self.history = ' '

        # Create header
        self.header = np.zeros(1, dtype=nessidtype)

        # Create data array
        self.traces = np.zeros(1, dtype=np.float32)
