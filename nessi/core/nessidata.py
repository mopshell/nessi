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
import os
import sys
import numpy as np

class NessiData():
    """
    Class to handle seismic dataset. The data structure use a classic
    ``numpy`` array to store data and a ``numpy`` array with a custom
    datatype to store trace header values. The header array follows the
    Seismic Unix CWP (revision 0) structure. Further, the data structure
    embedded a history log (in text format) which stores all signal
    processing applyed to the data.
    """

    def __init__(self):
        """
        Initialize the NESSI datastructure.
        """

        # Initialize empty header and traces members
        self.header = []
        self.traces = []

        # Initialize history log member.
        self.history = '>> History log\n'

    def savehist(self, fname, path='.'):
        """
        Write the history log in a text file.

        :param fname: filename to write the history log without the .txt
            extension.
        :param fpath: (optional) path where the text file will be saved.
            By default, the file is saved in the current directory.
        """

        # Test if path exist
        # Test if file already exist

        # Open the history log text file to write
        fhist = open(path+'/'+fname, 'w')

        # Write history to history log text file
        fhist.write(self.history)

        # Close the history log text file
        fhist.close()

    def sutype():
        """
        Method which return the numpy custom data type for NESSI
        data structure. This data type follows the Seismic Unix CWP (revision 0)
        binary header structure.
        """
        sudtype = np.dtype([
            ('tracl', np.int32), ('tracr', np.int32), ('fldr', np.int32), \
            ('tracf', np.int32), ('ep', np.int32), ('cdp', np.int32), \
            ('cdpt', np.int32), ('trid', np.int16), ('nvs', np.int16), \
            ('nhs', np.int16), ('duse', np.int16), ('offset', np.int32), \
            ('gelev', np.int32), ('selev', np.int32), ('sdepth', np.int32), \
            ('gdel', np.int32), ('sdel', np.int32), ('swdep', np.int32), \
            ('gwdep', np.int32), ('scalel', np.int16), ('scalco', np.int16), \
            ('sx', np.int32), ('sy', np.int32), ('gx', np.int32), \
            ('gy', np.int32), ('counit', np.int16), ('wevel', np.int16), \
            ('swevel', np.int16), ('sut', np.int16), ('gut', np.int16), \
            ('sstat', np.int16), ('gstat', np.int16), ('tstat', np.int16), \
            ('laga', np.int16), ('lagb', np.int16), ('delrt', np.int16), \
            ('muts', np.int16), ('mute', np.int16), ('ns', np.uint16), \
            ('dt', np.uint16), ('gain', np.int16), ('igc', np.int16), \
            ('igi', np.int16), ('corr', np.int16), ('sfs', np.int16), \
            ('sfe', np.int16), ('slen', np.int16), ('styp', np.int16), \
            ('stas', np.int16), ('stae', np.int16), ('tatyp', np.int16), \
            ('afilf', np.int16), ('afils', np.int16), ('nofilf', np.int16), \
            ('nofils', np.int16), ('lcf', np.int16), ('hcf', np.int16), \
            ('lcs', np.int16), ('hcs', np.int16), ('year', np.int16), \
            ('day', np.int16), ('hour', np.int16), ('minute', np.int16), \
            ('sec', np.int16), ('timebas', np.int16), ('trwf', np.int16), \
            ('grnors', np.int16), ('grnofr', np.int16), ('grnlof', np.int16), \
            ('gaps', np.int16), ('otrav', np.int16), ('d1', np.float32),\
            ('f1', np.float32), ('d2', np.float32), ('f2', np.float32), \
            ('ungpow', np.float32), ('unscale', np.float32), ('ntr', np.int32), \
            ('mark', np.int16), ('shortpad', np.int16), \
            ('unassignedInt1', np.int32), ('unassignedInt2', np.int32), \
            ('unassignedInt3', np.int32), ('unassignedInt4', np.int32), \
            ('unassignedFloat1', np.float32), ('unassignedFloat2', np.float32), \
            ('unassignedFloat3', np.float32)])

        return sudtype
