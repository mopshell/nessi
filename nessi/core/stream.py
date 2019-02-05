#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: stream.py
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
import copy
import numpy as np

from nessi.graphics import ximage, xwigg
from nessi.signal import stack as sustack
from nessi.signal import time_window

class Stream():
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
        Initialize the NESSI data structure. The data structure is divided in
        header, traces, and data processing history. Additionnaly, object
        variable exist to store the path of the original file and the file
        format to access to additionnal informations if needed.
        """

        # This data type follows the Seismic Unix CWP (revision 0)
        # binary header structure.
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

        # Orignal file
        self.origin = ''
        self.format = ''

        # Initialize empty header and traces members
        self.header = np.zeros(1, dtype=sudtype)
        self.traces = np.zeros(1, dtype=np.float32)

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


        # Open the history log text file to write
        fhist = open(path+'/'+fname, 'w')

        # Write history to history log text file
        fhist.write(self.history)

        # Close the history log text file
        fhist.close()

    def create(self, data, **options):
        """
        Create a stream object from a numpy array. The recommanded parameters
        are ``trid`` and ``dt`` for time data and ``trid``, ``n1`` and ``d1``
        for others. If no optional parameter is given, the header is filled with
        the default parameters:
        ``dt = 0.04 ms``
        ``trid = 1 (seismic data)``
        If ``trid`` is given for non time data but ``d1`` and/or ``d2`` not
        given:
        ``d1 = 1``
        ``d2 = 1``

        By default, for trid=1,  ``ns = np.size(data, axis=1)`` and
        ``n1 = np.size(data, axis=1)`` for trid !=0.

        Trace dependant keyword values can be set afterward.

        :param data: numpy array containing the data.
        :param trid: trace identification (default 1, seismic data)
        :param dt: time sampling if trid=1 (default=400 µs)
        :param d1: for trid != 1 (default=1)
        :param d2: for trid != 1 (default=1)
        """

        # Get array size
        if np.ndim(data) == 1:
            ntrac = 1
            ns = len(data)
        if np.ndim(data) == 2:
            ntrac = np.size(data, axis=0)
            ns = np.size(data, axis=1)

        # Create a new Stream object
        #object = Stream()
        self.header.resize(ntrac)
        self.traces.resize((ntrac, ns))

        # Get trace identification code
        trid = options.get('trid', 1)
        self.header[:]['trid'] = trid
        if trid == 1:
            # Get time sampling
            dt = options.get('dt', 400)
            self.header[:]['dt'] = dt
        if trid != 1:
            # Get sampling in the 1st and 2nd dimensions
            d1 = options.get('d1', 1)
            d2 = options.get('d2', 1)
            self.header[:]['d1'] = d1
            self.header[:]['d2'] = d2

        # Fill traces
        if ntrac == 1:
            self.traces[0, :] = data[:]
        else:
            self.traces[:, :] = data[:, :]

    def write(self, fname, path='.'):
        """
        Write the stream object on disk as a Seismic Unix CWP file (rev.0).

        :param fname: output file name without the ``.su`` extension.
        :param path: path to write the file (default is the current directory)
        """

        # Test if path exist


        # Open file to write
        sufile = open(path+'/'+fname+'.su', 'wb')

        # Get the number of traces
        ntrac = len(self.header)

        # Loop over traces
        for itrac in range(0, ntrac):
            sufile.write(self.header[itrac])
            sufile.write(self.traces[itrac, :])

        # Close file
        sufile.close()

    def copy(self):
        """
        Return a copy of the Stream object.
        """
        return copy.deepcopy(self)

    def wind(self, type='time', **options):
        """
        Windowing traces in time or space.

        :param type: 'time' (default) or 'space' windowing.
        :param vmin: minimum value to pass (in time or space)
        :param vmax: maximum value to pass (in time or space)
        """

        if type == 'time':
            time_window(self, **options)
        else:
            print('space window')

    def image(self, **options):
        ximage(self, **options)

    def wiggle(self, **options):
        xwigg(self, **options)

    def stack(self, **options):
        sustack(self, **options)
