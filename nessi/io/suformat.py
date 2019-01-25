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
import os
import sys
import numpy as np
import pandas as pd
from .nessidata import DataStruct

def _check_endian(fname):
    """
    Check endianess (little or big)
    """
    file = open(fname, 'rb')
    btmp = file.read()
    bsize = os.stat(fname).st_size
    nsl = np.frombuffer(btmp, dtype='<h', count=1, offset=114)[0]
    nsb = np.frombuffer(btmp, dtype='>h', count=1, offset=114)[0]
    if(bsize%((nsl*4)+240) == 0):
        endian = 'l'
    else:
        if(bsize%((nsb*4)+240) == 0):
            endian = 'b'
        else:
            sys.exit("Unable to read "+fname+"\n")
    file.close()

    return endian

def _sutype():
    """
    Numpy custom data type for Seismic Unix files.
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

def suread(fname):
    """
    Read Seismic Unix files and store in NESSI data structure.

    :param fname: Seismic Unix filename and path
    """

    # Initialize NESSI data structure
    sudata = DataStruct()

    # Check endianess
    endian = _check_endian(fname)

    # Open the file to read
    sufile = open(fname, 'rb')

    # Endianess parameters
    if endian == 'b': # Big endian
        sudtype = _sutype().newbyteorder()
        npdtype = '>f4'
    if endian == 'l': # Little endian
        sudtype = _sutype()
        npdtype = '<f4'

    # Get the header of the first trace (240 lenght)
    bhdr = sufile.read(240)

    # Read the header of the first trace
    hdr = np.frombuffer(bhdr, dtype=sudtype, count=1)[0]

    # Calculate the lenght of traces in bites
    btrc = sufile.read(hdr['ns']*4)

    # Close SU file
    sufile.close()

    # Convert first trace to NESSI data structure
    if hdr['trid'] != 0:
        sudata.header['id'][0] = hdr['trid']
    else:
        sudata.header['id'][0] = 10 # Seismic trace (default)

    # Time domain data header values
    sudata.header['ns'] = hdr['ns']
    sudata.header['dt'] = hdr['dt']
    sudata.header['delrt'] = hdr['delrt']
    # Acquisition
    scalco = hdr['scalco']
    if scalco == 0:
        fscalco = 1.
    if scalco < 0:
        fscalco = -1./np.float(scalco)
    if scalco > 0:
        fscalco = np.float(scalco)
    scalel = hdr['scalel']
    if scalel == 0:
        fscalel = 1.
    if scalel < 0:
        fscalel = -1./np.float(scalel)
    if scalel > 0:
        fscalel = np.float(scalel)
    sudata.header['sx'] = hdr['sx']*fscalco
    sudata.header['sy'] = hdr['sy']*fscalco
    sudata.header['sz'] = hdr['selev']*fscalel
    sudata.header['gx'] = hdr['gx']*fscalco
    sudata.header['gy'] = hdr['gy']*fscalco
    sudata.header['gz'] = hdr['gelev']*fscalel

    return sudata
