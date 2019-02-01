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
import matplotlib.pyplot as plt

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

    def image(self, key='tracl', bclip=None, wclip=None, clip=None, legend=0,
        label1=' ', label2=' ', title=' ', cmap='gray', style='normal',
        interpolation=None):
        """
        matplotlib.pyplot.imshow adapted to plot SU files

        :param key: header keyword (default tracl)
        :param bclip: data values outside of [bclip,wclip] are clipped
        :param wclip: data values outside of [bclip,wclip] are clipped
        :param clip: clip used to determine bclip and wclip
        :param legend: colorbar 0=no colorbar (default) 1=colorbar
        :param label1: x-axis label
        :param label2: y-axis label
        :param title: title of the image
        :param cmap: color map (defautl 'gray'): gray, jet, ...
        """

        # Check clip, bclip and wclip
        if(clip == None and bclip == None and clip == None):
            bclip = np.amin(self.traces)
            wclip = np.amax(self.traces)
        else:
            if(clip != None and bclip == None and wclip == None):
                bclip = -1.*clip
                wclip = clip

        # Get ns and dt from header
        ns = self.header[0]['ns']
        dt = float(self.header[0]['dt']/1000000.)
        if dt != 0:
            y0 = float(self.header[0]['delrt'])/1000.
            y1 = float(ns-1)*dt+y0
            x0 = self.header[0][key]
            x1 = self.header[-1][key]

        if self.header[0]['trid'] == 118:
            # Get d1
            d1 = float(self.header[0]['d1'])
            y0 = 0.
            y1 = float(ns-1)*d1
            x0 = self.header[0][key]
            x1 = self.header[-1][key]

        if self.header[0]['trid'] == 122:
            # Get d1
            d1 = float(self.header[0]['d1'])
            y0 = 0.
            y1 = float(ns-1)*d1
            # Get d2
            d2 = float(self.header[0]['d2'])
            x0 = float(self.header[0]['f2'])
            x1 = x0+float(len(self.header)-1)*d2

        if self.header[0]['trid'] == 132:
            # Get d1
            d1 = float(self.header[0]['d1'])
            y0 = float(self.header[0]['f1'])
            y1 = y0+float(ns-1)*d1
            # Get d2
            d2 = float(self.header[0]['d2'])
            x0 = float(self.header[0]['f2'])
            x1 = x0+float(len(self.header)-1)*d2

        if style == 'normal':
            # Add labels to axes
            plt.xlabel(label1)
            plt.ylabel(label2)

            # Add title to axis
            plt.title(title)

            # Plot surface
            plt.imshow(self.traces.swapaxes(1,0), aspect='auto', cmap=cmap,
                        extent=[x0, x1, y1, y0],
                        vmin=bclip, vmax=wclip, interpolation=interpolation)
        if style == 'masw':
            # Add labels to axes
            plt.xlabel(label1)
            plt.ylabel(label2)

            # Add title to axis
            plt.title(title)

            # Plot surface
            plt.imshow(self.traces, origin='bottom-left', aspect='auto', cmap=cmap,
                        extent=[y0, y1, x0, x1],
                        vmin=bclip, vmax=wclip, interpolation=interpolation)
        # Add legend
        if legend == 1:
            plt.colorbar()

    def wiggle(self, clip=-1., key='tracl', label1=' ', label2=' ', title=' ',
        tracecolor='black', tracestyle='-', skip=1, xcur=1):
        """
        Wiggle for SU files

        :param clip: clip used to determine outside values to be clipped [-clip, clip]
        :param key: header keyword (default tracl)
        :param label1: x-axis label
        :param label2: y-axis label
        :param title: title of the image
        :param tracecolor: color of the traces
        :param tracestyle: style of the traces ('--', ':', ...)
        :param skip: number of traces to skip for each plotted trace
        :param xcur: factor to increase trace amplitudes on output
        """

        # Get ns and dt from header
        ns = self.header[0]['ns']
        dt = float(self.header[0]['dt']/1000000.)
        ntrac = len(self.header)
        if dt != 0:
            y0 = float(self.header[0]['delrt'])/1000.
            y1 = float(ns-1)*dt+y0
            x0 = self.header[0][key]
            x1 = self.header[-1][key]
            d2 = 1.

        if self.header[0]['trid'] == 118:
            # Get d1
            d1 = float(self.header[0]['d1'])
            y0 = 0.
            y1 = float(ns-1)*d1
            x0 = self.header[0][key]
            x1 = self.header[-1][key]

        if self.header[0]['trid'] == 122:
            # Get d1
            d1 = float(self.header[0]['d1'])
            y0 = 0.
            y1 = float(ns-1)*d1
            # Get d2
            d2 = float(self.header[0]['d2'])
            x0 = float(self.header[0]['f2'])
            x1 = x0+float(len(self.header)-1)*d2

        # Add labels
        plt.xlabel(label1)
        plt.ylabel(label2)

        # Add axes
        plt.title(title)

        # Get the normalization parameter (for output)
        y = np.linspace(y0, y1, ns)
        if clip >= 0. :
            norm = clip
        else:
            norm = np.amax(np.abs(self.traces))

        # Plot the traces
        for itrac in range(0, ntrac, skip):
            wig = self.traces[itrac]/norm*d2*float(skip-1)*xcur
            plt.plot(wig+x0+float(itrac)*d2, y, color=tracecolor, linestyle=tracestyle)
