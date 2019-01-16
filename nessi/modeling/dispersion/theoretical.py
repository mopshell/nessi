#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: theoretical.py
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2019 Damien Pageot
# ------------------------------------------------------------------
"""
Theoretical Love and Rayleigh wave dispersion calculation.

:copyright:
    Damien Pageot (nessi.develop@protonmail.com)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def Tmatrix(nu, vs, ro, z):
    """
    Calculate the Tn matrix
    """
    T = np.zeros((2, 2), dtype=np.complex64)
    T[0, 0] = nu*ro*np.exp(nu*z)
    T[0, 1] = -np.exp(nu*z)/(vs*vs)
    T[1, 0] = nu*ro*np.exp(-nu*z)
    T[1, 1] = 1./(vs*vs)*np.exp(-nu*z)
    return 1./(2.*nu*ro)*T

def Gmatrix(nu, vs, ro, zl):
    """
    Calculate the G matrix
    """
    G = np.zeros((2, 2), dtype=np.complex64)
    A = nu*zl
    mu = ro*vs*vs
    G[0, 0] = np.cosh(A) #(1.+np.exp(-2.*A))/2. #np.exp(A)*(1.+np.exp(-2.*A))/2.
    G[0, 1] = 1./(nu*mu)*np.sinh(A) #1./(nu*mu)*(1.-np.exp(-2.*A))/2. #np.exp(A)*(1.-np.exp(-2.*A))/2.
    G[1, 0] = nu*mu*np.sinh(A) #1./(nu*mu)*np.sinh(A) #1./(nu*mu)*(1.-np.exp(-2.*A))/2. #np.exp(A)*(1.-np.exp(-2.*A))/2.
    G[1, 1] = G[0, 0] #np.exp(A)*(1.+np.exp(-2.*A))/2.
    return G

class Disp():
    """
    Theoretical dispersion curve calculation.
    """

    def __init__(self):
        """
        Initialization
        """
        self.initialization = 1


    def initmodel(self, nl):
        """
        Initialize model definition arrays.

        :param nl: number of layers
        """

        # Initialize arrays
        self.vp = np.zeros(nl, dtype=np.float32)
        self.vs = np.zeros(nl, dtype=np.float32)
        self.ro = np.zeros(nl, dtype=np.float32)
        self.hl = np.zeros(nl, dtype=np.float32)


    def initcurve(self, fmin, fmax, nf, vmin, vmax):
        """
        Initialize dispersion curve parameters.

        :param fmin: minimum frequency
        :param fmax: maximum frequency
        :param nf: number of frequencies
        :param vmin: minimum velocity
        :param vmax: maximum velocity
        """

        self.freq = np.linspace(fmin, fmax, nf)
        self.vmin = vmin
        self.vmax = vmax
        self.curveinit = 1


    def initdiag(self, fmin, fmax, nf, vmin, vmax, nv):
        """
        Initialize dispersion diagram parameters.

        :param fmin: minimum frequency
        :param fmax: maximum frequency
        :param nf: number of frequencies samples
        :param vmin: minimum velocity
        :param vmax: maximum velocity
        :param nv: number of velocity samples
        """

        self.freq = np.linspace(fmin, fmax, nf)
        self.vel = np.linspace(vmin, vmax, nv)
        self.diaginit = 1

    def lovediag(self):
        """
        Calculate theoretical Love wave dispersion diagram.
        """

        # Initializations
        nw = len(self.freq)
        nv = len(self.vel)
        nl = len(self.vp)

        # Initialize dispersion diagram array
        self.diagram = np.zeros((nw, nv), dtype=np.float32)

        # Loop over frequencies
        for iw in range(0, nw):
            # Loop over velocities
            for iv in range(0, nv):
                # Loop over layers
                for il in range(0, nl):
                    # Wavenumbers
                    kt = (2.*np.pi*self.freq[iw]/self.vs[nl-il-1])
                    ke = (2.*np.pi*self.freq[iw]/self.vel[iv])
                    # Delay
                    nu = complex(ke*ke-kt*kt, 0.)
                    if il == 0: #Bottom layer
                        R = Tmatrix(np.sqrt(nu), self.vs[nl-il-1], self.ro[nl-il-1], self.hl[nl-il-1])
                    else:
                        G = Gmatrix(np.sqrt(nu), self.vs[nl-il-1], self.ro[nl-il-1], self.hl[nl-il-1])
                        R = np.matmul(R, G)
                # Store result in dispersion diagram array
                self.diagram[iw, iv] = np.real(R[1, 0])

    def rayleighdiag(self):
        """
        Calculate theoretical Rayleigh dispersion diagram.
        """

        # Initializations
        nw = len(self.freq)
        nv = len(self.vel)
        nl = len(self.vp)

        # Initialize dispersion diagram array
        self.diagram = np.zeros((nw, nv), dtype=np.float32)
