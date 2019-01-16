import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float DTYPE_f

@cython.boundscheck(False)
@cython.wraparound(False)


def modext2(np.ndarray[float, ndim=2, mode='c'] v, int npml):
  """
  Extend a physical parameter model with PML bands.

  :param v: 2D float numpy array containing the physical parameter model.
  :param npml: size, in points, of the PML bands.
  """

  cdef Py_ssize_t i, i1, i2

  # Get dimenions
  cdef int n1 = v.shape[0]
  cdef int n2 = v.shape[1]
  cdef int n1e = n1+2*npml
  cdef int n2e = n2+2*npml

  # Allocate extended model
  cdef np.ndarray[float, ndim=2, mode='c'] ve = np.zeros((n1e, n2e), dtype=np.float32)

  # Fill the extended model
  #ve[npml:n1+npml, npml:n2+npml] = v[:, :]
  for i2 in range(0, n2):
    for i1 in range(0, n1):
      ve[npml+i1, npml+i2] = v[i1, i2]

  # Fill the lateral model extensions with boundary values
  for i1 in range(0, n1e):
    for i in range(0, npml):
      ve[i1, i] = ve[i1, npml]
      ve[i1, n2+npml+i] = ve[i1,n2+npml-1]

  # Fill the top and bottom model extensions with boundary values
  for i2 in range(0, n2e):
    for i in range(0, npml):
      ve[i, i2] = ve[npml, i2]
      ve[n1+npml+i, i2] = ve[n1+npml-1,i2]

  return ve

def modbuo2(np.ndarray[float, ndim=2, mode='c'] roe):
  """
  Calculate buyonacy from density.

  :param roe: the extended density parameter model.
  """

  cdef Py_ssize_t i1, i2

  # Get dimenions
  cdef int n1e = roe.shape[0]
  cdef int n2e = roe.shape[1]

  # Declare arrays
  cdef np.ndarray[float, ndim=2, mode='c'] bux = np.zeros((n1e, n2e), dtype=np.float32)
  cdef np.ndarray[float, ndim=2, mode='c'] buz = np.zeros((n1e, n2e), dtype=np.float32)

  # Calculate bux
  for i2 in range(0, n2e-1):
    for i1 in range(0, n1e):
      bux[i1, i2] = 0.5*((1./roe[i1,i2])+(1./roe[i1,i2+1]))
      bux[i1, n2e-1] = 1./roe[i1, n2e-1]

  # Calculate buz
  for i2 in range(0, n2e):
    for i1 in range(0, n1e-1):
      buz[i1, i2] = 0.5*((1./roe[i1, i2])+(1./roe[i1+1,i2]))
      buz[n1e-1, i2] = 1./roe[n1e-1, i2]

  return bux, buz

def modlame2(np.ndarray[float, ndim=2, mode='c'] vpe, np.ndarray[float, ndim=2, mode='c'] vse, np.ndarray[float, ndim=2, mode='c'] roe):
  """
  Calculate the Lamé's parameters from Vp, Vs and density.

  :param vpe: extended P-wave velocity model
  :param vse: extended S-wave velocity model
  :param roe: extended density model
  """

  cdef Py_ssize_t i1, i2

  # Get dimenions
  cdef int n1e = vpe.shape[0]
  cdef int n2e = vpe.shape[1]

  # Declare arrays
  cdef np.ndarray[float, ndim=2, mode='c'] mu = np.zeros((n1e, n2e), dtype=np.float32)
  cdef np.ndarray[float, ndim=2, mode='c'] lbd = np.zeros((n1e, n2e), dtype=np.float32)
  cdef np.ndarray[float, ndim=2, mode='c'] lbdmu = np.zeros((n1e, n2e), dtype=np.float32)

  # Declare local array
  cdef np.ndarray[float, ndim=2, mode='c'] mu0 = np.zeros((n1e, n2e), dtype=np.float32)

  # Calculate mu0
  for i2 in range(0, n2e):
    for i1 in range(0, n1e):
      mu0[i1, i2] = vse[i1, i2]*vse[i1, i2]*roe[i1, i2]

  for i2 in range(0, n2e-1):
    for i1 in range(0, n1e-1):
      mu[i1, i2] = 1./((1./4.)*(1./mu0[i1, i2]+1./mu0[i1+1, i2]+1./mu0[i1, i2+1]+1./mu0[i1+1, i2+1]))

  for i2 in range(0, n2e):
    mu[n1e-1, i2] = mu[n1e-2, i2]

  for i1 in range(0, n1e):
    mu[i1, n2e-1] = mu[i1, n2e-2]

  for i2 in range(0, n2e):
    for i1 in range(0, n1e):
      # Calculate lbd
      lbd[i1, i2] = vpe[i1, i2]*vpe[i1, i2]*roe[i1, i2]-2.*mu0[i1, i2]
      # Calculate lbdmu
      lbdmu[i1, i2] = lbd[i1, i2]+2.*mu0[i1, i2]

  return mu, lbd, lbdmu
