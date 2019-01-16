# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [0.3.0] - YYYY-MM-DD

### Added
- linear source inversion *lsrcinv* in *nessi.signal*
- dispersion module *nessi.modeling.dispersion*
- Rayleigh wave velocity estimation *vrayleigh* in *nessi.modeling.dispersion*

## [0.2.2] - 2018-12-05

### Modified
- vectorization of FD functions in *swm_deriv.f90*
- loop vectorization in *swm_marching.f90*

## [0.2.1] - 2018-10-12

### Added
- test functions in *test_swarm* for *update* method with *pupd* parameter
- test functions in *test_swarm* for *fiupdate* method

### Modified
- test suite for particle swarm optimization (attempted output update)
- *masw* method of SUdata() has now the *whitening* option

## [0.2.0] - 2018-08-16

### Added
- *resamp* method for SUdata()
- *mute* method for SUdata()
- *specfx* method for SUdata()
- *specfk* method for SUdata()
- *wiggle* method for SUdata()
- Standard Genetic Algorithm implementation (mu, pc, pm)
- Centroidal Voronoi tessellation option (McQueen algorithm) in *Swarm.init_particles*
- *dispick*: a simple dispersion diagram picking method to get the *effective* dispersion curve
- *suread* function (SUdata module) which declare SU object and read a SU file in the same time
- *sucreate* function (SUdata module) which declare SU object and create a SU file in the same time
- *grdinterp* functions for coarse to fine grid interpolation in modbuilder (cython reimplementation)
- *fiupdate* in *Swarm()* for fully informed PSO update
- *_get_neighbors* private function in *Swarm()* for the FIPS update
- *pupd* parameter update probability to all PSO update methods

### Modified
- SU header structure now fits SU/CWP format (before SEG-Y rev 1)
- remove useless SUdata variables
- *image* method of SUdata() to handle *masw*, *specfx* and *specfk* method outputs
- *image* has now 'normal' and 'masw' style options (style=)
- *image* has now interpolation option (see matplotlib.pyplot.imshow)
- docstrings for SUdata methods
- *masw* method output is now a SU file
- *ringx*: ring topology excluding the particle to update for Swarm.update()
- *toroidalx*: toroidal topology excluding the particle to update for Swarm.update()
- *init_particle*: force initialization to zero for arrays *current*, *velocity*, *history* and *misfit*
- *grd_ds1* was renamed *grd_sib* (interp2d) *sibson1* and *sibson2* removed and replaced by *sibson* function only
- *_power2* function is Genalg class for better performances

### Fixed
- *kill* method for SUdata()
- use of header keywords in *masw* method for SUdata()
- add some SU header keyword initializations in *create* for SUdata()

### Removed
- *sibson2* (and *grd_ds2.c*) removed because useless (only *sibson1* now called *sibson* is kept)
- *examples* folder is useless (examples in nessi.material repository)

## [0.1.2] - 2018-06-05

### Added
- *CHANGELOG.md* in docs/
- *Interface with Geopsy-gpdc* in docs/
- *Particle Swarm Optimization: basics* in docs/
- *Read, write and create SU data* tutorial in docs/
- *Windowing SU data* tutorial in docs/
- *Tapering SU data* tutorial in docs/
- **lib** folder for future C/Fortran libraries
- *nessi.modeling.swm references* in docs/
- *nessi.globopt references* in docs
- *MASW from SU data* tutorial in docs/
- *masw* method in *SU/CWP references* chapter in docs/
- *Dispersion curve inversion using GPDC and PSO* tutorial in docs/

### Modified
- add __ZENODO__ DOI in the README.md
- interfaces references documentation
- *Seismic modeling example* now in tutorials
- *SU/CWP references* now in *NeSSI API* chapter
- *Getting started* chapter in docs/

### Fixed
- *wind* method of SUdata() can now apply a window in time and space at the same time.
- *pfilter* method of SUdata() can now handle axis 0 and 1
- *masw* method of SUdata() can now return freq and vel arrays as expected
- *masw* method of SUdata() now handles correctly the *scalco* keyword
- obselete dependencies

## Removed
- *2D seismic modeling* part in docs/ (temporary)

## [0.1.1] - 2018-06-01

### Added
- *test_windowing.py* in nessi/signal/tests
- *test_tapering.py* in nessi/signal/tests
- *test_filtering.py* in nessi/signal/tests

### Modified
- README.md

### Fixed
- remake html documentation
- *signal.time_window* issue for 1D signal
- *signal.taper1d* issue for 1D signal

## [0.1.0] - 2018-05-31

### Added
- CONTRIBUTE.md
- *getting started* chapter in docs/
- *interfaces* chapter in docs/
- *gprMax* paragraph in *interfaces* (docs/)
- *interfaces references* chapter in docs
- *geopsy-gpdc* paragraph in *interfaces* (docs/)
- geopsy-gpdc interface in nessi.modeling.interfaces
- *test_gpdcwrap.py* in nessi/modeling/tests
- *NeSSI Global Optimization* references (docs/)
- *test_swarm.py* in nessi/globopt/tests
- *SU/CWP references* (docs/)
- time_window function in nessi.signal windowing.py
- space_window function in nessi.signal windowing.py
- taper1d function in nessi.signal tapering.py
- sine and cosine taper types
- sin2filter polynomial filter nessi.signal filtering.py
- pfilter method in nessi.io for SU data format
- adding taper method in nessi.io for SU data format
- adding kill method (zero out traces) in nessi.io for SU data format
- add *modbuilder* module (perspectives)

### Modified
- README.md
- nessi.swm functions are now located at nessi.modeling.swm
- nessi.pso.Swarm class is now callable from nessi.globopt.Swarm
- adding window in space to suwind method in nessi.io for SU data format
- wind method (SU format) now depends nessi.signal.time_window and nessi.signal.space_window
- all *__init__.py headers*
- all python file headers (docstrings...)
- all Fortran source code headers (infos)
- rename *grd* module in *interp2d* and move it in *modbuilder* module

### Fixed
- *get_gbest* method of the Swarm class (nessi.globopt), option *ring*

## [0.0.0] - 2018-05-21
- starting point for development of the true first version (0.1.0); not a valid version.

## [X.X.X] - YYYY-MM-DD
### Added
### Fixed
### Removed
