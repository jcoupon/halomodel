# halomodel

Jean coupon - 2016
script to run wrapped halomodel routines in c

Required libraries:

for c (set path in Makefile if necessary):
- fftw3 3.3.4 (http://www.fftw.org/)
- gsl 2.1 (https://www.gnu.org/software/gsl/)
- nicaea 2.5 (http://www.cosmostat.org/software/nicaea/)

for python:
- numpy 1.10.2 (http://www.numpy.org/)
- scipy 0.17.1 (https://www.scipy.org/scipylib/download.html)
- (for tests only) astropy 1.2.1 (http://www.astropy.org/)

To install the library:

First edit the Makefile to specify the path to FFTW, GSL and NICAEA, then run
```
$ git clone https://github.com/jcoupon/halomodel.git
$ cd halomodel
$ make
```

Then:
- set "HALOMODEL_DIRNAME" as the path to halomodel in halomodel.py
- add the path to PYTHONPATH
- in python: "import halomodel"

To test the installation:
```
$ python
>>> import halomodel
>>> halomodel.test()
```
