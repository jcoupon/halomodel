# halomodel

Jean coupon - 2016
script to run wrapped halomodel routines in c

To install the library:

```
$ git clone https://github.com/jcoupon/halomodel.git
$ cd halomodel
$ make
$ halomodel.py test
```

Then:
- set "HALOMODEL_DIRNAME" as the path to halomodel in halomodel.py
- add the path to PYTHONPATH
- in python "import halomodel"

Requirements:

for c (set path in Makefile if necessary):
- nicaea 2.5 (http://www.cosmostat.org/software/nicaea/)
- fftw3 3.3.4 (http://www.fftw.org/)
- gsl 2.1 (https://www.gnu.org/software/gsl/)

for python:
- numpy 1.10.2 (http://www.numpy.org/)
- scipy 0.17.1 (https://www.scipy.org/scipylib/download.html)
- (for tests only) astropy 1.2.1 (http://www.astropy.org/)
