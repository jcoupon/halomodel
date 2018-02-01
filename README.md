# halomodel

Jean coupon - 2016 - 2018
Library to run halomodel routines

Required libraries:

for c (set path in Makefile if necessary):
- fftw3 3.3.4 (http://www.fftw.org/)
- gsl 2.1 (https://www.gnu.org/software/gsl/)
- nicaea 2.7 (http://www.cosmostat.org/software/nicaea/)

for python:
- numpy 1.10.2 (http://www.numpy.org/)
- scipy 0.17.1 (https://www.scipy.org/scipylib/download.html)
- astropy 1.2.1 (http://www.astropy.org/)

To install the library:

run
```
$ git clone https://github.com/jcoupon/halomodel.git
$ cd halomodel
$ make
```

Options:

- `PREFIX_GSL=DIRECTORY` (default: /usr/local)
- `PREFIX_FFTW=DIRECTORY` (default: /usr/local)
- `PREFIX_NICAEA=DIRECTORY` (default: $(HOME)/local/source/build/nicaea_2.7)


Then:
- add the halomodel path to PYTHONPATH
- in python: "import halomodel"

To test the installation:
```
$ python
>>> import halomodel
>>> halomodel.test()
```
