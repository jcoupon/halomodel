# Makefile for halomodel
SHELL := /bin/bash

# compiler option
CC      = gcc
CFLAGS  = -fPIC -Wall -Wextra -O3 #-g
LDFLAGS =
RM      = rm -f
NAME    = halomodel
EXT     = so

# required libraries
PREFIX_FFTW   = # /usr/local
PREFIX_GSL    = # /usr/local
PREFIX_NICAEA = # $(HOME)/local/source/build/nicaea_2.7

ifneq ($(PREFIX_GSL), )
	GSL = $(PREFIX_GSL)
else
	GSL = /usr/local
endif

ifneq ($(PREFIX_FFTW), )
	FFTW = $(PREFIX_FFTW)
else
	FFTW = /usr/local
endif

ifneq ($(PREFIX_NICAEA), )
	NICAEA = $(PREFIX_NICAEA)
else
	NICAEA = $(HOME)/local/source/build/nicaea_2.7
endif

# FFTW   = /usr/local
# GSL    = /usr/local
# NICAEA = $(HOME)/local/source/build/nicaea_2.7

# source files
SRCS    = utils.c cosmo.c hod.c abundance.c lensing.c clustering.c xray.c
OBJS    = $(SRCS:.c=.o)

# extra headers
CFLAGS += -Iinclude -I$(FFTW)/include  -I$(GSL)/include -I$(NICAEA)/include
LFLAGS += -lm  -lfftw3 -lgsl -lgslcblas -L$(FFTW)/lib -L$(GSL)/lib -L$(NICAEA)/lib -lnicaea

LDFLAGS += -Wl,-rpath,$(NICAEA)/lib -Wl,-rpath,$(FFTW)/lib -Wl,-rpath,$(GSL)/lib

# if trouble with link to conda python library, run
# sudo install_name_tool -id /PATH/TO/anaconda/lib/libpythonx.x.dylib /PATH/TO/anaconda/lib/libpythonx.x.dylib

PYTHON_LIB=$(shell which python | sed "s/bin\/python/lib/")

# python interpreter
CFLAGS += $(shell python-config --cflags)
LFLAGS += $(shell python-config --ldflags) -L$(PYTHON_LIB)

# -L/Users/coupon/anaconda/lib

.PHONY: all
all: ./lib/lib$(NAME).$(EXT)

vpath %.h $(PWD)/include
vpath %.c $(PWD)/src

xray:  $(OBJS)
	$(CC)  $(CFLAGS) -o $@ $^ $(LFLAGS) $(LDFLAGS) -L.
	mv xray ./bin/xray

./lib/lib$(NAME).$(EXT): $(OBJS)
	$(CC)  $(CFLAGS) -shared -o $@ $^ $(LFLAGS) $(LDFLAGS) -L.

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

%.h:

.PHONY: clean
clean:
	-${RM}  ${OBJS}
