# Makefile template for shared library
SHELL := /bin/bash

# compiler option
CC      = gcc
CFLAGS  = -fPIC -Wall -Wextra -O3 #-g
LDFLAGS =
RM      = rm -f
NAME    = halomodel
EXT     = so

FFTW   = /usr/local
GSL    = /usr/local
NICAEA = $(HOME)/local/source/build/nicaea_2.5

# source files
SRCS    = utils.c cosmo.c hod.c abundance.c lensing.c clustering.c xray.c
OBJS    = $(SRCS:.c=.o)

# extra headers
CFLAGS += -Iinclude -I$(FFTW)/include  -I$(GSL)/include -I$(NICAEA)
LFLAGS += -lm  -lfftw3 -lgsl -lgslcblas -L$(FFTW)/lib -L$(GSL)/lib -L$(NICAEA)/Demo -lnicaea

LDFLAGS += -Wl,-rpath,$(NICAEA)/Demo

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
