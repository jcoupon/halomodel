# Makefile template for shared library

# compiler option
CC      = gcc
CFLAGS  = -fPIC -Wall -Wextra -O3 #-g
LDFLAGS = -shared
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
CFLAGS +=  -Iinclude -I$(FFTW)/include  -I$(GSL)/include -I$(NICAEA)
LFLAGS += -lm  -lfftw3 -lgsl -lgslcblas -lm -L$(FFTW)/lib -L$(GSL)/lib -L$(NICAEA)/Demo -lnicaea

.PHONY: all
all: lib$(NAME).$(EXT)

vpath %.h include
vpath %.c src

test: $(NAME)_test
	LD_LIBRARY_PATH=. ./$(NAME)_test

$(NAME)_test: lib$(NAME).$(EXT)
	$(CC)  $(CFLAGS)  $(LFLAGS) src/test.c -o $@ -L. -l$(NAME)

lib$(NAME).$(EXT): $(OBJS)
	$(CC)  $(CFLAGS) $(LFLAGS) ${LDFLAGS} -o $@ $^

%.h:

.PHONY: clean
clean:
	-${RM} $(NAME)_test ${OBJS}
