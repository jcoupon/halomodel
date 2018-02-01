/* ------------------------------------------------------------ *
 * abundance.h                                                  *
 * halomodel library                                            *
 * Jean Coupon 2016	                                           *
 * ------------------------------------------------------------ */

#ifndef ABUNDANCE_H
#define ABUNDANCE_H

#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "hod.h"
#include "cosmo.h"

/* ---------------------------------------------------------------- *
 * Stellar mass function
 * ---------------------------------------------------------------- */

void dndlog10Mstar(const Model *model, double *log10Mstar, int N, double z, int obs_type, double *result);

double logM_lim(const Model *model, double r, double c, double z, int obs_type);
double ngal_triax(const Model *model, double r, double c, double z, double obs_type);
double ngal_den(const Model *model, double logMh_max, double log10Mstar_min, double log10Mstar_max, double z, int obs_type);
double int_for_ngal_den(double lnMh, void *p);

/* ---------------------------------------------------------------- *
 * Number-weighted quantities
 * ---------------------------------------------------------------- */


double MstarMean(const Model *model, double z, int obs_type);
double intForMstarMean(double log10Mstar, void *p);

#endif
