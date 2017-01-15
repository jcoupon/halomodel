/* ------------------------------------------------------------ *
 * clustering.h                                                 *
 * halomodel library                                            *
 * Jean Coupon 2016	                                           *
 * ------------------------------------------------------------ */

#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "hod.h"
#include "abundance.h"
#include "cosmo.h"

/* ---------------------------------------------------------------- *
 * two-point correlation function
 * ---------------------------------------------------------------- */
void wOfTheta(const Model *model, double *theta, int N, double z, int obs_type, double *result);

void xi_gg(const Model *model, double *r, int N, double z, int obs_type, double *result);
void wOfThetaAll(const Model *model, double *theta, int N, double z, double *result);

void xi_gg_censat(const Model *model, double *r, int N, double z, double *result);
double intForxi_gg_censat(double logMh, void *p);

void xi_gg_satsat(const Model *model, double *r, int N, double z, double *result);
double intForxi_gg_satsat(double k, void *p);
double P_gg_satsat(const Model *model, double k, double z);
double intForP_gg_satsat(double logMh, void *p);

void xi_gg_twohalo(const Model *model, double *r, int N, double z, double *result);
double intForxi_gg_twohalo(double k, void *p);
double P_gg_twohalo(double k, void *p);
double intForP_gg_twohalo(double logMh, void *p);



#endif
