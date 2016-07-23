/* ------------------------------------------------------------ *
 * lensing.h                                                    *
 * halomodel library                                            *
 * Jean Coupon 2016	                                           *
 * ------------------------------------------------------------ */

#ifndef LENSING_H
#define LENSING_H

#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "hod.h"
#include "abundance.h"
#include "cosmo.h"

void DeltaSigma(const Model *model, double *R, int N, double z, int obs_type, double *result);
double intForDeltaSigma(double logR, void *p);

void Sigma(const Model *model, double *R, int N, double z, int obs_type, double *result);
double intForSigma(double logz, void *p);

void DeltaSigmaStar(const Model *model, double *r, int N, double z, double *result);
double intForDeltaSigmaStar(double log10Mstar, void *p);
void DeltaSigmaAll(const Model *model, double *R, int N, double z, double *result);

void xi_gm_cen(const Model *model, double *r, int N, double z, double *result);
double intForxi_gm_cen(double logMh, void *p);

void xi_gm_sat(const Model *model, double *r, int N, double z, double *result);
double intForxi_gm_sat(double k, void *p);

double P_gm_satsat(const Model *model, double k, double z);
double intForP_gm_satsat(double logMh, void *p);

void xi_gm_twohalo(const Model *model, double *r, int N, double z, double *result);
double intForxi_gm_twohalo(double k, void *p);
double P_gm_twohalo(double k, void *p);

double intForP_twohalo_g(double logMh, void *p);
double intForP_twohalo_m(double logMh, void *p);


#endif
