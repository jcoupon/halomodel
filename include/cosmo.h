/*
 *    cosmo.h
 *    halomodel library
 *    Jean Coupon 2016
 */

#ifndef COSMO_H
#define COSMO_H


#include <stdio.h>
#include <math.h>

#include "utils.h"
#include "cosmo.h"

/*
 *    cosmological parameters
 */

cosmo *initCosmo(const Model *model);

/*
 *    Angular correlation function of dark matter
 */

void xi_m(const Model *model, double *r, int N, double z, double *result);
double intForxi_m(double k, void *params);

/*
 *    halo bias
 */

double bias_h(const Model *model, double Mh, double z);

/*
 *    halo mass function
 */

double dndlnMh(const Model *model, double z, double Mh);

/*
 *    halo profile functions
 */

double uHalo(const Model *model, double k, double Mh, double c, double z);

double intForUHalo(double r, void *p);
double uHaloClosedFormula(const Model *model, double k, double Mh, double c, double z);

double rhoHalo(const Model *model, double r, double Mh, double c, double z);
double NFW(double r, double r_s, double rho_s);
double concentration(const Model *model, double Mh, double z, char *concenDef);

/*
 *    mass definition
 */

double Delta(const Model *model, double z, char *massDef);
double Delta_vir(const Model *model, double z);
double r_vir(const Model *model, double Mh, double c, double z);
double M_vir(const Model *model, double Mh, char *massDef, double c, double z);
void M1_to_M2(const Model *model, double M1, double c1, double Delta1, double Delta2, double z, double *M2, double *c2);
double rh(const Model *model, double Mh, double D, double z);
double Mh_rh(const Model *model, double r, double z);

/*
 *    halo model functions
 */

double f_sigma(const Model *model, double sigma, double z);
double b_sigma(const Model *model, double sigma, double z);
double dsigmaM_m1_dlnM(const Model *model, double Mh);
double sigma2M(const Model *model, double Mh);
double sigma2R(const Model *model, double R);
double int_for_sigma2R(double lnk, void *p);

/*
 *    cosmology functions
 */

double delta_c(const Model *model, double z);
double Dplus(const Model *model, double z);
double P_m_nonlin(const Model *model, double z, double k);
double P_m_lin(const Model *model, double z, double k);
double E2(const Model *model, double z,  int wOmegar);
double Omega_m_z(const Model *model, double z, double E2pre);
double rho_crit(const Model *model, double z);
double rho_bar(const Model *model, double z);
double dr_dz(const Model *model, double z);
double DC(const Model *model, double z, int wOmegar);
double DM(const Model *model, double z, int wOmegar);
double DA(const Model *model, double z, int wOmegar);
double DL(const Model *model, double z, int wOmegar);

#endif
