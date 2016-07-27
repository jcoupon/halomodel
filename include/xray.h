/*
 *    xray.h
 *    halomodel library
 *    Jean Coupon 2015-2016
 */

#ifndef XRAY_H
#define XRAY_H

#include "utils.h"
#include "hod.h"
#include "abundance.h"
#include "cosmo.h"

void SigmaIx(const Model *model, double *R, int N, double Mh, double c, double z, int obs_type, double *result);
double intForIx(double logz, void *p);

void SigmaIxAll(const Model *model, double *R, int N, double Mh, double c, double z, double *result);

void Ix1hc(const Model *model, double *r, int N, double Mh, double c, double z, double *result);
double intForIx1hc(double logMh, void *p);

double ix(const Model *model, double r, double Mh, double c, double z);
double MhToTx(const Model *model, double Mh, double z);
double MhToZGas(const Model *model, double Mh, double z);
double TxToMh(const Model *model, double Tx, double z);

double MGas(const Model *model, double r, double Mh, double c, double z);
double intForMGas(double logr, void *p);
double Lambda(double Tx, double ZGas);
double CRToLx(const Model *model, double z, double Tx, double ZGas);

double nGas(const Model *model, double r, double Mh, double c, double z);
double betaModel(double r, double n0, double beta, double rc);

double inter_gas_log10n0(const Model *model, double log10Mh);
double inter_gas_log10beta(const Model *model, double log10Mh);
double inter_gas_log10rc(const Model *model, double log10Mh);


#endif


#if 0


void Ix(const Model *model, double *R, int N, double z, int obs_type, double *result);


double intForIx(double logz, void *p);

void IxXB(double *r, int N, const Model *model, double *result);


void Ix1hs(double *r, int N, const Model *model, double *result);
double intForIx1hs(double k, void *p);
double PIx1hs(double k, const Model *model);
double intForPIx1hs(double logMh, void *p);

void Ix1hc(double *r, int N, const Model *model, double *result);
double intForIx1hc(double logMh, void *p);

void uIx3D(double *k, int N, const Model *model, double log10Mh,  double *result);
double intForUIx3D(double r, void *p);
double Ix3D(double r, double log10Mh,  const Model *model);

double nGas2(double r, double log10Mh, const Model *model);

double inter_gas_log10n0(const Model *model, double log10Mh);
double inter_gas_log10beta(const Model *model, double log10Mh);
double inter_gas_log10r_c(const Model *model, double log10Mh);

double NormIx3D(const Model *model, double log10Mh, double rmax);
double intForNormIx3D(double logr, void *p);

double intForMGas(double logr, void *p);
double MGas( const Model *model, double log10Mh, double rmax);


double betaModel(double r, double n0, double beta, double r_c);
double betaModelSqProj(double r, double n0, double beta, double r_c);


double bias_log10Mh(const Model *model, double log10Lx);
double int_for_PLxGivenMh(double logMh, void *p);
double int_for_PLxGivenMh_norm(double logMh, void *p);
double normal(double x, double mu, double sigma);


#endif
