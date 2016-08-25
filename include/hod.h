/* ------------------------------------------------------------ *
 * hod.h	                 		                                  *
 * halomodel library                                            *
 * Jean Coupon 2015	         				                      *
 * ------------------------------------------------------------ */

#ifndef HOD_H
#define HOD_H

#include <stdio.h>
#include <math.h>
#include "utils.h"

double Ngal_c(const Model *model, double Mh, double log10Mstar_min, double log10Mstar_max);
double eta_cen(const Model *model, double Mh);
double sigma_log_M(const Model *model, double log10Mstar);

double Ngal_s(const Model *model, double Mh, double log10Mstar_min, double log10Mstar_max);
double Ngal(const Model *model, double Mh, double log10Mstar_min, double log10Mstar_max);

double msmh_log10Mstar(const Model *model, double Mh);
double msmh_log10Mh(const Model *model, double log10Mstar);

void copyModelHOD(const Model *from, Model *to);
int changeModelHOD(const Model *before, const Model *after);

#endif
