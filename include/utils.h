/*
 *    utils.h
 *    halomodel library
 *    Jean Coupon 2015
 */

#ifndef UTILS_H
#define UTILS_H


#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <fftw3.h>

#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_expint.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>

#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>


#include <Demo/errorlist.h>
#include <Demo/cosmo.h>
#include <Demo/io.h>
#include <Demo/maths.h>

typedef struct Model
{

   /* cosmology */
   double Omega_m;
   double Omega_de;
   double H0;
   char   *massDef;
   char   *concenDef;
   char   *hmfDef;
   char   *biasDef;

   /* HOD parameters */
   double log10M1;
   double log10Mstar0;
   double beta;
   double delta;
   double gamma;
   double log10Mstar_min;
   double log10Mstar_max;
   double sigma_log_M0;
   double sigma_lambda;
   double B_cut;
   double B_sat;
   double beta_cut;
   double beta_sat;
   double alpha;
   double fcen1;
   double fcen2;

   /* for X-ray luminosity profiles */
   int    hod;

   /* for X-ray binaries */
   double IxXB_Re;
   double IxXB_CR;

   double gas_log10n0;
   double gas_log10beta;
   double gas_log10rc;

   double gas_log10n0_1;
   double gas_log10n0_2;
   double gas_log10n0_3;
   double gas_log10n0_4;
   double gas_log10beta_1;
   double gas_log10beta_2;
   double gas_log10beta_3;
   double gas_log10beta_4;
   double gas_log10rc_1;
   double gas_log10rc_2;
   double gas_log10rc_3;
   double gas_log10rc_4;

   /* for gg lensing */
   double ggl_pi_max;
   double ggl_log10c;
   double ggl_log10Mh;
   double ggl_log10Mstar;

   int     wtheta_nz_N;
   double *wtheta_nz_z;
   double *wtheta_nz;

   /* XMM PSF (King's profile, must be normalised) */
   // double XMM_PSF_A;
   double XMM_PSF_rc_deg;
   double XMM_PSF_alpha;

}  Model;


typedef struct FFTLog_complex
{
  double re;
  double im;
  double amp;
  double arg;
}  FFTLog_complex;

typedef struct {
  int N;
  fftw_plan p_forward;
  fftw_plan p_backward;
  fftw_complex *an;
  fftw_complex *ak;
  fftw_complex *cm;
  fftw_complex *um;
  fftw_complex *cmum;
  double min;
  double max;
  double q;
  double mu;
  double kr;
} FFTLog_config;


typedef struct {

   double r, R, k, z;
   double ng, ngp, logMlim;

   double Mh, log10Mh, c;

   double log10Lx;
   double a, b, sigma;

   const Model *model;

   gsl_interp_accel *acc;
   gsl_spline *spline;

   double eps;

   double logrmin;
   double logrmax;

   double log10Mstar_min;
   double log10Mstar_max;

   // NICAEA cosmological parameters
   // cosmo *cosmo;
   int obs_type;

} params;


typedef double (*funcwithparsHalomodel)(double, void*);

typedef struct gsl_int_params
{
  void *params;
  funcwithparsHalomodel func;

} gsl_int_params;

/* Limits for xi(r), 1- and 2-halo, in Mpc.h */
#define RMIN1 0.001
#define RMAX1 5.0
#define RMIN2 0.1
#define RMAX2 400.0

#define RMIN  RMIN1
#define RMAX  RMAX2

/* Limits for quantities integrated over halo mass function */
#define LNMH_MIN 6.90  // (3.0*log(10.0))
#define LNMH_MAX 37.99 // (16.5*log(10.0))

#define CM3TOMPC3  (3.40366918775e-74)

/*
 *    Limits for power spectra. Note: these are
 *    simple numerical boundaries. It does not mean
 *    we know the matter power spectrum
 *    within this full range.
 */
#define KMIN  3.336e-6 //1.e-6
#define KMAX  333.6   // 1.e+6

/* Present critical density [M_sol h^2 / Mpc^3] */
// #define RHO_C0 2.7754e11

double FFTLog_TMP;

#define ODD 0
#define EVEN 1

/*
 *    obs_type
 *    1-point or 2-point observable type
 */


#define star     0 // star quantity
#define cen      1 // central or central-dark matter quantity
#define sat      2 // satellite or satellite-dark matter quantity
#define all      3 // satellite + central
#define XB       4 // for X-Ray binary stars
#define cencen  11 // central-central quantity -> 2-halo term
#define censat  12 // central-satellite quantity
#define satsat  22 // satellite-satellite quantity
#define twohalo 33 // (cen+sat)-(cen+sat) = 2-halo quantity

#define MIN(a,b) ((a)<(b) ? (a) : (b))
#define MAX(a,b) ((a)>(b) ? (a) : (b))
#define ABS(a) ((a) < 0 ? -(a) : (a))
#define PARITY(a) (a)%2 ? ODD : EVEN
#define FFTLog_SWAP(a,b) {FFTLog_TMP = (a); (a) = (b); (b) = FFTLog_TMP;}

double xi_from_Pkr(gsl_function *ak, double r_prime, FFTLog_config *fc);
void FFTLog(FFTLog_config *fc, const gsl_function *ar_in, double *k, double *ak_out, int dir);
FFTLog_config *FFTLog_init(int N, double min, double max, double q, double mu);
void FFTLog_free(FFTLog_config *fc);
FFTLog_complex FFTLog_U_mu(double mu, FFTLog_complex z);

double int_gsl(funcwithparsHalomodel func, void *params, double a, double b, double eps);
double int_gsl_QNG(funcwithparsHalomodel func, void *params, double a, double b, double eps);
double int_gsl_FFT(funcwithparsHalomodel func, void *params, double a, double b, double eps);
double integrand_gsl(double x, void *p);

double sinc(double x);

double trapz(double *x, double *y, int N);
double King(double r, double A, double rc, double alpha);
int assert_float(double before, double after);
int assert_int(double before, double after);


#endif
