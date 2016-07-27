/* ------------------------------------------------------------ *
 * hod.c                                                        *
 * halomodel library                                            *
 * Jean Coupon 2016                                             *
 * ------------------------------------------------------------ */

#include "hod.h"

/* ---------------------------------------------------------------- *
 * HOD model functions                                              *
 * ---------------------------------------------------------------- */

double Ngal_c(const Model *model, double Mh, double log10Mstar_min, double log10Mstar_max){
   /*
   Number of central galaxies per halo between
   Mstellar_min and Mstellar_max.
   Set Mstellar_max = -1 to get threshold number
   */
   double arg, result;

   if(Mh < 1.e6) return 0.0;

   double log10Mh = log10(Mh);

   arg    = (log10Mstar_min - msmh_log10Mstar(model, log10Mh))/(sqrt(2.0)*sigma_log_M(model, log10Mstar_min));
   result = eta_cen(model, Mh)*0.5*(1.0-gsl_sf_erf(arg));

   if(log10Mstar_max > 0){
      arg     = (log10Mstar_max - msmh_log10Mstar(model, log10Mh))/(sqrt(2.0)*sigma_log_M(model, log10Mstar_max));
      result -= eta_cen(model, Mh)*0.5*(1.0-gsl_sf_erf(arg));
   }

   return result;
}

double eta_cen(const Model *model, double Mh){
   /*
   fraction of centrals as function of halo mass:
   = 1 if fcen1 = -1 and fcen2 = -2 (total sample)
   increasing form if fcen1 > 0.0 (passive)
   decreasing form if fcen1 < 0.0 (star-forming)
   */

   double result;

   double log10Mh = log10(Mh);

   if(model->fcen1 < 0.0 || model->fcen2 < 0.0){
      result = 1.0;
   }else{
      result = 0.5*(1.0-gsl_sf_erf((model->fcen1+log10Mh)/model->fcen2));
   }

   return result;
}


double sigma_log_M(const Model *model, double log10Mstar){
   /* varying sigma_log_M */

   return model->sigma_log_M0 * pow(pow(10.0, log10Mstar)/1.0e10, -model->sigma_lambda);
}


double Ngal_s(const Model *model, double Mh, double log10Mstar_min, double log10Mstar_max){
   /*
   Number of satellite galaxies per halo mass.
   */

   double M0, log10Msat,log10Mcut,  result;

   if(Mh < 1.e6) return 0.0;

   //double Mh = pow(10, log10Mh);

   log10Msat = log10(model->B_sat) + model->beta_sat*msmh_log10Mh(model, log10Mstar_min) + 12.0*(1.0-model->beta_sat);
   log10Mcut = msmh_log10Mh(model, log10Mstar_min) - 0.5;
   M0        = pow(10.0, log10Mcut);
   if(Mh - M0 > 0.0){
      result = pow((Mh - M0)/pow(10.0, log10Msat), model->alpha);
   }else{
      result = 0.0;
   }

   /* Stellar bin */
   if(log10Mstar_max > 0){
      log10Msat = log10(model->B_sat) + model->beta_sat*msmh_log10Mh(model, log10Mstar_max) + 12.0*(1.0-model->beta_sat);
      log10Mcut = msmh_log10Mh(model, log10Mstar_max) - 0.5;
      M0        = pow(10.0, log10Mcut);
      if(Mh - M0 > 0.0){
         result -= pow((Mh - M0)/pow(10.0, log10Msat), model->alpha);
      }
   }

   return MAX(0.0, result);
}

double Ngal(const Model *model, double Mh, double log10Mstar_min, double log10Mstar_max){
   /* Total number of galaxies per halo.
   */
   double Ngalc, Ngals;

   Ngalc = Ngal_c(model, Mh, log10Mstar_min, log10Mstar_max);
   Ngals = Ngal_s(model, Mh, log10Mstar_min, log10Mstar_max);

   return Ngalc + Ngals;
}


double msmh_log10Mstar(const Model *model, double log10Mh){
   /*
   Returns Mstar = f(Mh) for a given Mstar-Mh relation.
   The Mstar-Mh relation is in fact defined via its
   inverse function, i.e. Mh = f(Mstar). The routine
   first evaluates Mh = f(Mstar) and compute the
   inverse value through interpolation.

   The relation is evaluated only when the HOD
   parameters changes (TO DO).
   */

   int i, N = 64;
   double log10Mstar_min = 6.0, log10Mstar_max = 12.5;

   static int firstcall = 1;
   static gsl_interp_accel *acc;
   static gsl_spline *spline;

   static double *t_log10Mh;
   static double *t_log10Mstar;
   static double dlog10Mstar;

   if (firstcall) {
      firstcall = 0;

      /* tabulate log10Mh = f(log10Mstar) */
      t_log10Mh     = (double *)malloc(N*sizeof(double));
      t_log10Mstar  = (double *)malloc(N*sizeof(double));
      dlog10Mstar   = (log10Mstar_max - log10Mstar_min)/(double)N;

      for(i=0;i<N;i++){
         t_log10Mstar[i] = log10Mstar_min + dlog10Mstar*(double)i;
         t_log10Mh[i]    = msmh_log10Mh(model, t_log10Mstar[i]);
      }

      acc       = gsl_interp_accel_alloc();
      spline    = gsl_spline_alloc (gsl_interp_cspline, N);

      gsl_spline_init(spline, t_log10Mh, t_log10Mstar, N);

   }


   if (changeModelHOD(model)) {
      /* update log10Mh = f(log10Mstar) */
      for(i=0;i<N;i++){
         t_log10Mh[i] = msmh_log10Mh(model, t_log10Mstar[i]);
      }
      gsl_spline_init(spline, t_log10Mh, t_log10Mstar, N);
   }

   if (t_log10Mh[0] < log10Mh && log10Mh < t_log10Mh[N-1]){
      return gsl_spline_eval(spline, log10Mh, acc);
   }else{
      return 0.0;
   }

   /*
   free(t_log10Mh);
   free(t_log10Mstar);
   gsl_spline_free (spline);
   gsl_interp_accel_free (acc);
   */
}


double msmh_log10Mh(const Model *model, double log10Mstar){
   /*
   Mstar - Mh relation.
   Parameterization from
   Behroozi et al. (2010).
   */
   double result;

   double log10A = log10Mstar - model->log10Mstar0;
   double A      = pow(10.0, log10A);
   result        = model->log10M1 + model->beta * log10A;
   result       += pow(A, model->delta)/(1.0 + pow(A, -model->gamma)) - 0.5;

   return result;
}

#define EPS 1.e-8

int changeModelHOD(const Model *model){
   /* test if any of the HOD parameters changed */


   static Model model_tmp;
   static int firstcall = 1;
   int result;

   if (firstcall) {
      firstcall = 0;

      model_tmp.log10M1 = model->log10M1;
      model_tmp.log10Mstar0 = model->log10Mstar0;
      model_tmp.beta = model->beta;
      model_tmp.delta = model->delta;
      model_tmp.gamma = model->gamma;
      model_tmp.sigma_log_M0 = model->sigma_log_M0;
      model_tmp.sigma_lambda = model->sigma_lambda;
      model_tmp.B_cut = model->B_cut;
      model_tmp.B_sat = model->B_sat;;
      model_tmp.beta_cut = model->beta_cut;
      model_tmp.beta_sat = model->beta_sat;
      model_tmp.alpha = model->alpha;
      model_tmp.fcen1 = model->fcen1;
      model_tmp.fcen2 = model->fcen2;

   }

   result = 0;
   if (fabs(model_tmp.log10M1 - model->log10M1) > EPS) {
      model_tmp.log10M1 = model->log10M1;
      result = 1;
   }
   if (fabs(model_tmp.log10Mstar0 - model->log10Mstar0) > EPS) {
      model_tmp.log10Mstar0 = model->log10Mstar0;
      result = 1;
   }
   if (fabs(model_tmp.beta - model->beta) > EPS) {
      model_tmp.beta = model->beta;
      result = 1;
   }
   if (fabs(model_tmp.delta - model->delta) > EPS) {
      model_tmp.delta = model->delta;
      result = 1;
   }
   if (fabs(model_tmp.gamma - model->gamma) > EPS) {
      model_tmp.gamma = model->gamma;
      result = 1;
   }
   if (fabs(model_tmp.sigma_log_M0 - model->sigma_log_M0) > EPS) {
      model_tmp.sigma_log_M0 = model->sigma_log_M0;
      result = 1;
   }
   if (fabs(model_tmp.sigma_lambda - model->sigma_lambda) > EPS) {
      model_tmp.sigma_lambda = model->sigma_lambda;
      result = 1;
   }
   if (fabs(model_tmp.B_cut - model->B_cut) > EPS) {
      model_tmp.B_cut = model->B_cut;
      result = 1;
   }
   if (fabs(model_tmp.B_sat - model->B_sat) > EPS) {
      model_tmp.B_sat = model->B_sat;
      result = 1;
   }
   if (fabs(model_tmp.beta_cut - model->beta_cut) > EPS) {
      model_tmp.beta_cut = model->beta_cut;
      result = 1;
   }
   if (fabs(model_tmp.beta_sat - model->beta_sat) > EPS) {
      model_tmp.beta_sat = model->beta_sat;
      result = 1;
   }
   if (fabs(model_tmp.alpha - model->alpha) > EPS) {
      model_tmp.alpha = model->alpha;
      result = 1;
   }
   if (fabs(model_tmp.fcen1 - model->fcen1) > EPS) {
      model_tmp.fcen1 = model->fcen1;
      result = 1;
   }
   if (fabs(model_tmp.fcen2 - model->fcen2) > EPS) {
      model_tmp.fcen2 = model->fcen2;
      result = 1;
   }

   return result;

}

#undef EPS
