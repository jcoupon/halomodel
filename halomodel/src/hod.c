/*
 *    hod.c
 *    halomodel library
 *    Jean Coupon 2016
 */

#include "hod.h"

/* ---------------------------------------------------------------- *
 * HOD model functions                                              *
 * ---------------------------------------------------------------- */

double phi_c(const Model *model, double log10Mstar, double log10Mh){
    /*
     *   phi(Mstar|Mh) for central galaxies
     *   Eq. (1) of Leauthaud et al. (2011)
     *   without ln(10) (typo?)
     */
   double arg, result;
   double Mh = pow(10.0, log10Mh);

   double dlog10Mstar = 1.e-3;
   // double dN = Ngal_c(model, Mh, log10Mstar-dlog10Mstar/2.0, -1.0) - Ngal_c(model, Mh, log10Mstar+dlog10Mstar/2.0, -1.0);
   double dN = Ngal_c(model, Mh, log10Mstar-dlog10Mstar/2.0,  log10Mstar+dlog10Mstar/2.0);
   result = dN/dlog10Mstar;

   /*
   double sigma_logM = sigma_log_M(model, log10Mstar);
   arg = 0.5*pow( (log10Mstar - msmh_log10Mstar(model, log10Mh))/sigma_logM, 2.0);
   result = 1.0/(sqrt(2.0*M_PI)*sigma_logM) * exp(-arg);
   */
   return result;
 }

double phi_s(const Model *model, double log10Mstar, double log10Mh){
   /*
    *    phi(Mstar|Mh) for satellite galaxies
    *
    */

   double arg, result;
   double Mh = pow(10.0, log10Mh);

   // double dlog10Mstar = 0.01;
   // double dN = Ngal_s(model, Mh, log10Mstar-dlog10Mstar/2.0, -1.0) - Ngal_s(model, Mh, log10Mstar+dlog10Mstar/2.0, -1.0);

   // DEBUGGING
   double dlog10Mstar = 1.e-3;
   double dN = Ngal_s(model, Mh, log10Mstar-dlog10Mstar/2.0, log10Mstar+dlog10Mstar/2.0);

   result = dN/dlog10Mstar;

   return result;
}

double Ngal_c(const Model *model, double Mh, double log10Mstar_min, double log10Mstar_max){
   /*
    *    Number of central galaxies per halo between
    *    Mstellar_min and Mstellar_max.
    *    Set Mstellar_max = -1 to get threshold number
    */
   double arg, result;

   if(Mh < 1.e6) return 0.0;

   double log10Mh = log10(Mh);

   if (model->HOD_cen_N > 0){

      int i, N = model->HOD_cen_N;

      static Model model_tmp;
      static int firstcall = 1;
      static gsl_interp_accel *acc;
      static gsl_spline *spline;

      if (firstcall || changeModelHOD(model, &model_tmp)) {
         firstcall = 0;
         copyModelHOD(model, &model_tmp);
         acc = gsl_interp_accel_alloc();
         // spline = gsl_spline_alloc (gsl_interp_cspline, N);
         // DEBUGGING
         spline = gsl_spline_alloc (gsl_interp_linear, N);
         gsl_spline_init(spline, model->HOD_cen_log10Mh, model->HOD_cen_Ngal, N);
      }

      if (model->HOD_cen_log10Mh[0] > log10Mh || log10Mh > model->HOD_cen_log10Mh[N-1]){
         return 0.0;
      }else{
         return gsl_spline_eval(spline, log10Mh, acc);
      }
   }

   arg = (log10Mstar_min - msmh_log10Mstar(model, log10Mh))/(sqrt(2.0)*sigma_log_M(model, log10Mstar_min));
   result = eta_cen(model, Mh)*0.5*(1.0-gsl_sf_erf(arg));

   if(log10Mstar_max > 0){
      arg = (log10Mstar_max - msmh_log10Mstar(model, log10Mh))/(sqrt(2.0)*sigma_log_M(model, log10Mstar_max));
      result -= eta_cen(model, Mh)*0.5*(1.0-gsl_sf_erf(arg));
   }

   return MAX(0.0, result);
}

double eta_cen(const Model *model, double Mh){
   /*
    *    fraction of centrals as function of halo mass:
    *    = 1 if fcen1 = -1 and fcen2 = -2 (total sample)
    *    increasing form if fcen1 > 0.0 (passive)
    *    decreasing form if fcen1 < 0.0 (star-forming)
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
    *    Number of satellite galaxies per halo mass.
    */

   double M0, log10Msat,log10Mcut, Msat, Mcut, Mh_cen, result;

   if(Mh < 1.e6) return 0.0;

   if (model->HOD_sat_N > 0){

      double log10Mh = log10(Mh);

      int i, N = model->HOD_sat_N;

      static Model model_tmp;
      static int firstcall = 1;
      static gsl_interp_accel *acc;
      static gsl_spline *spline;

      if (firstcall || changeModelHOD(model, &model_tmp)) {
         firstcall = 0;
         copyModelHOD(model, &model_tmp);
         acc = gsl_interp_accel_alloc();
         // spline = gsl_spline_alloc (gsl_interp_cspline, N);
         // DEBUGGING
         spline = gsl_spline_alloc (gsl_interp_linear, N);
         gsl_spline_init(spline, model->HOD_sat_log10Mh, model->HOD_sat_Ngal, N);
      }

      if (model->HOD_sat_log10Mh[0] > log10Mh || log10Mh > model->HOD_sat_log10Mh[N-1]){
         return 0.0;
      }else{
         return gsl_spline_eval(spline, log10Mh, acc);
      }
   }

   log10Msat = log10M_sat(model, log10Mstar_min);
   log10Mcut = log10M_cut(model, log10Mstar_min);
   if (model->B_cut < 0.0){
      M0 = pow(10.0, log10Mcut);
      if(Mh - M0 > 0.0){
         result = pow((Mh - M0)/pow(10.0, log10Msat), model->alpha);
      }else{
         result = 0.0;
      }
   }else{
      Msat = pow(10.0, log10Msat);
      Mcut = pow(10.0, log10Mcut);
      Mh_cen = pow(10.0, msmh_log10Mh(model, log10Mstar_min));
      // result = Ngal_c(model, Mh, log10Mstar_min, -1.0)*pow(Mh/Msat, model->alpha)*exp(-Mcut/Mh);

      // DEBUGGING
      result = pow(Mh/Msat, model->alpha)*exp(-(Mcut+Mh_cen)/Mh);

   }


   /*    Stellar mass bin */
   if(log10Mstar_max > 0){

      log10Msat = log10M_sat(model, log10Mstar_max);
      log10Mcut = log10M_cut(model, log10Mstar_max);
      if (model->B_cut < 0.0){
         M0 = pow(10.0, log10Mcut);
         if(Mh - M0 > 0.0){
            result -= pow((Mh - M0)/pow(10.0, log10Msat), model->alpha);
         }
      }else{
         Msat = pow(10.0, log10Msat);
         Mcut = pow(10.0, log10Mcut);
         Mh_cen = pow(10.0, msmh_log10Mh(model, log10Mstar_max));
         // result -= Ngal_c(model, Mh, log10Mstar_max, -1.0)*pow(Mh/Msat, model->alpha) * exp(-Mcut/Mh);

         // DEBUGGING
         result -= pow(Mh/Msat, model->alpha)*exp(-(Mcut+Mh_cen)/Mh);


      }
   }
   // return result;
   return MAX(0.0, result);
}

double log10M_sat(const Model *model, double log10Mstar){
   double result;

   result = log10(model->B_sat) + model->beta_sat*msmh_log10Mh(model, log10Mstar) + (12.0+model->log10h)*(1.0-model->beta_sat);

   return result;
}

double log10M_cut(const Model *model, double log10Mstar){
   double result;

   if (model->B_cut < 0.0){
      result =  msmh_log10Mh(model, log10Mstar) - 0.5;
   }else{
      result = log10(model->B_cut) + model->beta_cut*msmh_log10Mh(model, log10Mstar) + (12.0+model->log10h)*(1.0-model->beta_cut);
   }

   return result;
}


double Ngal(const Model *model, double Mh, double log10Mstar_min, double log10Mstar_max){
   /*
    *    Total number of galaxies per halo.
    */
   double Ngalc, Ngals;

   Ngalc = Ngal_c(model, Mh, log10Mstar_min, log10Mstar_max);
   Ngals = Ngal_s(model, Mh, log10Mstar_min, log10Mstar_max);

   return Ngalc + Ngals;
}


double shmr_c(const Model *model, double Mh, double log10Mstar_min, double log10Mstar_max){
  /*     returns the total stellar mass from central galaxies as
   *     a function of halo mass and stellar bin.
   */

   params p;
   p.model = model;
   p.Mh = Mh;

   double result;

   result  = int_gsl(intForShmr_c, (void*)&p, log10Mstar_min, log10Mstar_max, 1.e-3);
   result -= Ngal_c(model, Mh, log10Mstar_max, -1)*pow(10.0, log10Mstar_max) - Ngal_c(model, Mh, log10Mstar_min, -1)*pow(10.0, log10Mstar_min);

   return result/Mh;
}

double intForShmr_c(double log10Mstar, void *p){

   const Model *model = ((params *)p)->model;
   double Mh = ((params *)p)->Mh;

  return Ngal_c(model, Mh, log10Mstar, -1) * pow(10.0, log10Mstar)*log(10);
}



double shmr_s(const Model *model, double Mh, double log10Mstar_min, double log10Mstar_max){
  /*     returns the total stellar mass from satellite galaxies as
   *     a function of halo mass and stellar bin.
   */

   params p;
   p.model = model;
   p.Mh = Mh;

   double result;

   result  = int_gsl(intForShmr_s, (void*)&p, log10Mstar_min, log10Mstar_max, 1.e-3);
   result -= Ngal_s(model, Mh, log10Mstar_max, -1)*pow(10.0, log10Mstar_max) - Ngal_s(model, Mh, log10Mstar_min, -1)*pow(10.0, log10Mstar_min);

   return result/Mh;
}

double intForShmr_s(double log10Mstar, void *p){

   const Model *model = ((params *)p)->model;
   double Mh = ((params *)p)->Mh;

  return Ngal_s(model, Mh, log10Mstar, -1) * pow(10.0, log10Mstar)*log(10);
}

double shmr(const Model *model, double Mh, double log10Mstar_min, double log10Mstar_max){
   /*
    *    Total number of galaxies per halo.
    */
   double shmrc, shmrs;

   shmrc = shmr_c(model, Mh, log10Mstar_min, log10Mstar_max);
   shmrs = shmr_s(model, Mh, log10Mstar_min, log10Mstar_max);

   return shmrc + shmrs;
}


double msmh_log10Mstar(const Model *model, double log10Mh){
   /*
    *    Returns Mstar = f(Mh) for a given Mstar-Mh relation.
    *    The Mstar-Mh relation is in fact defined via its
    *    inverse function, i.e. Mh = f(Mstar). The routine
    *    first evaluates Mh = f(Mstar) and compute the
    *    inverse value through interpolation.
    *    The relation is evaluated only when the HOD
    *    parameters changes (TO DO).
    */

   int i, N = 64;
   double log10Mstar_min = 3.0, log10Mstar_max = 13.0;

   static int firstcall = 1;
   static gsl_interp_accel *acc;
   static gsl_spline *spline;

   static double *t_log10Mh;
   static double *t_log10Mstar;
   static double dlog10Mstar;
   static Model model_tmp;

   if (firstcall) {
      firstcall = 0;

      copyModelHOD(model, &model_tmp);

      /*    tabulate log10Mh = f(log10Mstar) */
      t_log10Mh = (double *)malloc(N*sizeof(double));
      t_log10Mstar = (double *)malloc(N*sizeof(double));
      dlog10Mstar = (log10Mstar_max - log10Mstar_min)/(double)N;

      for(i=0;i<N;i++){
         t_log10Mstar[i] = log10Mstar_min + dlog10Mstar*(double)i;
         t_log10Mh[i] = msmh_log10Mh(model, t_log10Mstar[i]);
      }

      acc = gsl_interp_accel_alloc();
      spline = gsl_spline_alloc (gsl_interp_cspline, N);

      gsl_spline_init(spline, t_log10Mh, t_log10Mstar, N);

   }

   if (changeModelHOD(model, &model_tmp)) {

      copyModelHOD(model, &model_tmp);

      /*    update log10Mh = f(log10Mstar) */
      for(i=0;i<N;i++){
         t_log10Mh[i] = msmh_log10Mh(model, t_log10Mstar[i]);
         // printf("%f %f\n", t_log10Mstar[i], t_log10Mh[i]);
      }
      gsl_spline_init(spline, t_log10Mh, t_log10Mstar, N);
      // printf("\n\n");
   }



   if (t_log10Mh[0] < log10Mh && log10Mh < t_log10Mh[N-1]){
      return gsl_spline_eval(spline, log10Mh, acc);
   }else{
      return 0.0;
   }

#if 0
   free(t_log10Mh);
   free(t_log10Mstar);
   gsl_spline_free (spline);
   gsl_interp_accel_free (acc);
#endif

}


double msmh_log10Mh(const Model *model, double log10Mstar){
   /*
    *    Mstar - Mh relation.
    *    Parameterization from
    *    Behroozi et al. (2010).
    */


   double result;

   double log10A = log10Mstar - model->log10Mstar0;
   double A = pow(10.0, log10A);
   result = model->log10M1 + model->beta * log10A;
   result += pow(A, model->delta)/(1.0 + pow(A, -model->gamma)) - 0.5;

   return result;
}

#define EPS 1.e-8

void copyModelHOD(const Model *from, Model *to){
   /*    Copies model "from" to model "to" */

   int i;

   to->hod = from->hod;
   to->log10M1 = from->log10M1;
   to->log10Mstar0 = from->log10Mstar0;
   to->beta = from->beta;
   to->delta = from->delta;
   to->gamma = from->gamma;
   to->sigma_log_M0 = from->sigma_log_M0;
   to->sigma_lambda = from->sigma_lambda;
   to->B_cut = from->B_cut;
   to->B_sat = from->B_sat;
   to->beta_cut = from->beta_cut;
   to->beta_sat = from->beta_sat;
   to->alpha = from->alpha;
   to->fcen1 = from->fcen1;
   to->fcen2 = from->fcen2;

   to->HOD_cen_N = from->HOD_cen_N;
   to->HOD_cen_log10Mh = (double *)malloc(from->HOD_cen_N*sizeof(double));
   to->HOD_cen_Ngal = (double *)malloc(from->HOD_cen_N*sizeof(double));
   for (i=0;i<from->HOD_cen_N;i++){
      to->HOD_cen_log10Mh[i] = from->HOD_cen_log10Mh[i];
      to->HOD_cen_Ngal[i] = from->HOD_cen_Ngal[i];
   }
   to->HOD_sat_N = from->HOD_sat_N;
   to->HOD_sat_log10Mh = (double *)malloc(from->HOD_sat_N*sizeof(double));
   to->HOD_sat_Ngal = (double *)malloc(from->HOD_sat_N*sizeof(double));
   for (i=0;i<from->HOD_sat_N;i++){
      to->HOD_sat_log10Mh[i] = from->HOD_sat_log10Mh[i];
      to->HOD_sat_Ngal[i] = from->HOD_sat_Ngal[i];
   }

   return;
}


int changeModelHOD(const Model *before, const Model *after){
   /* test if any of the X-ray parameters changed */

   int result = 0;

   result += assert_int(before->hod, after->hod);
   result += assert_float(before->log10M1, after->log10M1);
   result += assert_float(before->log10Mstar0, after->log10Mstar0);
   result += assert_float(before->beta, after->beta);
   result += assert_float(before->delta, after->delta);
   result += assert_float(before->gamma, after->gamma);
   result += assert_float(before->sigma_log_M0, after->sigma_log_M0);
   result += assert_float(before->sigma_lambda, after->sigma_lambda);
   result += assert_float(before->B_cut, after->B_cut);
   result += assert_float(before->B_sat, after->B_sat);
   result += assert_float(before->beta_cut, after->beta_cut);
   result += assert_float(before->beta_sat, after->beta_sat);
   result += assert_float(before->alpha, after->alpha);
   result += assert_float(before->fcen1, after->fcen1);
   result += assert_float(before->fcen2, after->fcen2);

   if (before->HOD_cen_N > 0 || after->HOD_cen_N > 0){
      result += assert_int(before->HOD_cen_N, after->HOD_cen_N);
      result += assert_float_table(before->HOD_cen_log10Mh, before->HOD_cen_N, after->HOD_cen_log10Mh, after->HOD_cen_N);
      result += assert_float_table(before->HOD_cen_Ngal, before->HOD_cen_N, after->HOD_cen_Ngal, after->HOD_cen_N);
   }
   if (before->HOD_sat_N > 0 || after->HOD_sat_N > 0){
      result += assert_int(before->HOD_sat_N, after->HOD_sat_N);
      result += assert_float_table(before->HOD_sat_log10Mh, before->HOD_sat_N, after->HOD_sat_log10Mh, after->HOD_sat_N);
      result += assert_float_table(before->HOD_sat_Ngal, before->HOD_sat_N, before->HOD_sat_Ngal, after->HOD_sat_N);
   }
   return result;

}

#undef EPS
