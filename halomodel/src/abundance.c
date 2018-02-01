/* ------------------------------------------------------------ *
 * abundance.c                                                  *
 * halomodel library                                            *
 * Jean Coupon 2016                                             *
 * ------------------------------------------------------------ */

#include "abundance.h"

/* ---------------------------------------------------------------- *
 * Stellar mass function
 * ---------------------------------------------------------------- */



void dndlog10Mstar(const Model *model, double *log10Mstar, int N, double z, int obs_type, double *result)
{
   /* Computes the stellar mass function.
   See Leauthaud et al. (2011) eq. 17
   and Coupon et al. (2012) Eq. A. 21.

   takes Mstar in (Msun/h^2), returns result in (Mpc/h)^-3 dex^-1
   */

   int i;
   double dlog10Mstar = 0.01;

   if (model->hod == 0){
      printf("The SMF is not supported for non-HOD models. Exiting...\n");
      exit(-1);
   }

   for(i=0; i<N; i++){
      result[i] = ngal_den(model, LNMH_MAX,
         log10Mstar[i] - dlog10Mstar/2.0,
         log10Mstar[i] + dlog10Mstar/2.0, z, obs_type)/dlog10Mstar;

      /* to avoid numerical issues when comparing results
      between machines
      */
      if (result[i] < 1.e-10){
         result[i] = 0.0;
      }

   }

   return;

}

double logM_lim(const Model *model, double r, double c, double z, int obs_type)
{
   if (r > 2.0) {
      return LNMH_MAX;
   }

   double ng_triax, ng, logM, dlogM, logM_hi, logM_lo;
   ng_triax = ngal_triax(model, r, c, z, obs_type);

   dlogM = 0.01;
   logM_lo = LNMH_MIN;
   logM_hi = LNMH_MAX;

   do {
      logM = (logM_hi+logM_lo)/2.0;
      ng = ngal_den(model, logM, model->log10Mstar_min, model->log10Mstar_max, z, obs_type);
      if (ng < ng_triax) logM_lo = logM;
      else logM_hi = logM;
   } while (logM_hi-logM_lo > dlogM);

   logM = (logM_hi+logM_lo)/2.0;

   return logM;
}

double ngal_triax(const Model *model, double r, double c, double z, double obs_type)
{
   int i, j, N = 200;
   double x,y,P,sum1,sum2,lnM1,lnM2,R1,R2;

   params p;
   p.model = model;
   p.z = z;
   p.log10Mstar_min = model->log10Mstar_min;
   p.log10Mstar_max = model->log10Mstar_max;
   p.obs_type = obs_type;

   double dlogM = (LNMH_MAX - LNMH_MIN)/(double)N;

   sum1 = 0.0;
   for(i=0;i<N;i++){
      lnM1 = LNMH_MIN + dlogM*(double)i;
      R1 = r_vir(model, exp(lnM1), c, z);
      sum2 = 0.0;
      for(j=0;j<N;j++){
         lnM2 = LNMH_MIN + dlogM*(double)j;
         R2 = r_vir(model, exp(lnM2), c, z);
         x = r/(R1 + R2);
         // This matches Leauthaud et al. (2011):
         // x = r/(R1);
         y = (x - 0.8)/0.29;
         if (y<0) {
            sum2 += 0.0;
         } else if (y>1) {
            sum2 +=  int_for_ngal_den(lnM2, (void*)&p);
         } else {
            P  = (3.0 - 2.0*y)*y*y;
            sum2 +=  int_for_ngal_den(lnM2, (void*)&p)*P;
         }
      }
      sum1 +=  int_for_ngal_den(lnM1,(void*)&p)*sum2*dlogM;
   }
   return sqrt(sum1*dlogM);
}


double ngal_den(const Model *model, double lnMh_max, double log10Mstar_min, double log10Mstar_max, double z, int obs_type)
/*
 *    galaxy number density per unit volume for a given HOD and halo mass function.
 *    In Mpc^-3.
 *    logMh_max is the maximum halo mass to integrate. Mstellar_min/max
 *    are the stellar mass bin limit (set Mstellar_max = -1 for
 *    threshold sample)
 */
{
   params p;
   p.model = model;
   p.z = z;
   p.log10Mstar_min = log10Mstar_min;
   p.log10Mstar_max = log10Mstar_max;
   p.obs_type = obs_type;

   return int_gsl(int_for_ngal_den, (void*)&p, LNMH_MIN, lnMh_max, 1.e-5);
}

double int_for_ngal_den(double lnMh, void *p) {

   const Model *model = ((params *)p)->model;
   double z = ((params *)p)->z;
   double log10Mstar_min = ((params *)p)->log10Mstar_min;
   double log10Mstar_max = ((params *)p)->log10Mstar_max;
   int obs_type = ((params *)p)->obs_type;

   double Mh = exp(lnMh);

   switch (obs_type){
      case cen:
         return Ngal_c(model, Mh, log10Mstar_min, log10Mstar_max) * dndlnMh(model, Mh, z);
         break;
      case sat:
         return Ngal_s(model, Mh, log10Mstar_min, log10Mstar_max) * dndlnMh(model, Mh, z);
         break;
      case all:
         return Ngal(model, Mh, log10Mstar_min, log10Mstar_max) * dndlnMh(model, Mh, z);
         break;
      default:
         return 0.0;
         break;
   }
}


double MstarMean(const Model *model, double z, int obs_type)
{
  /* returns the mean stellar mass if HOD model .*/

  params p;
  p.model = model;
  p.z = z;
  p.obs_type = obs_type;

   return int_gsl(intForMstarMean, (void*)&p, model->log10Mstar_min, model->log10Mstar_max, 1.e-3) /
      ngal_den(model, LNMH_MAX, model->log10Mstar_min, model->log10Mstar_max, z, obs_type);
}


double intForMstarMean(double log10Mstar, void *p)
{

   const Model *model = ((params *)p)->model;
   double z = ((params *)p)->z;
   int obs_type = ((params *)p)->obs_type;

   double Mstar = pow(10.0, log10Mstar);

   double dlog10 = 0.01;

   /* This is M * SMF */
   return Mstar * ngal_den(model, LNMH_MAX, log10Mstar - dlog10/2.0, log10Mstar + dlog10/2.0, z, obs_type)/dlog10;

}
