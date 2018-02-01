/*
 *    lensing.c
 *    halomodel library
 *    Jean Coupon 2016 - 2017
 */

#include "lensing.h"

/*
 *    Delta Sigma (galaxy-galaxy lensing)
 */
void DeltaSigma(const Model *model, double *R, int N, double z, int obs_type, double *result)
{
   /*
    *    Computes DeltaSigma = Sigma(<r) - Sigma(r)
    *    See Yoo et al. (2006), Leauthaud et al. (2011)
    *    This is gamma X Sigma_crit X 1e-12
    *
    *    ** DS and R IN COMOVING UNITS **
    *
    *    R in [h^-1 Mpc]
    *    DS in [Msun h^2 pc^-2]
    *    R_como  = R_phys * (1+z)
    *    DS_como = DS_phys / (1+z)^2

    */

   int i;

   if(obs_type == star){
      DeltaSigmaStar(model, R, N, z, result);
      return;
   }

   if(obs_type == all){
      DeltaSigmaAll(model, R, N, z, result);
      return;
   }

   double pi_max = model->ggl_pi_max;

   /* interpolate to speed up integration  */
   int Ninter = 40;
   double *logrinter = (double *)malloc(Ninter*sizeof(double));
   double *rinter = (double *)malloc(Ninter*sizeof(double));
   double rinter_max = pi_max;
   double rinter_min = MIN(RMIN1, R[0]);
   double dlogrinter = log(rinter_max/rinter_min)/(double)Ninter;

   for(i=0;i<Ninter;i++){
      logrinter[i] = log(rinter_min)+dlogrinter*(double)i;
      rinter[i] = exp(logrinter[i]);
   }

   /* projected density contrast (see Yoo et al. (2006) Eq. 2 */
   double *Sigma_inter = (double *)malloc(Ninter*sizeof(double));

   Sigma(model, rinter, Ninter, z, obs_type, Sigma_inter);

   /* Interpolation */
   gsl_interp_accel *acc = gsl_interp_accel_alloc();
   gsl_spline *spline    = gsl_spline_alloc (gsl_interp_cspline, Ninter);
   gsl_spline_init(spline, logrinter, Sigma_inter, Ninter);

   params p;
   p.acc = acc;
   p.spline = spline;
   p.eps = 1.0e-4;
   p.logrmin = logrinter[0];
   p.logrmax = logrinter[Ninter-1];

   double rhobar = rho_bar(model, 0.0);

   /* Loop over input scales */
   for(i=0; i<N; i++){

      if(p.logrmin < log(R[i]) && log(R[i]) < p.logrmax){
         result[i] = 1.0e-12*rhobar*(2.0/(R[i]*R[i])
            * int_gsl(intForDeltaSigma, (void*)&p, log(rinter_min), log(R[i]), p.eps)
            - gsl_spline_eval(spline, log(R[i]), acc));
      }else{
         result[i] = 0.0;
      }
   }

   free(Sigma_inter);
   free(logrinter);
   free(rinter);
   gsl_spline_free (spline);
   gsl_interp_accel_free (acc);
}

double intForDeltaSigma(double logR, void *p)
{
   double result = 0.0;
   double R = exp(logR);
   double logrmin = ((params *)p)->logrmin;
   double logrmax = ((params *)p)->logrmax;
   gsl_interp_accel *acc = ((params *)p)->acc;
   gsl_spline *spline = ((params *)p)->spline;

   if(logrmin < logR && logR < logrmax){
      result = R*R*gsl_spline_eval(spline, logR, acc);
   }

   return result;
}


void Sigma(const Model *model, double *R, int N, double z, int obs_type, double *result)
{
   /*
    *    Computes xi, projects it along z and returns Sigma.
    *    See e.g. Zehavi et al. (2005) Eq. (3).
    */

   double pi_max = model->ggl_pi_max;

   /* interpolate to speed up integration  */
   int i, Ninter = 40;
   double *logrinter = (double *)malloc(Ninter*sizeof(double));
   double *rinter = (double *)malloc(Ninter*sizeof(double));
   double rinter_min = RMIN;
   double rinter_max = RMAX;
   double dlogrinter = log(rinter_max/rinter_min)/(double)Ninter;

   for(i=0;i<Ninter;i++){
      logrinter[i] = log(rinter_min)+dlogrinter*(double)i;
      rinter[i] = exp(logrinter[i]);
   }

   double *S = (double *)malloc(Ninter*sizeof(double));

   switch (obs_type){
      case cen:
         xi_gm_cen(model, rinter, Ninter, z, S);
         break;
      case sat:
         xi_gm_sat(model, rinter, Ninter, z, S);
         break;
      case twohalo:
         xi_gm_twohalo(model, rinter, Ninter, z, S);
         break;
   }

   /* interpolate xi(r) */
   gsl_interp_accel *acc = gsl_interp_accel_alloc();
   gsl_spline *spline = gsl_spline_alloc (gsl_interp_cspline, Ninter);
   gsl_spline_init(spline, logrinter, S, Ninter);

   params p;
   p.acc = acc;
   p.spline = spline;
   p.eps = 1.0e-4;
   p.logrmin = logrinter[0];
   p.logrmax = logrinter[Ninter-1];

   for(i=0;i<N;i++){
      p.R = R[i];
      //result[i] = 2.0*int_gsl(intForSigma, (void*)&p, log(rinter_min), log(pi_max), p.eps);
      result[i] = 2.0*int_gsl(intForSigma, (void*)&p, log(R[i]), log(pi_max), p.eps);
   }

   free(S);
   free(rinter);
   free(logrinter);
   gsl_spline_free (spline);
   gsl_interp_accel_free (acc);
}

double intForSigma(double logz, void *p)
{
   double result = 0.0;
   double z = exp(logz);
   double logrmin = ((params *)p)->logrmin;
   double logrmax = ((params *)p)->logrmax;
   double R = ((params *)p)->R;
   gsl_interp_accel *acc = ((params *)p)->acc;
   gsl_spline *spline = ((params *)p)->spline;

   /*
   double r    = sqrt(R*R + z*z);
   double logr = log(r);

   if(logrmin < logr && logr < logrmax){
      result = gsl_spline_eval(spline, logr, acc) * z;
   }
   */

   if(logrmin < logz && logz < logrmax){
      result = z*z*gsl_spline_eval(spline, logz, acc)/sqrt(z*z - R*R);
   }

   return result;
}



void DeltaSigmaAll(const Model *model, double *R, int N, double z, double *result)
{

   int i;

   double *result_tmp = (double *)malloc(N*sizeof(double));

   DeltaSigma(model, R, N, z, star, result_tmp);
   for(i=0;i<N;i++){
      result[i]  = result_tmp[i];
   }

   DeltaSigma(model, R, N, z, cen, result_tmp);
   for(i=0;i<N;i++){
      result[i] += result_tmp[i];
   }

   DeltaSigma(model, R, N, z, sat, result_tmp);
   for(i=0;i<N;i++){
      result[i] += result_tmp[i];
   }

   DeltaSigma(model, R, N, z, twohalo, result_tmp);
   for(i=0;i<N;i++){
      result[i] += result_tmp[i];
   }

   free(result_tmp);

   return;
}


void DeltaSigmaStar(const Model *model, double *R, int N, double z, double *result)
{
   /*
    *    Returns the stellar lensing part. Assumed point source.
    */

   int i;
   double Mstar;

   if(model->hod){

      params p;
      p.model = model;
      p.z     = z;

      double ng = ngal_den(model, LNMH_MAX, model->log10Mstar_min, model->log10Mstar_max, z, all);
      for(i=0;i<N;i++){
         if (R[i] < RMAX1){
            p.r = R[i];
            result[i] = int_gsl(intForDeltaSigmaStar, (void*)&p, model->log10Mstar_min, model->log10Mstar_max, 1.e-2)/ng;
         }else{
            result[i] = 0.0;
         }
      }

   }else{

      if (isnan(model->ggl_log10Mstar)){
         Mstar = 0.0;
      }else{
          // need to divide by h to convert to h^-1 Mpc (Mstar is in h^-2 Mpc)
         Mstar = pow(10.0, model->ggl_log10Mstar) / model->h;
      }
      for(i=0;i<N;i++){
         result[i] = 1.e-12 * Mstar/(M_PI*R[i]*R[i]);
      }
   }

  return;
}

double intForDeltaSigmaStar(double log10Mstar, void *p)
{

   const Model *model = ((params *)p)->model;
   double r = ((params *)p)->r;
   double z = ((params *)p)->z;

   double Mstar = pow(10.0, log10Mstar);

   double dlog10 = 0.001;

   /* This is M/r^2 * SMF */
   return 1.0e-12*Mstar/(r*r)*ngal_den(model, LNMH_MAX, log10Mstar - dlog10/2.0, log10Mstar + dlog10/2.0, z, all)/dlog10/log(10.0);
}


void xi_gm_cen(const Model *model, double *r, int N, double z, double *result)
{
   /*
    *    Returns the 1-halo central galaxy-dark matter
    *    two-point correlation function.
    */

   int i;
   double Mh, c;

   double rhobar = rho_bar(model, 0.0);

   if(model->hod){

      params p;
      p.model = model;
      p.z = z;
      p.c = NAN;  /* for the HOD model, the concentration(Mh) relationship is fixed */

      double ng = ngal_den(model, LNMH_MAX, model->log10Mstar_min, model->log10Mstar_max, z, all);

      for(i=0;i<N;i++){
         p.r       = r[i];
         result[i] = int_gsl(intForxi_gm_cen, (void*)&p, log(Mh_rh(model, r[i], z)), LNMH_MAX, 1.e-3)/ ng /rhobar;
      }

   }else{
      Mh = pow(10.0, model->ggl_log10Mh);
      c = pow(10.0, model->ggl_log10c);
      for(i=0;i<N;i++){
         result[i] = rhoHalo(model, r[i], Mh, c, z) / rhobar;
      }
   }
   return;
}

double intForxi_gm_cen(double logMh, void *p)
{

   const Model *model = ((params *)p)->model;
   double r = ((params *)p)->r;
   double z = ((params *)p)->z;
   double c = ((params *)p)->c;

   double Mh = exp(logMh);

   return Ngal_c(model, Mh, model->log10Mstar_min, model->log10Mstar_max)
      * rhoHalo(model, r, Mh, c, z)
      * dndlnMh(model, Mh, z);
}

void xi_gm_sat(const Model *model, double *r, int N, double z, double *result){

   /*
    *    Returns the fourier transform of a
    *    P1hc given r_s and rho_s.
    *    Both r_s and rho_s depend on cosmology and
    *    redshift but are computed by the wrapper.
    */

   int i;

   /*    FFTLog config */
   double q = 0.0, mu = 0.5;
   int j, FFT_N = 64;
   FFTLog_config *fc = FFTLog_init(FFT_N, KMIN, KMAX, q, mu);
   double *r_FFT = (double *)malloc(FFT_N*sizeof(double));
   double *ar = (double *)malloc(FFT_N*sizeof(double));
   double *logr_FFT = (double *)malloc(FFT_N*sizeof(double));

   /*    parameters to pass to the function */
   params p;
   p.model = model;
   p.z = z;

   /*    fonction with parameters to fourier transform */
   gsl_function Pk;
   Pk.function = &intForxi_gm_sat;
   Pk.params = &p;

   /*    fourier transform... */
   FFTLog(fc, &Pk, r_FFT, ar, -1);

   /*    return values through interpolation */
   gsl_interp_accel *acc = gsl_interp_accel_alloc ();
   gsl_spline *spline = gsl_spline_alloc (gsl_interp_cspline, FFT_N);

   /*    attention: N and FFT_N are different */
   for(j=0;j<FFT_N;j++) logr_FFT[j] = log(r_FFT[j]);
   gsl_spline_init (spline, logr_FFT, ar, FFT_N);

   for(i=0;i<N;i++){
      if (logr_FFT[0] < log(r[i]) && log(r[i]) <  logr_FFT[FFT_N-1] && r[i] < RMAX1){
         result[i] = gsl_spline_eval(spline, log(r[i]), acc)*pow(2.0*M_PI*r[i],-1.5);
      }else{
         result[i] = 0.0;
      }
   }

   /*    free memory */
   free(r_FFT);
   free(ar);
   free(logr_FFT);
   FFTLog_free(fc);

   gsl_spline_free (spline);
   gsl_interp_accel_free (acc);

   return;
}

double intForxi_gm_sat(double k, void *p){

   const Model *model = ((params *)p)->model;
   const double z = ((params *)p)->z;

   return pow(k, 1.5 )* P_gm_satsat(model, k, z);

}

double P_gm_satsat(const Model *model, double k, double z)
{

   double rhobar = rho_bar(model, 0.0);

   if(model->hod){

      params p;
      p.model = model;
      p.k = k;
      p.z = z;
      p.c = NAN;

      double ng = ngal_den(model, LNMH_MAX, model->log10Mstar_min, model->log10Mstar_max, z, all);
      return int_gsl(intForP_gm_satsat, (void*)&p, LNMH_MIN, LNMH_MAX, 1.e-3)/ng/rhobar;

   }else{
      double Mh, c;

      Mh = pow(10.0, model->ggl_log10Mh);
      c = pow(10.0, model->ggl_log10c);

      return Mh * pow(uHalo(model, k, Mh, c, z), 2.0)/rhobar;
   }
}

double intForP_gm_satsat(double logMh, void *p){

   const Model *model = ((params *)p)->model;
   double k = ((params *)p)->k;
   double z = ((params *)p)->z;
   double c = ((params *)p)->c;

   double Mh = exp(logMh);

   return Ngal_s(model, Mh, model->log10Mstar_min, model->log10Mstar_max) * Mh
      * pow(uHalo(model, k, Mh, c, z), 2.0)
      * dndlnMh(model, Mh, z);

}

void xi_gm_twohalo(const Model *model, double *r, int N, double z, double *result){

   /*
    *    Returns the 2-halo galaxy-dark matter
    *    two-point correlation function.
    */


   int i;
   double bias_fac;

   if(model->hod){
      /* FFTLog config */
      double q = 0.0, mu = 0.5;
      int FFT_N = 64;
      FFTLog_config *fc = FFTLog_init(FFT_N, KMIN, KMAX, q, mu);

      /* parameters to pass to the function */
      params p;
      p.model = model;
      p.z = z;
      p.c = NAN;  /*    for the HOD model, the concentration(Mh) relationship is fixed */

      /* fonction with parameters to fourier transform */
      gsl_function Pk;
      Pk.function = &intForxi_gm_twohalo;
      Pk.params = &p;

      double *xidm = malloc(N*sizeof(double));
      xi_m(model, r, N, z, xidm);

      p.ng  = ngal_den(model, LNMH_MAX, model->log10Mstar_min, model->log10Mstar_max, z, all);
      for(i=0;i<N;i++){

         bias_fac  = sqrt(pow(1.0+1.17*xidm[i],1.49)/pow(1.0+0.69*xidm[i],2.09));
         p.logMlim = logM_lim(model, r[i], p.c, z, all);
         p.r  = r[i];
         p.ngp = ngal_den(model, p.logMlim, model->log10Mstar_min, model->log10Mstar_max, z, all);

         if(p.ng < 1.0e-14 || p.ngp < 1.0e-14 || r[i] < RMIN2){
            result[i] = 0.0;
         }else{
            if( !strcmp(model->massDef, "MvirC15")){
   	         result[i] = pow(p.ngp/p.ng, 2.0)*pow(bias_fac, 2.0)*xi_from_Pkr(&Pk, r[i], fc);  /* matches Coupon et al. (2015) (bug) */
            }else{
               result[i] = (p.ngp/p.ng)*pow(bias_fac, 2.0)*xi_from_Pkr(&Pk, r[i], fc);
            }
         }
      }
      FFTLog_free(fc);
   }else{

      double bh;
      xi_m(model, r, N, z, result);

      double Mh = pow(10.0, model->ggl_log10Mh);
      double c = pow(10.0, model->ggl_log10c);

      bh = bias_h(model, Mh, z);
      for(i=0;i<N;i++){
         // DEBUGGING
         //if(r[i] < r_vir(model, Mh, c, z)){
         if(r[i] < rh(model, Mh, NAN, z)){
            result[i] = 0.0;
         }else{
            result[i] *= bh;
         }
      }

   }

   return;

}

double intForxi_gm_twohalo(double k, void *p){
   return pow(k, 1.5 )* P_gm_twohalo(k, p);
}

double P_gm_twohalo(double k, void *p)
{

   const Model *model = ((params *)p)->model;
   const double z = ((params *)p)->z;
   const double ngp = ((params *)p)->ngp;
   const double logMlim = ((params *)p)->logMlim;

   ((params *)p)->k = k;

   return P_m_nonlin(model, k, z)*int_gsl(intForP_twohalo_g, p, LNMH_MIN, logMlim, 1.e-3)*int_gsl(intForP_twohalo_m, p, LNMH_MIN, logMlim, 1.e-3)/ngp;
}

double intForP_twohalo_g(double logMh, void *p){

   const Model *model = ((params *)p)->model;
   const double k = ((params *)p)->k;
   const double z = ((params *)p)->z;
   const double c = ((params *)p)->c;

   const double Mh = exp(logMh);

   return Ngal(model, Mh, model->log10Mstar_min, model->log10Mstar_max) * uHalo(model, k, Mh, c, z)
      * bias_h(model, Mh, z) * dndlnMh(model, Mh, z);
}

double intForP_twohalo_m(double logMh, void *p){

   const Model *model = ((params *)p)->model;
   const double k = ((params *)p)->k;
   const double z = ((params *)p)->z;
   const double c = ((params *)p)->c;

   double Mh = exp(logMh);
   return  (Mh / rho_bar(model, 0.0))* uHalo(model, k, Mh, c, z)
      * bias_h(model, Mh, z) * dndlnMh(model, Mh, z);
}
