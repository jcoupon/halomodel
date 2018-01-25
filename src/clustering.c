/*
 *    clustering.c
 *    halomodel library
 *    Jean Coupon 2016 - 2017
 */

#include "clustering.h"


void wOfThetaFromXi(const Model *model, double *theta, int N, double z, double *u, int Ninter, double *xi, double *result)
{
   /*
    *    Project inpu xi using Limber equation
    *    and return w(theta).
    *    See Bartelmann & Schneider (2001) eq. (2.79),
    *    Tinker et al. 2010 eq. (3), Ross et al. 2009 eq (27), ...
    *    Theta in degrees.
    */

   double nsqr_dzdr;

   /*    tabulate xi(r) */
   int i,j,k;
   double *logu = malloc(Ninter*sizeof(double));
   double umin = u[0], umax = u[Ninter-1];
   double dlogu = log(umax/umin)/(double)Ninter;

   for(i=0;i<Ninter;i++){
      logu[i] = log(u[i]);
   }

   /*    interpolate xi(r) */
   gsl_interp_accel *acc = gsl_interp_accel_alloc();
   gsl_spline *spline = gsl_spline_alloc (gsl_interp_cspline, Ninter);
   gsl_spline_init(spline, logu, xi, Ninter);

   /*    out of xi(r) limits, use the last two points
   and compute a linear extrapolation, y = ax+b */
   double a = (xi[Ninter-2]-xi[Ninter-1])/(u[Ninter-2]-u[Ninter-1]);
   double b = xi[Ninter-1] - a*u[Ninter-1];

   double deg_to_rad_sqr = pow(M_PI/180.0, 2.0);
   double r, x, sum;

   double nz_Norm = trapz(model->wtheta_nz_z, model->wtheta_nz, model->wtheta_nz_N);
   double nz_dz = model->wtheta_nz_z[1] - model->wtheta_nz_z[0];

   /*    Limber equation - project xi to get w */
   for(i=0;i<N;i++){ /* loop over theta */
      result[i] = 0.0;
      /*    loop over z */
      for (j=0; j<model->wtheta_nz_N; j++) {
         /*    loop over u */
         x = DM(model, z, 0);
         sum = 0.0;
         for (k=0;k<Ninter;k++) {
            r = sqrt(u[k]*u[k] + x*x*theta[i]*theta[i]*deg_to_rad_sqr);
            if (log(r) < logu[Ninter-1]) {
               /*    to unsure log(r) lies within interpolation limits */
               sum += u[k]*gsl_spline_eval(spline,log(r),acc);
            }else{
               /*    linear extrapolation with last 2 points */
               // sum += u[k]*(a*r+b);
            }
         }
         nsqr_dzdr = pow(model->wtheta_nz[j], 2.0)/ dr_dz(model, z);

         result[i] += nsqr_dzdr * sum;
      }

      result[i] *= 2.0*nz_dz*dlogu/pow(nz_Norm, 2.0);
   }

   //free(xi);
   //free(u);
   free(logu);
   gsl_spline_free (spline);
   gsl_interp_accel_free (acc);

   return;
}



void wOfTheta(const Model *model, double *theta, int N, double z, int obs_type, double *result)
{
   /*
    *    First compute xi, project using Limber equation at mean z
    *    and return w(theta).
    *    See Bartelmann & Schneider (2001) eq. (2.79),
    *    Tinker et al. 2010 eq. (3), Ross et al. 2009 eq (27), ...
    *    Theta in degrees.
    *    Ninter is the number of points that sample xi.The speed of the code
    *    is basically inversely proportional to this number...choose
    *    it carefuly!! You can also play with N in FFTLog routines.
    */

   if(obs_type == all){
      wOfThetaAll(model, theta, N, z, result);
      return;
   }

   double nsqr_dzdr;

   /*    tabulate xi(r) */
   int i,j,k,Ninter = 40;
   double *u = malloc(Ninter*sizeof(double));
   double *logu = malloc(Ninter*sizeof(double));
   double umin = RMIN, umax = RMAX;
   double dlogu = log(umax/umin)/(double)Ninter;

   for(i=0;i<Ninter;i++){
      logu[i] = log(umin)+dlogu*(double)i;
      u[i] = exp(logu[i]);
   }

   double *xi = malloc(Ninter*sizeof(double));
   xi_gg(model, u, Ninter, z, obs_type, xi);

   /*    interpolate xi(r) */
   gsl_interp_accel *acc = gsl_interp_accel_alloc();
   gsl_spline *spline = gsl_spline_alloc (gsl_interp_cspline, Ninter);
   gsl_spline_init(spline, logu, xi, Ninter);

   double deg_to_rad_sqr = pow(M_PI/180.0, 2.0);
   double r, x, sum;

   double nz_Norm = trapz(model->wtheta_nz_z, model->wtheta_nz, model->wtheta_nz_N);
   double nz_dz = model->wtheta_nz_z[1] - model->wtheta_nz_z[0];

   /*    Limber equation - project xi to get w */
   for(i=0;i<N;i++){ /* loop over theta */
      result[i] = 0.0;
      /*    loop over z */
      for (j=0; j<model->wtheta_nz_N; j++) {
         /*    loop over u */
         x = DM(model, z, 0);
         sum = 0.0;
         for (k=0;k<Ninter;k++) {
            r = sqrt(u[k]*u[k] + x*x*theta[i]*theta[i]*deg_to_rad_sqr);
            if (log(r) < logu[Ninter-1]) {
               /* to unsure log(r) lies within interpolation limits */
               sum += u[k]*gsl_spline_eval(spline,log(r),acc);
   	      }
         }
         nsqr_dzdr = pow(model->wtheta_nz[j], 2.0)/ dr_dz(model, z);

         result[i] += nsqr_dzdr * sum;
      }

      result[i] *= 2.0*nz_dz*dlogu/pow(nz_Norm, 2.0);
   }

   free(xi);
   free(u);
   free(logu);
   gsl_spline_free (spline);
   gsl_interp_accel_free (acc);

   return;
}

void xi_gg(const Model *model, double *r, int N, double z, int obs_type, double *result)
{
   /*
    *    Computes xi(r)
    */

   int i;
   double *result_tmp;

   switch (obs_type){
      case censat:
         xi_gg_censat(model, r, N, z, result);
         break;
      case satsat:
         xi_gg_satsat(model, r, N, z, result);
         break;
      case twohalo:
         xi_gg_twohalo(model, r, N, z, result);
         break;
      case all:
         result_tmp = (double *)malloc(N*sizeof(double));
         xi_gg_censat(model, r, N, z, result_tmp); for(i=0;i<N;i++){result[i] = result_tmp[i];}
         xi_gg_satsat(model, r, N, z, result_tmp); for(i=0;i<N;i++){result[i] += result_tmp[i];}
         xi_gg_twohalo(model, r, N, z, result_tmp); for(i=0;i<N;i++){result[i] += result_tmp[i];}
         free(result_tmp);
         break;
   }
   return;
}

void wOfThetaAll(const Model *model, double *theta, int N, double z, double *result)
{

   int i;

   double *result_tmp = (double *)malloc(N*sizeof(double));

   wOfTheta(model, theta, N, z, censat,  result_tmp); for(i=0;i<N;i++){result[i]  = result_tmp[i];}
   wOfTheta(model, theta, N, z, satsat,  result_tmp); for(i=0;i<N;i++){result[i] += result_tmp[i];}
   wOfTheta(model, theta, N, z, twohalo, result_tmp); for(i=0;i<N;i++){result[i] += result_tmp[i];}

   free(result_tmp);

   return;
}


void xi_gg_censat(const Model *model, double *r, int N, double z, double *result)
{
   /*
    *    Returns the central/sagtellite
    *    two-point correlation function.
    */

   int i;

   if(model->hod){

      params p;
      p.model = model;
      p.z = z;
      p.c = NAN;  // for the HOD model, the concentration(Mh) relationship is fixed

      double ng = ngal_den(model, LNMH_MAX, model->log10Mstar_min, model->log10Mstar_max, z, all);
      p.ng = ng;

      for(i=0;i<N;i++){
         p.r = r[i];
         if(r[i] < RMAX1){
            result[i] = int_gsl(intForxi_gg_censat, (void*)&p, log(Mh_rh(model, r[i], z)), LNMH_MAX, 1.e-3)/pow(ng, 2.0);
         }else{
            result[i] = 0.0;
         }
      }

   }else{
      printf("w(theta) not supported in non-HOD models. Exiting..."); exit(-1);
   }
   return;
}

double intForxi_gg_censat(double logMh, void *p)
{

   const Model *model = ((params *)p)->model;
   double r = ((params *)p)->r;
   double z = ((params *)p)->z;
   double c = ((params *)p)->c;
   // double ng = ((params *)p)->ng;

   double Mh = exp(logMh);

   // DEBUGGING
   //return 0.5*Ngal_c(model, Mh, model->log10Mstar_min, model->log10Mstar_max)* Ngal_s(model, Mh, model->log10Mstar_min, model->log10Mstar_max)
   //   * rhoHalo(model, r, Mh, c, z)
   //   * dndlnMh(model, Mh, z) /(0.5*ng*ng) / Mh;

   return Ngal_c(model, Mh, model->log10Mstar_min, model->log10Mstar_max)
      *Ngal_s(model, Mh, model->log10Mstar_min, model->log10Mstar_max)
      *rhoHalo(model, r, Mh, c, z)
      *dndlnMh(model, Mh, z) / Mh;
   // return 0.0;
}


void xi_gg_satsat(const Model *model, double *r, int N, double z, double *result)
{

   /*
    *    Returns the fourier transform of the
    *    sat-sat two-point correlation function.
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
   Pk.function = &intForxi_gg_satsat;
   Pk.params  = &p;

   /*    fourier transform... */
   FFTLog(fc, &Pk, r_FFT, ar, -1);

   /*    return values through interpolation */
   gsl_interp_accel *acc = gsl_interp_accel_alloc ();
   gsl_spline *spline    = gsl_spline_alloc (gsl_interp_cspline, FFT_N);

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

double intForxi_gg_satsat(double k, void *p){

   const Model *model = ((params *)p)->model;
   const double z = ((params *)p)->z;

   return pow(k, 1.5 )* P_gg_satsat(model, k, z);

}

double P_gg_satsat(const Model *model, double k, double z)
{

   if(model->hod){

      params p;
      p.model = model;
      p.k = k;
      p.z = z;
      p.c = NAN;

      double ng = ngal_den(model, LNMH_MAX, model->log10Mstar_min, model->log10Mstar_max, z, all);

      return 0.5*int_gsl(intForP_gg_satsat, (void*)&p, LNMH_MIN, LNMH_MAX, 1.e-3)/pow(ng, 2.0);


   }else{
      printf("w(theta) not supported in non-HOD models. Exiting..."); exit(-1);
   }
}

double intForP_gg_satsat(double logMh, void *p){

   const Model *model = ((params *)p)->model;
   double k = ((params *)p)->k;
   double z = ((params *)p)->z;
   double c = ((params *)p)->c;

   double Mh = exp(logMh);


   return  pow(Ngal_s(model, Mh, model->log10Mstar_min, model->log10Mstar_max), 2.0)
      * pow(uHalo(model, k, Mh, c, z), 2.0)
      * dndlnMh(model, Mh, z);


   // The code below is equivalent to the sum of cen-sat + sat+sat profiles

   /*
   return pow(Ngal_s(model, Mh, model->log10Mstar_min, model->log10Mstar_max), 2.0)
            * pow(uHalo(model, k, Mh, c, z), 2.0)
            * dndlnMh(model, Mh, z)
            + 2.0*Ngal_c(model, Mh, model->log10Mstar_min, model->log10Mstar_max)*Ngal_s(model, Mh, model->log10Mstar_min, model->log10Mstar_max)
            * uHalo(model, k, Mh, c, z)
            * dndlnMh(model, Mh, z);
   */

}


void xi_gg_twohalo(const Model *model, double *r, int N, double z, double *result){

   /*
    *    Returns the 2-halo galaxy-dark matter
    *    two-point correlation function.
    */
   int i;
   double bias_fac;

   if(model->hod){
      /*    FFTLog config */
      double q = 0.0, mu = 0.5;
      int FFT_N = 64;
      FFTLog_config *fc = FFTLog_init(FFT_N, KMIN, KMAX, q, mu);

      /*    parameters to pass to the function */
      params p;
      p.model = model;
      p.z = z;
      p.c = NAN;  // for the HOD model, the concentration(Mh) relationship is fixed

      /*    fonction with parameters to fourier transform */
      gsl_function Pk;
      Pk.function = &intForxi_gg_twohalo;
      Pk.params = &p;

      double *xidm = malloc(N*sizeof(double));
      xi_m(model, r, N, z, xidm);

      p.ng  = ngal_den(model, LNMH_MAX, model->log10Mstar_min, model->log10Mstar_max, z, all);
      for(i=0;i<N;i++){

			bias_fac = sqrt(pow(1.0+1.17*xidm[i],1.49)/pow(1.0+0.69*xidm[i],2.09));

         // Halo exclusion
         if (model->haloExcl) {
            p.logMlim = logM_lim(model, r[i], p.c, z, all);
            p.r = r[i];
            p.ngp = ngal_den(model, p.logMlim, model->log10Mstar_min, model->log10Mstar_max, z, all);
         }else{
            p.logMlim = LNMH_MAX;
            p.ngp = p.ng;
         }

         if(p.ng < 1.0e-14 || p.ngp < 1.0e-14 || r[i] < RMIN2){
            result[i] = 0.0;
         }else{
   	      result[i] = pow(p.ngp/p.ng*bias_fac, 2.0)*xi_from_Pkr(&Pk, r[i], fc);
         }
      }
      FFTLog_free(fc);
   }else{
      printf("w(theta) not supported in non-HOD models. Exiting..."); exit(-1);
   }

   return;

}

double intForxi_gg_twohalo(double k, void *p){
   return pow(k, 1.5 )* P_gg_twohalo(k, p);
}

double P_gg_twohalo(double k, void *p)
{

   const Model *model = ((params *)p)->model;
   const double z  = ((params *)p)->z;
   const double ngp = ((params *)p)->ngp;
   const double logMlim = ((params *)p)->logMlim;

   ((params *)p)->k = k;

   // DEBUGGING
   return P_m_nonlin(model, k, z)*pow(int_gsl(intForP_gg_twohalo, p, LNMH_MIN, logMlim, 1.e-3)/ngp, 2.0);
   // return P_m_lin(model, k, z)*pow(int_gsl(intForP_gg_twohalo, p, LNMH_MIN, logMlim, 1.e-3)/ngp, 2.0);

}

double intForP_gg_twohalo(double logMh, void *p){

   const Model *model = ((params *)p)->model;
   const double k = ((params *)p)->k;
   const double z = ((params *)p)->z;
   const double c = ((params *)p)->c;

   const double Mh = exp(logMh);

   // DEBUGING
//   return Ngal_c(model, Mh, model->log10Mstar_min, model->log10Mstar_max) * uHalo(model, k, Mh, c, z)
//      * bias_h(model, Mh, z) * dndlnMh(model, Mh, z);

   return Ngal(model, Mh, model->log10Mstar_min, model->log10Mstar_max) * uHalo(model, k, Mh, c, z)
      * bias_h(model, Mh, z) * dndlnMh(model, Mh, z);
}
