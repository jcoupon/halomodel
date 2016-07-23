/* ------------------------------------------------------------ *
 * xray.c	                 				                         *
 * halomodel library                                            *
 * Jean Coupon 2015	         		                            *
 * ------------------------------------------------------------ */

#include "xray.h"

void Ix(double *R, int N, const Model *model, double *result, int type)
/* ---------------------------------------------------------------- *
* Computes projected beta model for X-ray luminosity profiles      *
* See Vikhlinin et al. (2006) and Cavaliere & Fusco-Femiano (1978) *
*                                                                  *
* If model.hod is set to 1, it will use the full               *
* HOD parametrisation, and needs a halo mass function. Otherwise   *
* if set to 0, it will only use the three parameters that controls *
* the shape of a X-ray profile.                                    *
* ---------------------------------------------------------------- */
{

   /* interpolate to speed up integration  */
   int i, j, k, Ninter = 40;
   if (type == XB){
      Ninter = 256;
   }
   double *logrinter  = (double *)malloc(Ninter*sizeof(double));
   double *rinter     = (double *)malloc(Ninter*sizeof(double));
   double rinter_min  = 1.e-3;
   double rinter_max  = 1.e+2;
   double dlogrinter  = log(rinter_max/rinter_min)/(double)Ninter;


   for(i=0;i<Ninter;i++){
      logrinter[i] = log(rinter_min)+dlogrinter*(double)i;
      rinter[i]    = exp(logrinter[i]);
   }

   double *xi = (double *)malloc(Ninter*sizeof(double));

   switch (type){
      case cen:
      Ix1hc(rinter, Ninter, model, xi);
      break;
      case sat:
      Ix1hs(rinter, Ninter, model, xi);
      break;
      case XB:
      IxXB(rinter, Ninter, model, xi);
      break;
   }

   /* interpolate xi(r) */
   gsl_interp_accel *acc = gsl_interp_accel_alloc();
   gsl_spline *spline    = gsl_spline_alloc (gsl_interp_cspline, Ninter);
   gsl_spline_init(spline, logrinter, xi, Ninter);

   params p;
   p.acc       = acc;
   p.spline    = spline;
   p.eps       = 1.0e-4;
   p.logrmin   = logrinter[0];
   p.logrmax   = logrinter[Ninter-1];

   //printf("%f %f %f\n", model->XMM_PSF_A, model->XMM_PSF_r_c, model->XMM_PSF_alpha);
   //printf("%f %f\n", model->log10Mstar_min, model->log10Mstar_max);


   /* If convolved by PSF */
   /* TO DO: move into external function */
   if (model->XMM_PSF_A > 0.0){
      /*
      First interpolate the projected signal over wide range,
      keeping the previous interpolation space
      */
      double *Ix = (double *)malloc(Ninter*sizeof(double));

      switch (type){

         /* For central and satellites, computed the projected 3D profile */
         case cen: case sat:

         for(i=0;i<Ninter;i++){
            p.R = rinter[i];
            Ix[i] = 2.0*int_gsl(intForIx, (void*)&p, log(rinter_min), log(rinter_max), p.eps);
         }
         break;


         /* For X-ray binaries, it is simply the de Vaucouleurs profile */
         case XB:

         for(i=0;i<Ninter;i++){
            double logr = log( rinter[i]);
            Ix[i] = gsl_spline_eval(spline, logr, acc);
         }
         break;
      }
      gsl_spline_init(spline, logrinter, Ix, Ninter);

      /* theta and R prime */
      int N_PSF           = 64;
      double Rpp;
      double *theta       =  (double *)malloc(N_PSF*sizeof(double));
      double *Rp          =  (double *)malloc(N_PSF*sizeof(double));
      double *intForTheta =  (double *)malloc(N_PSF*sizeof(double));
      double *intForRp    =  (double *)malloc(N_PSF*sizeof(double));
      for(i=0;i<N_PSF;i++){
         Rp[i]    = exp(log(rinter_min)+dlogrinter*(double)i);
         theta[i] = 0.0+2.0*M_PI / (double)(N_PSF-1) * (double)i;
      }

      /* Main loop */
      for(i=0;i<N;i++){
         /* loop over R' - trapeze integration */
         for(j=0;j<N_PSF;j++){
            /* loop over theta - trapeze integration  */
            for(k=0;k<N_PSF;k++){
               Rpp = sqrt(R[i]*R[i] + Rp[j]*Rp[j] - 2.0*R[i]*Rp[j]*cos(theta[k]));
               if(logrinter[0] < log(Rpp) && log(Rpp) < logrinter[Ninter-1]){
                  intForTheta[k] = gsl_spline_eval(spline, log(Rpp), acc);
               }
            }
            intForRp[j] = King(Rp[j], model->XMM_PSF_A, model->XMM_PSF_r_c, model->XMM_PSF_alpha) * trapz(theta, intForTheta, N_PSF);
         }

         result[i] = 1.0 / (2.0 * M_PI) * trapz(Rp, intForRp, N_PSF);

      }


      /* test
      for(i=0;i<N;i++){
      if(logrinter[0] < log(R[i]) && log(R[i]) < logrinter[Ninter-1]){
      result[i] = gsl_spline_eval(spline, log(R[i]), acc);
      }
      }

      */
      free(intForTheta);
      free(intForRp);
      free(theta);
      free(Rp);
      free(Ix);
   }else{
      /* Or simply returns the result */
      switch (type){

         /* For central and satellites, computed the projected 3D profile */
         case cen: case sat:
         for(i=0;i<N;i++){
            p.R = R[i];
            result[i] = 2.0*int_gsl(intForIx, (void*)&p, log(rinter_min), log(rinter_max), p.eps);
         }
         break;

         /* For X-ray binaries, it is simply the de Vaucouleurs profile */
         case XB:
         for(i=0;i<N;i++){
            double logr = log(R[i]);

            if(logrinter[0] < logr && logr < logrinter[Ninter-1]){
               result[i] = gsl_spline_eval(spline, logr, acc);
            }else
            result[i] = 0.0;
         }
         break;
      }


   }

   free(xi);
   free(rinter);
   free(logrinter);
   gsl_spline_free (spline);
   gsl_interp_accel_free (acc);
}

double intForIx(double logz, void *p)
{
   double result         = 0.0;
   double z              = exp(logz);
   double logrmin        = ((params *)p)->logrmin;
   double logrmax        = ((params *)p)->logrmax;
   double R              = ((params *)p)->R;
   gsl_interp_accel *acc = ((params *)p)->acc;
   gsl_spline *spline    = ((params *)p)->spline;

   double r    = sqrt(R*R + z*z);
   double logr = log(r);

   if(logrmin < logr && logr < logrmax){
      result = gsl_spline_eval(spline, logr, acc) * z;
   }

   return result;
}

void IxXB(double *r, int N, const Model *model, double *result){

   /*
   Returns the luminosity profile of
   X-Ray binary stars
   */
   int i;

   if (model->Ix_XB_Re < 0.0 || model->Ix_XB_L < 0.0){
      for(i=0;i<N;i++){
         result[i] = 0.0;
      }
      return;
   }

   /* Re_XB in Mpc */
   double Re = model->Ix_XB_Re;

   /* total X-ray luminosity of binary stars in CR Mpc-2*/
   double L = model->Ix_XB_L;

   double Ie = L / (7.215 * M_PI * Re * Re);

   for(i=0;i<N;i++){
      /* De vaucouleur profile */
      result[i] = Ie * exp(-7.669*(pow(r[i]/Re, 1.0/4.0) - 1.0));
   }

   return;
}


void Ix1hc(double *r, int N, const Model *model, double *result){

   /*
   Returns the 1-halo central X-ray
   3D luminosity profile.

   SB \propto rho(r) ^ 2
   */
   int i;

   if(model->hod){


      // ************************************************************ //
      // hack for tests - Don't integrate through dndM
      // only returns model at mean Mstar
      //printf("mh = %f\n", model->log10M1);
      /*
      for(i=0;i<N;i++){
         result[i] = Ix3D(r[i],  model->log10M1, model);

      }
      return
      */
      // ************************************************************ //


      double z = 0.0;

      params p;
      p.model = model;
      double ng = ngal_den(model, LNMH_MAX, model->log10Mstar_min, model->log10Mstar_max, z, cen)
      + ngal_den(model, LNMH_MAX, model->log10Mstar_min, model->log10Mstar_max, z, sat);

      for(i=0;i<N;i++){
         p.r       = r[i];
         result[i] = int_gsl(intForIx1hc, (void*)&p, LNMH_MIN, LNMH_MAX, 1.e-3)/ng;

      }


   }else{
      for(i=0;i<N;i++){
         result[i] = Ix3D(r[i], -1.0, model);
      }
   }
   return;
}


double intForIx1hc(double logMh, void *p) {
   /* Integrand for rhoGas if HOD model */

   const Model *model    = ((params *)p)->model;
   double r              = ((params *)p)->r;

   double log10Mh = logMh/log(10.0);

   double z = 0.0;

   return Ngal_c(model, log10Mh, model->log10Mstar_min, model->log10Mstar_max)
   * Ix3D(r, log10Mh, model)
   * dndlnMh(model, z, log10Mh);


}



void Ix1hs(double *r, int N, const Model *model, double *result){

   /*
   Returns the (backward) fourier transform of a
   x-ray luminosity power spectrum.
   */

   int i;

   /* FFTLog config */
   double q = 0.0, mu = 0.5;
   int j, FFT_N = 64;
   FFTLog_config *fc = FFTLog_init(FFT_N, KMIN, KMAX, q, mu);
   double *r_FFT     = (double *)malloc(FFT_N*sizeof(double));
   double *ar        = (double *)malloc(FFT_N*sizeof(double));
   double *logr_FFT  = (double *)malloc(FFT_N*sizeof(double));

   /* parameters to pass to the function */
   params p;
   p.model = model;

   /* fonction with parameters to fourier transform */
   gsl_function Pk;
   Pk.function = &intForIx1hs;
   Pk.params   = &p;

   /* fourier transform... */
   FFTLog(fc, &Pk, r_FFT, ar, -1);

   /* return values through interpolation */
   gsl_interp_accel *acc = gsl_interp_accel_alloc ();
   gsl_spline *spline    = gsl_spline_alloc (gsl_interp_cspline, FFT_N);

   /* attention: N and FFT_N are different */
   for(j=0;j<FFT_N;j++) logr_FFT[j] = log(r_FFT[j]);
   gsl_spline_init (spline, logr_FFT, ar, FFT_N);

   for(i=0;i<N;i++){
      if (logr_FFT[0] < log(r[i]) && log(r[i]) <  logr_FFT[FFT_N-1] && r[i] < RMAX1){
         result[i] = gsl_spline_eval(spline, log(r[i]), acc)*pow(2.0*M_PI*r[i],-1.5);
      }else{
         result[i] = 0.0;
      }
   }

   /* free memory */
   free(r_FFT);
   free(ar);
   free(logr_FFT);
   FFTLog_free(fc);

   gsl_spline_free (spline);
   gsl_interp_accel_free (acc);

   return;
}

double intForIx1hs(double k, void *p){

   const Model *model   =  ((params *)p)->model;
   return pow(k, 1.5 )* PIx1hs(k, model);

}


double PIx1hs(double k, const Model *model)
{
   double result;

   if(model->hod){


      // ************************************************************ //
      // hack for tests - Don't ingegrate through dndM
      // only returns model at mean Mstar
      //printf("mh = %f\n", model->log10M1);
      ///*
      //double Norm = NormIx3D(model, model->pi_max, model->rh_trunc);
      //uIx3D(&k, 1, model, model->pi_max, &result);
      //return   Norm * pow(result, 2.0);
      //*/
      // ************************************************************ //


      //printf("min= %f, max %f\n", model->log10Mstar_min, model->log10Mstar_max);


      params p;
      p.model = model;
      p.k     = k;

      double z = 0.0;

      double ng = ngal_den(model, LNMH_MAX, model->log10Mstar_min, model->log10Mstar_max, z, cen)
      + ngal_den(model, LNMH_MAX, model->log10Mstar_min, model->log10Mstar_max, z, sat);

      //printf("%f\n", ngal_den(model, LNMH_MAX, model->log10Mstar_min, model->log10Mstar_max, sat)/ng);
      //exit(-1);

      return int_gsl(intForPIx1hs, (void*)&p, LNMH_MIN, LNMH_MAX, 1.e-3)/ng;


   }else{

      // TODO change to r_vir
      //double Norm = NormIx3D(model, -1, model->rh_trunc);
      double Norm = NormIx3D(model, -1, 20.0);

      uIx3D(&k, 1, model, -1, &result);
      return   Norm * pow(result, 2.0);
   }

}


double intForPIx1hs(double logMh, void *p) {
   /* Integrand for rhoGas if HOD model */

   const Model *model    = ((params *)p)->model;
   double k              = ((params *)p)->k;

   double log10Mh = logMh/log(10.0);
   double Mh = exp(logMh);

   double result, Norm = NormIx3D(model, log10Mh, rh(model, Mh, 0.0));

   uIx3D(&k, 1, model, log10Mh, &result);


   double z = 0.0;

   return Ngal_s(model, log10Mh, model->log10Mstar_min, model->log10Mstar_max)
      * Norm * pow(result, 2.0)
      * dndlnMh(model, z, log10Mh);

}


void uIx3D(double *k, int N, const Model *model, double log10Mh, double *result){

   /*
   Returns the normalised (out to rh_trunc radius)
   fourier transform of the brightness profile.

   The constants are set by the wrapper and depend on
   halo properties, redshift, etc., so that all the
   quantities depending on cosmology are managed by
   the wrapper.

   */

   int i;

   /* FFTLog config */
   double q = 0.0, mu = 0.5;
   int j, FFT_N = 64;
   /* Note: a signicant increase in upper limit
   should be compensated by a similar increase
   in lower limit and presumably in FFT_N as well.
   Otherwise it creates rather dramatic
   wiggles.
   */

   FFTLog_config *fc = FFTLog_init(FFT_N, pow(10.0, model->gas_log10r_c) * 1.e-2, 20.0, q, mu);
   double *k_FFT     = (double *)malloc(FFT_N*sizeof(double));
   double *ar        = (double *)malloc(FFT_N*sizeof(double));
   double *logk_FFT  = (double *)malloc(FFT_N*sizeof(double));

   /* parameters to pass to the function */
   params p;
   p.model    = model;
   p.log10Mh  = log10Mh;

   /* fonction with parameters to fourier transform */
   gsl_function Pk;
   Pk.function = &intForUIx3D;
   Pk.params   = &p;

   /* fourier transform... */
   FFTLog(fc, &Pk, k_FFT, ar, -1);

   /* return values through interpolation */
   gsl_interp_accel *acc = gsl_interp_accel_alloc ();
   gsl_spline *spline    = gsl_spline_alloc (gsl_interp_cspline, FFT_N);

   /* attention: N and FFT_N are different */
   for(j=0;j<FFT_N;j++) logk_FFT[j] = log(k_FFT[j]);
   gsl_spline_init (spline, logk_FFT, ar, FFT_N);

   double Mh = pow(10.0, log10Mh);

   /* Normalisation */
   double Norm;
   if(model->hod){
      Norm = NormIx3D(model, log10Mh, rh(model, Mh, 0.0));
   }else{

      // TODO change to R_VIR
      // Norm = NormIx3D(model, log10Mh, model->rh_trunc);
      Norm = NormIx3D(model, log10Mh, 20.0);
   }

   for(i=0;i<N;i++){
      if (logk_FFT[0] < log(k[i]) && log(k[i]) < logk_FFT[FFT_N-1]){
         result[i] = gsl_spline_eval(spline, log(k[i]), acc)*pow(k[i],-1.5) / Norm;
      }else{
         result[i] = 0.0;
      }
   }

   /* free memory */
   free(k_FFT);
   free(ar);
   free(logk_FFT);
   FFTLog_free(fc);

   gsl_spline_free (spline);
   gsl_interp_accel_free (acc);

   return;

}

double intForUIx3D(double r, void *p){
   /*
   Integrand for uUIx3D().
   */

   const Model *model = ((params *)p)->model;
   double log10Mh     = ((params *)p)->log10Mh;

   return pow(2.0*M_PI*r, 1.5) * Ix3D(r, log10Mh, model);

}


double Ix3D(double r, double log10Mh,  const Model *model){
   /*
   Returns the 3D gas brightness assuming
   a 3D gas profile

   Ix3D \propto rhoGas^2
   */

   return pow(rhoGas(r, log10Mh, model), 2.0);

}


double MGas(const Model *model, double log10Mh, double rmax){
   /*
   Returns the total Mgas out to rh_trunc
   */

   /* parameters to pass to the function */
   params p;
   p.model   = model;
   p.eps     = 1.0e-4;
   p.log10Mh = log10Mh;

   double result = int_gsl(intForMGas, (void*)&p, log(1.e-6), log(rmax), p.eps);

   return result;
}

double intForMGas(double logr, void *p){

   const Model *model = ((params *)p)->model;
   double r           = exp(logr);
   double log10Mh     = ((params *)p)->log10Mh;

   return r * r * 4.0 * M_PI * rhoGas(r, log10Mh, model) * r;
}


double rhoGas(double r, double log10Mh, const Model *model){
   /*
   Returns a 3D gas density profile given a set
   of parameters, truncated at r = rh_trunc

   If computed using the halo model, the gas
   profile is integrated over the halo mass
   function times the galaxy HOD.

   Depends on log10Mh if log10Mh > 1
   */

   double Mh = pow(10.0, log10Mh);
   if(log10Mh > 1.0){


      if (r < rh(model, Mh, 0.0)){

         double log10rho0 = inter_gas_log10rho0(model, log10Mh);
         double beta      = inter_gas_log10beta(model, log10Mh);
         double log10r_c  = inter_gas_log10r_c(model, log10Mh);

         return betaModel(r, pow(10.0, log10rho0), pow(10.0, beta), pow(10.0, log10r_c));

      }else{
         return 0.0;
      }

   }else{

      // TODO change to r_vir
      // if (r < model->rh_trunc){
      if (r < 20.0){
         return betaModel(r, pow(10.0, model->gas_log10rho0), pow(10.0, model->gas_log10beta), pow(10.0, model->gas_log10r_c));
      }else{
         return 0.0;
      }
   }
}



#define LOG10MH_1 12.9478
#define LOG10MH_2 13.2884
#define LOG10MH_3 13.6673
#define LOG10MH_4 14.0646

#define CONCAT2(a, b)  a##b
#define CONCAT(a, b) CONCAT2(a, b)


#define PARA gas_log10rho0
double CONCAT(inter_, PARA)(const Model *model, double log10Mh){

   return model->CONCAT(PARA, _1) + model->CONCAT(PARA, _2) * (log10Mh-14.0);

/*
   static int firstcall = 1;
   static gsl_interp_accel *acc;
   static gsl_spline *spline;

   static double *t_log10Mh;
   static double *t_PARA;

   if(firstcall){
      firstcall = 0;
      acc = gsl_interp_accel_alloc();
      spline    = gsl_spline_alloc (gsl_interp_cspline, 4);
      t_log10Mh =  (double *)malloc(4*sizeof(double));
      t_PARA    =  (double *)malloc(4*sizeof(double));

      t_log10Mh[0] = LOG10MH_1;
      t_log10Mh[1] = LOG10MH_2;
      t_log10Mh[2] = LOG10MH_3;
      t_log10Mh[3] = LOG10MH_4;

   }

   t_PARA[0] = model->CONCAT(PARA, _1);
   t_PARA[1] = model->CONCAT(PARA, _2);
   t_PARA[2] = model->CONCAT(PARA, _3);
   t_PARA[3] = model->CONCAT(PARA, _4);

   gsl_spline_init(spline, t_log10Mh,  t_PARA, 4);

   if (t_log10Mh[0] < log10Mh && log10Mh < t_log10Mh[3]){
      return gsl_spline_eval(spline, log10Mh, acc);
   }else if (log10Mh > t_log10Mh[3]){
      return model->CONCAT(PARA, _4);
   }else{
      return model->CONCAT(PARA, _1);
   }

   */
}
#undef PARA

#define PARA gas_log10beta
double CONCAT(inter_, PARA)(const Model *model, double log10Mh){

   return model->CONCAT(PARA, _1) + model->CONCAT(PARA, _2) * (log10Mh-14.0);

   /*
   static int firstcall = 1;
   static gsl_interp_accel *acc;
   static gsl_spline *spline;

   static double *t_log10Mh;
   static double *t_PARA;

   if(firstcall){
      firstcall = 0;
      acc = gsl_interp_accel_alloc();
      spline    = gsl_spline_alloc (gsl_interp_cspline, 4);
      t_log10Mh =  (double *)malloc(4*sizeof(double));
      t_PARA    =  (double *)malloc(4*sizeof(double));

      t_log10Mh[0] = LOG10MH_1;
      t_log10Mh[1] = LOG10MH_2;
      t_log10Mh[2] = LOG10MH_3;
      t_log10Mh[3] = LOG10MH_4;


   }

   t_PARA[0] = model->CONCAT(PARA, _1);
   t_PARA[1] = model->CONCAT(PARA, _2);
   t_PARA[2] = model->CONCAT(PARA, _3);
   t_PARA[3] = model->CONCAT(PARA, _4);

   gsl_spline_init(spline, t_log10Mh,  t_PARA, 4);

   if (t_log10Mh[0] < log10Mh && log10Mh < t_log10Mh[3]){
      return gsl_spline_eval(spline, log10Mh, acc);
   }else if (log10Mh > t_log10Mh[3]){
      return model->CONCAT(PARA, _4);
   }else{
      return model->CONCAT(PARA, _1);
   }
   */
}
#undef PARA


#define PARA gas_log10r_c
double CONCAT(inter_, PARA)(const Model *model, double log10Mh){

   return model->CONCAT(PARA, _1) + model->CONCAT(PARA, _2) * (log10Mh-14.0);


/*


   static int firstcall = 1;
   static gsl_interp_accel *acc;
   static gsl_spline *spline;

   static double *t_log10Mh;
   static double *t_PARA;

   if(firstcall){
      firstcall = 0;
      acc = gsl_interp_accel_alloc();
      spline    = gsl_spline_alloc (gsl_interp_cspline, 4);
      t_log10Mh =  (double *)malloc(4*sizeof(double));
      t_PARA    =  (double *)malloc(4*sizeof(double));

      t_log10Mh[0] = LOG10MH_1;
      t_log10Mh[1] = LOG10MH_2;
      t_log10Mh[2] = LOG10MH_3;
      t_log10Mh[3] = LOG10MH_4;


   }

   t_PARA[0] = model->CONCAT(PARA, _1);
   t_PARA[1] = model->CONCAT(PARA, _2);
   t_PARA[2] = model->CONCAT(PARA, _3);
   t_PARA[3] = model->CONCAT(PARA, _4);

   gsl_spline_init(spline, t_log10Mh,  t_PARA, 4);

   if (t_log10Mh[0] < log10Mh && log10Mh < t_log10Mh[3]){
      return gsl_spline_eval(spline, log10Mh, acc);
   }else if (log10Mh > t_log10Mh[3]){
      return model->CONCAT(PARA, _4);
   }else{
      return model->CONCAT(PARA, _1);
   }

   */

}
#undef PARA


double NormIx3D(const Model *model, double log10Mh,  double rmax){
   /*
   Returns the total surface brightness out to rh_trunc
   */

   /* parameters to pass to the function */
   params p;
   p.model   = model;
   p.eps     = 1.0e-4;
   p.log10Mh = log10Mh;

   double result = int_gsl(intForNormIx3D, (void*)&p, log(1.e-6), log(rmax), p.eps);

   return result;
}

double intForNormIx3D(double logr, void *p){

   const Model *model =  ((params *)p)->model;
   double r           = exp(logr);
   double log10Mh     =  ((params *)p)->log10Mh;

   return Ix3D(r, log10Mh, model) * r;
}


double betaModel(double r, double rho0, double beta, double r_c){
   /*
   Returns a beta-model profile given rho0, beta, and rc.
   */

   return rho0 * pow(1.0 + pow(r/r_c, 2.0), -3.0*beta/2.0);
}


double betaModelSqProj(double r, double rho0, double beta, double r_c){
   /*
   Returns a beta-model profile square projected given rho0, beta, and rc.
   See Hudson D. S. et al. (2010)
   */

   double I0 = pow(rho0, 2.0) * r_c * sqrt(M_PI) * gsl_sf_gamma(3.0*beta - 0.5) / gsl_sf_gamma(3.0*beta);
   return I0 * pow(1.0 + pow(r/r_c, 2.0), -3.0*beta + 0.5);
}

double bias_log10Mh(const Model *model, double log10Lx){
  /* Returns the mass bias as a function of Lx
    in case of a scatter in the Lx|Mh relations */

  double norm, res;

  params p;
  p.model   = model;
  p.log10Lx = log10Lx;

  p.a     = 1.27;
  p.b     = -0.38;
  p.sigma = 0.186;
//  p.sigma = 0.5;

  res  = int_gsl(int_for_PLxGivenMh,      &p, LNMH_MIN, LNMH_MAX, 1.0e-10);
  norm = int_gsl(int_for_PLxGivenMh_norm, &p, LNMH_MIN, LNMH_MAX, 1.0e-10);

  double log10Mh_biased = res/norm;
  double log10Mh        = (log10Lx - 44.0) * 1/p.a - p.b/p.a + 14.48;

  return log10Mh_biased - log10Mh;

}

double int_for_PLxGivenMh(double logMh, void *p){

  const Model *model = ((params *)p)->model;
  double log10Lx     = ((params *)p)->log10Lx;
  double a           = ((params *)p)->a;
  double b           = ((params *)p)->b;
  double sigma       = ((params *)p)->sigma;

  double log10Mh    = logMh/log(10.0);
  double log10Lx0   = (log10Mh - 14.48) * a + b + 44.0;

  double z = 0.0;

  double res = normal(log10Lx, log10Lx0, sigma) * log10Mh * dndlnMh(model, z, log10Mh);

  return res;

}


double int_for_PLxGivenMh_norm(double logMh, void *p){


  const Model *model = ((params *)p)->model;
  double log10Lx     = ((params *)p)->log10Lx;
  double a           = ((params *)p)->a;
  double b           = ((params *)p)->b;
  double sigma       = ((params *)p)->sigma;

  double log10Mh    = logMh/log(10.0);
  double log10Lx0   = (log10Mh - 14.48) * a + b + 44.0;

  double z = 0.0;

  double res = normal(log10Lx, log10Lx0, sigma) * dndlnMh(model, z, log10Mh);

  return res;

}


double normal(double x, double mu, double sigma){

  return 1.0/(sigma*sqrt(2.0*M_PI)) * exp(-(x-mu)*(x-mu)/(2.0*sigma*sigma));

}
