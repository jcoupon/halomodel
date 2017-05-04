/*
 *    xray.c
 *    halomodel library
 *    Jean Coupon 2015-2017
 */

#include "xray.h"
#include <Python.h>

int main()
{
   /*
    *    THIS IS FOR TESTS ONLY
    *    Needs to define PYTHONHOME as:
    *    $ setenv PYTHONHOME ~/anaconda/
    */

   Py_Initialize();
   PyRun_SimpleString("import numpy as np; print \"Hello world\"");
   Py_Finalize();

   return 0;
}

void SigmaIx(const Model *model, double *theta, int N, double Mh, double c, double z, int obs_type, double *result)
{
   /*
    *    Computes projected  X-ray flux profiles
    *    See Vikhlinin et al. (2006) and Cavaliere & Fusco-Femiano (1978)
    *
    *    theta in [deg]
    *    SigmaIx in [CR s^-1 deg^-2]
    *    Mh in h^-1 Msun
    *
    *    If model.hod is set to 1, it will use the full
    *    HOD parametrisation, and needs a halo mass function. Otherwise
    *    if set to 0, it will only use the three parameters that controls
    *    the shape of a X-ray profile.
    */


   if(obs_type == all){
      SigmaIxAll(model, theta, N, Mh, c, z, result);
      return;
   }

   /*    interpolate to speed up integration for projection and PSF convolution */
   int i, j, k, Ninter = 128;

   /*    to convert from Mpc units to degree
    *    R[i] = theta[i] * degToMpcDegComo
    *    CR [... deg^-2 ] = CR [... Mpc^-1]*degToMpcDegComo^2
    */
   double degToMpcDegComo = DM(model, z, 0) * M_PI / 180.0;

   double *logrinter = (double *)malloc(Ninter*sizeof(double));
   double *rinter = (double *)malloc(Ninter*sizeof(double));
   double rinter_min = RMIN;
   double rinter_max = RMAX;
   double dlogrinter = log(rinter_max/rinter_min)/(double)Ninter;

   for(i=0;i<Ninter;i++){
      logrinter[i] = log(rinter_min)+dlogrinter*(double)i;
      rinter[i]    = exp(logrinter[i]);
   }

   double *Ix = (double *)malloc(Ninter*sizeof(double));
   double *SigmaIx_tmp = (double *)malloc(Ninter*sizeof(double));

   switch (obs_type){
      case cen:
         Ix1hc(model, rinter, Ninter, Mh, c, z, Ix);
         break;
      case sat:
         Ix1hs(model, rinter, Ninter, Mh, c, z, Ix);
         break;
      case XB:
         IxXB(model, rinter, Ninter, Mh, c, z, Ix);
         break;
      case twohalo:
         IxTwohalo(model, rinter, Ninter, Mh, c, z, Ix);
         break;
   }

   /*    interpolate Ix(r) */
   gsl_interp_accel *acc = gsl_interp_accel_alloc();
   gsl_spline *spline = gsl_spline_alloc (gsl_interp_cspline, Ninter);
   gsl_spline_init(spline, logrinter, Ix, Ninter);

   /*    project along line of sight */
   params p;
   switch (obs_type){
      /*    For central and satellites, computed the projected 3D profile */
      case cen: case sat: case twohalo:
         p.acc = acc;
         p.spline = spline;
         p.logrmin = logrinter[0];
         p.logrmax = logrinter[Ninter-1];
         for(i=0;i<Ninter;i++){
            p.R = rinter[i];
            // result[i] = 2.0*int_gsl(intForIx, (void*)&p, log(rinter_min), log(rinter_max), 1.e-4);
            SigmaIx_tmp[i] = 2.0*int_gsl(intForIx, (void*)&p, log(rinter[i]), log(RMAX), 1.e-4);
         }
         break;

      /*    For X-ray binaries, it is simply the de Vaucouleurs profile */
      case XB:
         for(i=0;i<Ninter;i++){
            double logr = log(rinter[i]);
            if(logrinter[0] < logr && logr < logrinter[Ninter-1]){
               SigmaIx_tmp[i] = gsl_spline_eval(spline, logr, acc);
            }else
            SigmaIx_tmp[i] = 0.0;
         }
         break;
   }

   /*    recycle interpolation space */
   gsl_spline_init(spline, logrinter, SigmaIx_tmp, Ninter);

   /*    convolve with PSF... */
   if (!isnan(model->XMM_PSF_rc_deg)){

      int N_PSF = 128;
      double Rpp, dlogr_PSF = log(rinter_max/rinter_min)/(double)N_PSF;
      double *phi = (double *)malloc(N_PSF*sizeof(double));
      double *Rp = (double *)malloc(N_PSF*sizeof(double));
      double *intForPhi = (double *)malloc(N_PSF*sizeof(double));
      double *intForRp = (double *)malloc(N_PSF*sizeof(double));

      double *PSF_profile = (double *)malloc(N_PSF*sizeof(double));

      for(i=0;i<N_PSF;i++){
         Rp[i] = exp(log(rinter_min)+dlogr_PSF*(double)i);
         //Rp[i] = rinter_min+exp(dlogr_PSF)*(double)i;
         phi[i] = 0.0+2.0*M_PI / (double)(N_PSF-1) * (double)i;
         intForPhi[i] = 0.0;
         intForRp[i] = 0.0;
         PSF_profile[i] = King(Rp[i], 1.0, model->XMM_PSF_rc_deg*degToMpcDegComo, model->XMM_PSF_alpha)*pow(degToMpcDegComo, 2.0);
      }

      // printf("%g %g\n", Rp[0]/degToMpcDegComo, Rp[N_PSF-1]/degToMpcDegComo );

      double norm_PSF = 1.0/trapz(Rp, PSF_profile, N_PSF);

      for(i=0;i<N;i++){                                        /*    Main loop */
         for(j=0;j<N_PSF;j++){                                 /*    loop over R' - trapeze integration */
            for(k=0;k<N_PSF;k++){                              /*    loop over phi - trapeze integration  */
               Rpp = sqrt(pow(theta[i]*degToMpcDegComo, 2.0) + Rp[j]*Rp[j] - 2.0*theta[i]*degToMpcDegComo*Rp[j]*cos(phi[k]));
               if(logrinter[0] < log(Rpp) && log(Rpp) < logrinter[Ninter-1]){
                  intForPhi[k] = gsl_spline_eval(spline, log(Rpp), acc);
               }
            }
            intForRp[j] = King(Rp[j], 1.0, model->XMM_PSF_rc_deg*degToMpcDegComo, model->XMM_PSF_alpha)*pow(degToMpcDegComo, 2.0)*trapz(phi, intForPhi, N_PSF);
         }
         result[i] = norm_PSF*1.0/(2.0* M_PI)*trapz(Rp, intForRp, N_PSF)*pow(degToMpcDegComo, 2.0);
      }

      free(intForPhi);
      free(intForRp);
      free(phi);
      free(Rp);

   }else{
      /*    ... or simply return result */
      for(i=0;i<N;i++){
         if(logrinter[0] < log(theta[i]*degToMpcDegComo) && log(theta[i]*degToMpcDegComo) < logrinter[Ninter-1]){
            result[i] = gsl_spline_eval(spline, log(theta[i]*degToMpcDegComo), acc)*pow(degToMpcDegComo, 2.0) ;
         }else{
            result[i] = 0.0;
         }
      }
   }

   free(SigmaIx_tmp);
   free(Ix);
   free(rinter);
   free(logrinter);
   gsl_spline_free (spline);
   gsl_interp_accel_free (acc);

   return;
}

double intForIx(double logz, void *p)
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

void SigmaIxAll(const Model *model, double *R, int N, double Mh, double c, double z, double *result)
{

   int i;

   double *result_tmp = (double *)malloc(N*sizeof(double));

   SigmaIx(model, R, N, Mh, c, z, cen, result_tmp);
   for(i=0;i<N;i++){
      result[i] += result_tmp[i];
   }

   SigmaIx(model, R, N, Mh, c, z, sat, result_tmp);
   for(i=0;i<N;i++){
      result[i] += result_tmp[i];
   }

   SigmaIx(model, R, N, Mh, c, z, XB, result_tmp);
   for(i=0;i<N;i++){
      result[i] += result_tmp[i];
   }

   SigmaIx(model, R, N, Mh, c, z, twohalo, result_tmp);
   for(i=0;i<N;i++){
      result[i] += result_tmp[i];
   }

   free(result_tmp);

   return;
}


void Ix1hc(const Model *model, double *r, int N, double Mh, double c, double z, double *result)
{
   /*
    *    Returns the 1-halo central X-ray
    *    3D luminosity profile.
    */

   int i;

   if(model->hod){

      /* Mh ignored here */

      params p;
      p.model = model;
      p.z = z;
      p.c = NAN;  /* for the HOD model, the concentration(Mh) relationship is fixed */

      double ng = ngal_den(model, LNMH_MAX, model->log10Mstar_min, model->log10Mstar_max, z, all);

      for(i=0;i<N;i++){
         p.r       = r[i];
         result[i] = int_gsl(intForIx1hc, (void*)&p, log(Mh_rh(model, r[i], z)) , LNMH_MAX, 1.e-3)/ng;
      }
   }else{
      for(i=0;i<N;i++){

         double TGas, ZGas;
         TGas = MhToTGas(model, Mh, z);
         ZGas = MhToZGas(model, Mh, z);

         double fac = CRToLx(model, z, TGas, ZGas);

         if (fac > 0.0){
            result[i] = ix(model, r[i], Mh, c, z)/fac;
         }else{
            result[i] = 0.0;
         }
      }
   }

   return;
}


double intForIx1hc(double logMh, void *p)
{
   /* Integrand for nGas if HOD model */

   const Model *model = ((params *)p)->model;
   double r = ((params *)p)->r;
   double c = ((params *)p)->c;
   double z = ((params *)p)->z;

   double Mh = exp(logMh);

   double TGas, ZGas;
   TGas = MhToTGas(model, Mh, z);
   ZGas = MhToZGas(model, Mh, z);

   double fac = CRToLx(model, z, TGas, ZGas);

   if (fac > 0.0){
      return Ngal_c(model, Mh, model->log10Mstar_min, model->log10Mstar_max)
         * ix(model, r, Mh, c, z)
         * dndlnMh(model, Mh, z)/fac;
   }else{
      return 0.0;
   }

}

void Ix1hs(const Model *model, double *r, int N, double Mh, double c, double z, double *result)
{

   /*
    *    Returns the (backward) fourier transform of a
    *    x-ray luminosity power spectrum (for satellites).
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
   p.Mh = Mh;
   p.c = c;
   p.z = z;

   /*    fonction with parameters to fourier transform */
   gsl_function Pk;
   Pk.function = &intForIx1hs;
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

   const Model *model = ((params *)p)->model;
   const double Mh = ((params *)p)->Mh;
   const double c = ((params *)p)->c;
   const double z = ((params *)p)->z;

   return pow(k, 1.5 )* PIx1hs(model, k, Mh, c, z);

}

#define SCALE (1.e44)

double PIx1hs(const Model *model, double k, const double Mh, const double c, const double z)
{
   if(model->hod){

      /* Mh ignored here */

      params p;
      p.model = model;
      p.k = k;
      p.z = z;
      p.c = NAN;  /*    for the HOD model, the concentration(Mh) relationship is fixed */

      double ng = ngal_den(model, LNMH_MAX, model->log10Mstar_min, model->log10Mstar_max, z, all);
      return int_gsl(intForPIx1hs, (void*)&p, LNMH_MIN, LNMH_MAX, 1.e-3)/ng;

   }else{

      double TGas, ZGas;
      TGas = MhToTGas(model, Mh, z);
      ZGas = MhToZGas(model, Mh, z);

      double fac = CRToLx(model, z, TGas, ZGas);
      double Norm = NormIx(model, Mh, c, z);

      if (fac*Norm > 0.0){
         return pow(uIx(model, k, Mh, c, z), 2.0) * Norm / fac * SCALE;
      }else{
         return 0.0;
      }
   }
}

double intForPIx1hs(double logMh, void *p)
{
   /*    Integrand for nGas if HOD model */

   const Model *model = ((params *)p)->model;
   const double k = ((params *)p)->k;
   const double c = ((params *)p)->c;
   const double z = ((params *)p)->z;

   double Mh = exp(logMh);

   double TGas, ZGas;
   TGas = MhToTGas(model, Mh, z);
   ZGas = MhToZGas(model, Mh, z);

   double fac = CRToLx(model, z, TGas, ZGas);
   double Norm = NormIx(model, Mh, c, z);

   if (fac*Norm > 0.0){
      return  Ngal_s(model, Mh, model->log10Mstar_min, model->log10Mstar_max)
         * pow(uIx(model, k, Mh, c, z), 2.0) * Norm
         * dndlnMh(model, Mh, z) / fac * SCALE;

   }else{
      return 0.0;
   }

}

double uIx(const Model *model, double k, double Mh, double c, double z)
{
   /*
    *    Returns the normalised (out to rh radius)
    *    fourier transform of the brightness profile.
    *    The constants are set by the wrapper and depend on
    *    halo properties, redshift, etc., so that all the
    *    quantities depending on cosmology are managed by
    *    the wrapper.
    *
    *    c is only used to compute the truncation radius
    */


#if 0
   // DEBUGGING
   params p;
   p.model = model;
   p.c = c;
   p.z = z;

   p.Mh = Mh;
   double Norm2 = NormIx(model, p.Mh, c, z);
   p.k = k;
   return 4.0*M_PI/Norm2*int_gsl(intForUIx, (void*)&p, log(1.e-6), log(2.0*M_PI/KMIN), 1.e-3);

#endif


   // TODO  no interpolation if c != nan ?

   static int firstcall = 1;

   static gsl_spline2d *spline = NULL;
   static gsl_interp_accel *xacc = NULL;
   static gsl_interp_accel *yacc = NULL;

   static double Norm = 1.0, result, inter_xmin, inter_xmax, inter_ymin, inter_ymax;
   static double *x = NULL, *logx = NULL, dlogx = 0.0;
   static double *y = NULL, *logy = NULL, dlogy = 0.0;
   static double *za;

   // static int i, j, Nx = 64, Ny = 64;
   static int i, j, Nx = 256, Ny = 256;

   static double c_tmp = NAN;
   static double z_tmp = NAN;
   static Model model_tmp;

   if(firstcall){

      copyModelXRay(model, &model_tmp);

      /*    initialize interpolation */
      const gsl_interp2d_type *T = gsl_interp2d_bilinear;
      spline = gsl_spline2d_alloc(T, Nx, Ny);
      xacc = gsl_interp_accel_alloc();
      yacc = gsl_interp_accel_alloc();

      /*    x axis = logMh */
      x = (double *)malloc(Nx*sizeof(double));
      logx = (double *)malloc(Nx*sizeof(double));
      dlogx = (LNMH_MAX - LNMH_MIN)/(double)Nx;
      for(i=0;i<Nx;i++){
         logx[i] = LNMH_MIN + dlogx*(double)i;
         x[i] = exp(logx[i]);
      }

      /*    y axis = logk */
      y = (double *)malloc(Ny*sizeof(double));
      logy = (double *)malloc(Ny*sizeof(double));
      dlogy = log(KMAX/KMIN)/(double)Ny;
      for(j=0;j<Ny;j++){
         logy[j] = log(KMIN) + dlogy*(double)j;
         y[j] = exp(logy[j]);
      }

      /*    z axis = log(uHalo) */
      za = (double *)malloc(Nx*Ny*sizeof(double));

      /*    interpolation range */
      inter_xmin = x[0];
      inter_xmax = x[Nx-1];

      inter_ymin = y[0];
      inter_ymax = y[Ny-1];

   }

   if( firstcall || changeModelXRay(model, &model_tmp) || assert_float(c_tmp, c) || assert_float(z_tmp, z)){
      /*    If first call or model parameters have changed, the table must be recomputed */

      firstcall = 0;
      c_tmp = c;
      z_tmp = z;
      copyModelXRay(model, &model_tmp);

      params p;
      p.model = model;
      p.c = c;
      p.z = z;

      for(i=0;i<Nx;i++){    /*      loop over halo mass */
         p.Mh = x[i];
         Norm = NormIx(model, p.Mh, c, z);
         for(j=0;j<Ny;j++){ /*      loop over k */
            p.k = y[j];
            result = 4.0*M_PI/Norm*int_gsl(intForUIx, (void*)&p, log(1.e-6), log(2.0*M_PI/KMIN), 1.e-3);
            if (Norm > 0.0){
               gsl_spline2d_set(spline, za, i, j, result);
            }else{
               gsl_spline2d_set(spline, za, i, j, 0.0);
            }
         }

      }
      /*    fill in interpolation space */
      gsl_spline2d_init(spline, logx, logy, za, Nx, Ny);

   }

   if (inter_xmin < Mh && Mh < inter_xmax && inter_ymin < k && k < inter_ymax){
      return gsl_spline2d_eval(spline, log(Mh), log(k), xacc, yacc);
   }else{
      return 0.0;
   }
}

double intForUIx(double logr, void *p)
{
   /*
    *    Integrand for uIx().
    */

   const Model *model = ((params *)p)->model;
   const double Mh = ((params *)p)->Mh;
   const double c = ((params *)p)->c;
   const double z = ((params *)p)->z;
   const double k = ((params *)p)->k;

   double r = exp(logr);
   return ix(model, r, Mh, c, z) * sinc(k*r) * r * r * r / SCALE;
}

double NormIx(const Model *model, double Mh, double c, double z)
{
   /*    Returns the integrated surface brightness
    *    c is only used to compute the truncation radius
    */

   static int firstcall = 1;

   int i, Ninter = 64;

   static gsl_interp_accel *acc;
   static gsl_spline *spline;
   static double inter_min, inter_max;

   static double dx, *x, *y;

   static double c_tmp;
   static double z_tmp;

   static Model model_tmp;

   if(firstcall){

      copyModelXRay(model, &model_tmp);

      dx = (LNMH_MAX-LNMH_MIN)/(double)Ninter;
      x = (double *)malloc(Ninter*sizeof(double));
      y = (double *)malloc(Ninter*sizeof(double));

      for(i=0;i<Ninter;i++){
         x[i]  = LNMH_MIN+dx*(double)i;
      }

      acc = gsl_interp_accel_alloc();
      spline = gsl_spline_alloc (gsl_interp_cspline, Ninter);

   }

   if(firstcall || changeModelXRay(model, &model_tmp) || assert_float(c_tmp, c) || assert_float(z_tmp, z)){
      /*    if first call or model parameters have changed, the table must be recomputed */

      firstcall = 0;
      c_tmp = c;
      z_tmp = z;
      copyModelXRay(model, &model_tmp);

      params p;
      p.model = model;
      p.c = c;
      p.z = z;

      for(i=0;i<Ninter;i++){
         p.Mh = exp(x[i]);
         // y[i] = int_gsl_QNG(intForNormIx, (void*)&p, log(RMIN), log(r_vir(model, p.Mh, c, z)), 1.0e-3);
         y[i] = int_gsl_QNG(intForNormIx, (void*)&p, log(RMIN), log(rh(model, p.Mh, c, z)), 1.0e-3);
      }

      inter_min = exp(x[0]);
      inter_max = exp(x[Ninter-1]);

      gsl_spline_init(spline, x, y, Ninter);

   }

   if (Mh < inter_min || Mh > inter_max){
      return 0.0;
   }else{
      return gsl_spline_eval(spline, log(Mh), acc);
   }
}


double intForNormIx(double logr, void *p){

   const Model *model =  ((params *)p)->model;
   const double r = exp(logr);
   const double Mh = ((params *)p)->Mh;
   const double c = ((params *)p)->c;
   const double z = ((params *)p)->z;

   return ix(model, r, Mh, c, z) * r / SCALE;
}


void IxXB(const Model *model, double *r, int N, double Mh, double c, double z, double *result)
{
   /*
    *    Returns the 2D gas brightness
    *    in CR Mpc-2 assuming
    *    a de Vaucouleur profile for binary stars
    */

   int i;

   if (model->IxXB_Re < 0.0 || model->IxXB_CR < 0.0){
      for(i=0;i<N;i++){
         result[i] = 0.0;
      }
      return;
   }

   /*    Re_XB in Mpc */
   double Re = model->IxXB_Re;

   /*    total X-ray luminosity of binary stars in CR */
   double CR = model->IxXB_CR;

   double Ie = CR / (7.215 * M_PI * Re * Re);

   for(i=0;i<N;i++){
      /*    De vaucouleur profile */
      result[i] = Ie * exp(-7.669*(pow(r[i]/Re, 1.0/4.0) - 1.0));
   }

   return;
}

void IxTwohalo(const Model *model, double *r, int N, double Mh, double c, double z, double *result){
   /*
    *    Returns the 2-halo term of x-ray
    *    surface brightness.
    */

   int i;
   double bias_fac;

   /*    FFTLog config */
   double q = 0.0, mu = 0.5;
   int FFT_N = 64;
   FFTLog_config *fc = FFTLog_init(FFT_N, KMIN, KMAX, q, mu);

   /*    parameters to pass to the function */
   params p;
   p.model = model;
   p.c = c;
   p.z = z;
   p.Mh = Mh;

   /*    fonction with parameters to fourier transform */
   gsl_function Pk;
   Pk.function = &intForIxTwohalo;
   Pk.params = &p;

   double *xidm = malloc(N*sizeof(double));
   xi_m(model, r, N, z, xidm);

   p.ng  = ngal_den(model, LNMH_MAX, model->log10Mstar_min, model->log10Mstar_max, z, all);
   for(i=0;i<N;i++){

      bias_fac = sqrt(pow(1.0+1.17*xidm[i],1.49)/pow(1.0+0.69*xidm[i],2.09));
      p.logMlim = logM_lim(model, r[i], p.c, z, all);
      p.r = r[i];
      if(model->hod){
         p.ngp = ngal_den(model, p.logMlim, model->log10Mstar_min, model->log10Mstar_max, z, all);
      }else{
         p.ngp = p.ng;
      }

      if(p.ng < 1.0e-14 || p.ngp < 1.0e-14 || r[i] < RMIN2){
         result[i] = 0.0;
      }else{
         result[i] = (p.ngp/p.ng)*pow(bias_fac, 2.0)*xi_from_Pkr(&Pk, r[i], fc);
      }
   }
   FFTLog_free(fc);

   free(xidm);

   return;
}


double intForIxTwohalo(double k, void *p){
   return pow(k, 1.5 )* P_Ix_twohalo(k, p);
}

double P_Ix_twohalo(double k, void *p)
{

   const Model *model =  ((params *)p)->model;
   const double z = ((params *)p)->z;
   const double ngp = ((params *)p)->ngp;
   const double logMlim = ((params *)p)->logMlim;

   ((params *)p)->k = k;

   if(model->hod){

      ((params *)p)->c = NAN;  /*    for the HOD model, the concentration(Mh) relationship is fixed */

      return P_m_nonlin(model, k, z)*pow(int_gsl(intForP_twohalo_Ix, p, LNMH_MIN, logMlim, 1.e-3), 2.0)/ngp;

   }else{

      const double Mh = ((params *)p)->Mh;
      const double z = ((params *)p)->z;
      const double c = ((params *)p)->c;

      double TGas, ZGas;
      TGas = MhToTGas(model, Mh, z);
      ZGas = MhToZGas(model, Mh, z);

      double fac = CRToLx(model, z, TGas, ZGas);
      double Norm = NormIx(model, Mh, c, z);

      // DEBUGGING
      //printf("%e %e %e\n", Mh , c, z);

      if (fac*Norm > 0.0){
         return  uIx(model, k, Mh, c, z) * Norm * bias_h(model, Mh, z) / fac * SCALE;
      }else{
         return 0.0;
      }
   }
}

double intForP_twohalo_Ix(double logMh, void *p){

   const Model *model = ((params *)p)->model;
   const double k = ((params *)p)->k;
   const double z = ((params *)p)->z;
   const double c = ((params *)p)->c;

   double Mh = exp(logMh);

   double TGas, ZGas;
   TGas = MhToTGas(model, Mh, z);
   ZGas = MhToZGas(model, Mh, z);

   double fac = CRToLx(model, z, TGas, ZGas);
   double Norm = NormIx(model, Mh, c, z);

   if (fac*Norm > 0.0){
      return  uIx(model, k, Mh, c, z) * Norm
         * bias_h(model, Mh, z) * dndlnMh(model, Mh, z) / fac * SCALE;
   }else{
      return 0.0;
   }

}

#undef SCALE


double ix(const Model *model, double r, double Mh, double c, double z){
   /*
    *    Returns the 3D gas brightness
    *    in erg s^-1 h^3 Mpc-3 / 1e.44 assuming
    *    a 3D gas profile
    *    ix \propto nGas^2
    *    = Ix if non HOD
    *    = ix if HOD (to be integrated over halo mass function)
    */

   double TGas, ZGas;
   TGas = MhToTGas(model, Mh, z);
   ZGas = MhToZGas(model, Mh, z);

   return LambdaBolo(TGas, ZGas)*pow(1.21*nGas(model, r, Mh, c, z), 2.0) / cm3toMpc3_como(model, z);

}


double MhToTGas(const Model *model, double Mh, double z)
{
   // double log10Mh = log10(Mh);
   // return pow(10.0, (log10Mh - 13.56) / 1.69);

   // return pow(10.0, (log10Mh - 13.56 - 0.24) / 1.69);

   int i, Ninter;

   static gsl_interp_accel *acc;
   static gsl_spline *spline;
   static double inter_min, inter_max;

   Ninter = model->gas_TGasMh_N;
   inter_min = model->gas_TGasMh_log10Mh[0];
   inter_max = model->gas_TGasMh_log10Mh[Ninter-1];

   acc = gsl_interp_accel_alloc();
   spline = gsl_spline_alloc (gsl_interp_cspline, Ninter);

   gsl_spline_init(spline, model->gas_TGasMh_log10Mh, model->gas_TGasMh_log10TGas, Ninter);

   if (log10(Mh) < inter_min){
      return model->gas_TGasMh_log10TGas[0];
   }else if (log10(Mh) > inter_max){
      return model->gas_TGasMh_log10TGas[Ninter-1];
   }else{
      return gsl_spline_eval(spline, log10(Mh), acc);
   }
}

/*
double TGasToMh(const Model *model, double TGas, double z)
{

   double log10TGas = log10(TGas);
   return pow(10.0, 1.69*log10TGas + 13.56);
}
*/

double MhToZGas(const Model *model, double Mh, double z)
{
   /*
    *    Metallicity
    *
    */
   //return 0.25;

   int i, Ninter;

   static gsl_interp_accel *acc;
   static gsl_spline *spline;
   static double inter_min, inter_max;

   Ninter = model->gas_ZGasMh_N;
   inter_min = model->gas_ZGasMh_log10Mh[0];
   inter_max = model->gas_ZGasMh_log10Mh[Ninter-1];

   acc = gsl_interp_accel_alloc();
   spline = gsl_spline_alloc (gsl_interp_cspline, Ninter);

   gsl_spline_init(spline, model->gas_ZGasMh_log10Mh, model->gas_ZGasMh_ZGas, Ninter);

   if (log10(Mh) < inter_min){
      return model->gas_ZGasMh_ZGas[0];
   }else if (log10(Mh) > inter_max){
      return model->gas_ZGasMh_ZGas[Ninter-1];
   }else{
      return gsl_spline_eval(spline, log10(Mh), acc);
   }

}

# define CONST_MGAS (3.33054952367e+16) /* = mu * mp * 2.21 / (cm3ToMpc3 * Msun) */

double const_Mgas_como(const Model *model, double z)
{
   /*    constant for Mgas. Returns
    *    result in h^-3 comoving Mpc. */

   static double result = 0.0, z_tmp = -1, H0_tmp = -1;

   if(assert_float(z_tmp, z) || assert_float(H0_tmp, model->H0)){
      z_tmp = z;
      H0_tmp = model->H0;
      result = CONST_MGAS / (pow(model->H0/100.0, 3.0) * pow(1.0+z, 3.0));
   }

   return result;
}

# undef CONST_MGAS


double MGas(const Model *model, double r, double Mh, double c, double z){
/*
 *    Returns the total Mgas(<r) out to r
 *    in h^-1 Msun
 */

   params p;
   p.model = model;
   p.Mh = Mh;
   p.c = c;
   p.z = z;

   return const_Mgas_como(model, z)*int_gsl(intForMGas, (void*)&p, log(RMIN), log(r), 1.e-4);
}

double intForMGas(double logr, void *p){

   const Model *model = ((params *)p)->model;
   double r = exp(logr);
   double Mh = ((params *)p)->Mh;
   double c = ((params *)p)->c;
   double z = ((params *)p)->z;

   return r * r * 4.0 * M_PI * nGas(model, r, Mh, c, z) * r;
}


# define EPS 1.e-8
double LambdaBolo(double TGas, double ZGas)
{
   /*
    *    Cooling function in erg s^-1 cm^3 for the bolometric luminosity
    *
    *    TGas: gas temperature
    *    Zgas: gas metallicity
    */

   static int firstcall = 1;

   static gsl_spline2d *spline   = NULL;
   static gsl_interp_accel *xacc = NULL;
   static gsl_interp_accel *yacc = NULL;

   static double inter_xmin, inter_xmax, inter_ymin, inter_ymax;

   if(firstcall){

      firstcall = 0;

      /* tabulated values from D. Eckert */
      // awk '!(/^#/) {printf("%f, ", log($3))}' info/cooling_function_Eckert.ascii
      double logLambda_t[328] = {-51.071595, -50.251454, -49.934409, -49.592541, -49.266065,
         -49.020393, -48.823258, -48.658719, -52.450425, -50.408139, -49.950616, -49.511885,
         -49.124261, -48.845570, -48.627860, -48.449176, -53.165734, -51.616334, -51.194254,
         -50.777915, -50.403549, -50.131753, -49.918288, -49.742475, -53.316281, -51.816177,
         -51.398802, -50.985547, -50.612999, -50.342229, -50.129343, -49.953938, -53.565854,
         -51.930066, -51.500365, -51.079047, -50.701704, -50.428383, -50.213957, -50.037490,
         -53.618955, -52.005416, -51.577620, -51.157551, -50.780944, -50.508012, -50.293786,
         -50.117488, -53.615012, -52.118231, -51.701215, -51.288144, -50.915746, -50.645023,
         -50.432186, -50.256801, -53.595092, -52.409297, -52.028794, -51.640339, -51.283204,
         -51.020598, -50.812807, -50.640900, -53.563609, -52.647534, -52.311119, -51.954471,
         -51.618019, -51.366708, -51.166054, -50.999010, -53.523701, -52.779505, -52.479478,
         -52.151010, -51.833791, -51.593343, -51.399659, -51.237457, -53.481720, -52.849043,
         -52.577070, -52.271659, -51.970879, -51.739958, -51.552495, -51.394709, -53.438127,
         -52.834676, -52.570745, -52.272137, -51.976334, -51.748362, -51.562824, -51.406363,
         -53.361067, -52.791065, -52.536696, -52.246365, -51.956684, -51.732340, -51.549222,
         -51.394519, -53.296558, -52.747480, -52.499314, -52.214410, -51.928803, -51.706908,
         -51.525421, -51.371893, -53.239919, -52.711613, -52.469743, -52.190424, -51.909055,
         -51.689719, -51.509944, -51.357606, -53.188084, -52.688004, -52.455022, -52.183648,
         -51.908387, -51.692772, -51.515500, -51.364954, -53.142889, -52.679041, -52.457961,
         -52.197521, -51.930749, -51.720364, -51.546652, -51.398705, -53.105937, -52.693170,
         -52.489841, -52.246145, -51.992778, -51.790793, -51.622828, -51.479060, -53.069896,
         -52.707260, -52.522629, -52.297182, -52.058841, -51.866528, -51.705301, -51.566483,
         -53.033005, -52.720259, -52.555560, -52.350322, -52.129278, -51.948373, -51.795219,
         -51.662437, -52.997474, -52.733430, -52.589571, -52.406475, -52.205074, -52.037498,
         -51.894029, -51.768585, -52.964439, -52.740772, -52.615378, -52.452585, -52.270017,
         -52.115690, -51.982026, -51.864140, -52.934852, -52.736698, -52.623492, -52.474626,
         -52.305389, -52.160667, -52.034270, -51.922072, -52.906115, -52.732600, -52.631701,
         -52.497164, -52.342036, -52.207768, -52.089395, -51.983581, -52.853450, -52.717798,
         -52.636591, -52.525949, -52.395229, -52.279633, -52.176026, -52.082157, -52.806925,
         -52.693453, -52.624336, -52.528896, -52.414369, -52.311595, -52.218425, -52.133183,
         -52.763290, -52.668461, -52.609857, -52.527954, -52.428258, -52.337586, -52.254480,
         -52.177731, -52.706592, -52.623943, -52.572434, -52.499801, -52.410504, -52.328531,
         -52.252777, -52.182332, -52.654306, -52.581637, -52.535982, -52.471166, -52.390791,
         -52.316430, -52.247189, -52.182438, -52.623585, -52.554952, -52.511654, -52.450095,
         -52.373455, -52.302279, -52.235849, -52.173539, -52.593752, -52.528929, -52.487942,
         -52.429435, -52.356415, -52.288346, -52.224636, -52.164724, -52.552014, -52.492057,
         -52.453978, -52.399447, -52.331099, -52.267100, -52.206976, -52.150244, -52.493399,
         -52.438214, -52.403029, -52.352500, -52.288857, -52.229000, -52.172546, -52.119111,
         -52.439135, -52.387738, -52.354901, -52.307564, -52.247748, -52.191286, -52.137864,
         -52.087156, -52.389506, -52.340737, -52.309505, -52.264422, -52.207277, -52.153240,
         -52.101974, -52.053209, -52.342916, -52.296527, -52.266740, -52.223677, -52.168979,
         -52.117136, -52.067849, -52.020877, -52.302984, -52.258550, -52.229988, -52.188618,
         -52.135992, -52.085995, -52.038382, -51.992952, -52.264561, -52.221958, -52.194538,
         -52.154742, -52.104055, -52.055818, -52.009782, -51.965787, -52.228802, -52.187818,
         -52.161405, -52.123041, -52.074083, -52.027415, -51.982828, -51.940126, -52.196887,
         -52.157285, -52.131735, -52.094586, -52.047098, -52.001767, -51.958399, -51.916855,
         -52.165953, -52.127653, -52.102899, -52.066898, -52.020822, -51.976776, -51.934570,
         -51.894091};

      // awk '!(/^#/) {print $1}' info/cooling_function_Eckert.ascii | sort -n | uniq | awk '{printf("%f, ", log($1))}
      double logTGas_t[41] = {-4.605170, -3.912023, -2.995732, -2.659260, -2.302585, -2.120264, -1.897120,
         -1.609438, -1.386294, -1.203973, -1.049822, -0.916291, -0.693147, -0.510826, -0.356675, -0.223144,
         -0.105361, 0.000000, 0.095310, 0.182322, 0.262364, 0.336472, 0.405465, 0.470004, 0.587787, 0.693147,
         0.788457, 0.916291, 1.029619, 1.098612, 1.163151, 1.252763, 1.386294, 1.504077, 1.609438, 1.704748,
         1.791759, 1.871802, 1.945910, 2.014903, 2.079442};

      // awk '!(/^#/) {print $2}' info/cooling_function_Eckert.ascii | sort -n | uniq | awk '{printf("%f, ", $1)}'
      double ZGas_t[8] = {0.00,0.15,0.25,0.40,0.60,0.80,1.00,1.20};

      int Ny = sizeof(logTGas_t)/sizeof(logTGas_t[0]);
      int Nx = sizeof(ZGas_t)/sizeof(ZGas_t[0]);

      ZGas_t[0] = ZGas_t[0];
      ZGas_t[Nx-1] = ZGas_t[Nx-1];

      /* initialize interpolation */
      const gsl_interp2d_type *T = gsl_interp2d_bilinear;
      spline = gsl_spline2d_alloc(T, Nx, Ny);
      xacc   = gsl_interp_accel_alloc();
      yacc   = gsl_interp_accel_alloc();

      inter_ymin = exp(logTGas_t[0]);
      inter_ymax = exp(logTGas_t[Ny-1]);

      /* EPS is to allow to get values at Z=0.0 and Z=0.40*/
      inter_xmin = ZGas_t[0]-EPS;
      inter_xmax = ZGas_t[Nx-1]+EPS;

      /* set interpolation */
      gsl_spline2d_init(spline, ZGas_t, logTGas_t, logLambda_t, Nx, Ny);

   }

   if (inter_ymin < TGas && TGas < inter_ymax && inter_xmin < ZGas && ZGas < inter_xmax){
      return exp(gsl_spline2d_eval(spline, ZGas, log(TGas), xacc, yacc));
   }else{
      return 0.0;
   }

   /* Ettori (2015) */
   // return 1.1995 * 0.85e-23 * pow(TGas, 0.5);
}


double Lambda0p5_2p0(double TGas, double ZGas)
{
   /*
    *    Cooling function in erg s^-1 cm^3 in the soft band
    *
    *    TGas: gas temperature
    *    Zgas: gas metallicity
    */

   static int firstcall = 1;

   static gsl_spline2d *spline   = NULL;
   static gsl_interp_accel *xacc = NULL;
   static gsl_interp_accel *yacc = NULL;

   static double inter_xmin, inter_xmax, inter_ymin, inter_ymax;

   if(firstcall){

      firstcall = 0;

      /* tabulated values from D. Eckert */
      // awk '!(/^#/) {printf("%f, ", log($5))}' info/cooling_function_Eckert.ascii
      double logLambda_t[328] = {-96.173538, -95.838914, -95.665298, -95.450937, -95.222016,
         -95.035810, -94.878958, -94.743407, -75.874814, -75.202498, -74.920101, -74.605983,
         -74.298911, -74.064313, -73.874426, -73.714954, -63.427108, -61.203205, -60.736615,
         -60.292371, -59.901509, -59.621243, -59.402548, -59.223203, -61.013501, -58.603487,
         -58.129270, -57.680456, -57.286927, -57.005292, -56.785793, -56.605890, -58.990062,
         -56.462877, -55.984517, -55.533289, -55.138383, -54.856024, -54.636082, -54.455906,
         -58.191893, -55.704510, -55.227515, -54.777045, -54.382590, -54.100474, -53.880672,
         -53.700623, -57.376324, -55.039067, -54.567646, -54.120501, -53.727962, -53.446814,
         -53.227611, -53.047926, -56.511305, -54.503782, -54.048178, -53.610647, -53.223640,
         -52.945367, -52.727901, -52.549363, -55.953756, -54.247377, -53.811931, -53.386981,
         -53.007430, -52.732959, -52.517859, -52.340910, -55.560152, -54.074614, -53.658727,
         -53.246372, -52.874418, -52.603918, -52.391233, -52.215940, -55.265747, -53.930220,
         -53.530590, -53.128965, -52.763660, -52.496648, -52.286145, -52.112336, -55.034715,
         -53.788749, -53.400160, -53.006096, -52.645472, -52.380992, -52.172009, -51.999261,
         -54.710027, -53.576824, -53.203906, -52.820739, -52.466965, -52.206184, -51.999540,
         -51.828352, -54.493157, -53.452523, -53.093968, -52.721118, -52.373962, -52.116783,
         -51.912398, -51.742766, -54.332184, -53.380180, -53.037070, -52.675414, -52.335639,
         -52.082494, -51.880663, -51.712819, -54.211973, -53.356463, -53.032028, -52.684442,
         -52.354109, -52.106200, -51.907706, -51.742159, -54.115424, -53.357477, -53.054246,
         -52.723232, -52.404257, -52.162775, -51.968430, -51.805767, -54.043457, -53.398750,
         -53.123493, -52.815383, -52.512645, -52.280576, -52.092331, -51.933986, -53.978318,
         -53.440182, -53.195302, -52.913289, -52.629873, -52.409297, -52.228696, -52.075772,
         -53.927731, -53.474246, -53.256650, -52.999457, -52.735282, -52.526507, -52.353863,
         -52.206698, -53.879578, -53.509590, -53.322073, -53.093813, -52.853126, -52.659308,
         -52.496994, -52.357384, -53.838965, -53.536558, -53.376176, -53.175436, -52.958298,
         -52.779998, -52.628766, -52.497397, -53.809815, -53.545657, -53.401713, -53.218567,
         -53.017100, -52.849491, -52.705986, -52.580521, -53.781479, -53.554839, -53.427991,
         -53.263643, -53.079578, -52.924187, -52.789681, -52.671185, -53.735434, -53.565854,
         -53.466902, -53.334749, -53.181940, -53.049465, -52.932465, -52.827762, -53.703201,
         -53.565687, -53.483483, -53.371640, -53.239618, -53.123011, -53.018593, -52.924100,
         -53.674384, -53.564606, -53.497539, -53.404685, -53.293002, -53.192489, -53.101215,
         -53.017533, -53.648545, -53.556240, -53.499096, -53.419035, -53.321485, -53.232551,
         -53.150885, -53.075388, -53.626447, -53.549090, -53.500733, -53.432209, -53.347654,
         -53.269693, -53.197374, -53.129880, -53.618077, -53.546800, -53.501983, -53.438274,
         -53.359236, -53.285990, -53.217685, -53.153804, -53.609777, -53.544514, -53.503233,
         -53.444378, -53.370955, -53.302556, -53.238477, -53.178315, -53.599560, -53.542235,
         -53.505739, -53.453416, -53.387677, -53.325930, -53.267776, -53.212876, -53.594664,
         -53.545085, -53.513375, -53.467656, -53.409730, -53.354910, -53.303003, -53.253659,
         -53.591240, -53.548190, -53.520433, -53.480190, -53.428936, -53.380180, -53.333691,
         -53.289268, -53.590642, -53.551551, -53.526259, -53.489561, -53.442535, -53.397692,
         -53.354775, -53.313624, -53.590642, -53.555251, -53.532442, -53.499096, -53.456248,
         -53.415231, -53.375832, -53.337859, -53.594577, -53.561782, -53.540448, -53.509353,
         -53.469318, -53.430825, -53.393759, -53.358017, -53.598527, -53.568272, -53.548518,
         -53.519716, -53.482487, -53.446666, -53.412014, -53.378521, -53.603098, -53.574973,
         -53.556653, -53.529707, -53.494975, -53.461336, -53.428863, -53.397341, -53.609168,
         -53.582228, -53.564689, -53.538906, -53.505504, -53.473259, -53.441946, -53.411657,
         -53.615187, -53.589534, -53.572707, -53.548107, -53.516224, -53.485248, -53.455278,
         -53.426106};

      // awk '!(/^#/) {print $1}' info/cooling_function_Eckert.ascii | sort -n | uniq | awk '{printf("%f, ", log($1))}
      double logTGas_t[41] = {-4.605170, -3.912023, -2.995732, -2.659260, -2.302585, -2.120264, -1.897120,
         -1.609438, -1.386294, -1.203973, -1.049822, -0.916291, -0.693147, -0.510826, -0.356675, -0.223144,
         -0.105361, 0.000000, 0.095310, 0.182322, 0.262364, 0.336472, 0.405465, 0.470004, 0.587787, 0.693147,
         0.788457, 0.916291, 1.029619, 1.098612, 1.163151, 1.252763, 1.386294, 1.504077, 1.609438, 1.704748,
         1.791759, 1.871802, 1.945910, 2.014903, 2.079442};

      // awk '!(/^#/) {print $2}' info/cooling_function_Eckert.ascii | sort -n | uniq | awk '{printf("%f, ", $1)}'
      double ZGas_t[8] = {0.00,0.15,0.25,0.40,0.60,0.80,1.00,1.20};

      int Ny = sizeof(logTGas_t)/sizeof(logTGas_t[0]);
      int Nx = sizeof(ZGas_t)/sizeof(ZGas_t[0]);

      ZGas_t[0] = ZGas_t[0];
      ZGas_t[Nx-1] = ZGas_t[Nx-1];

      /* initialize interpolation */
      const gsl_interp2d_type *T = gsl_interp2d_bilinear;
      spline = gsl_spline2d_alloc(T, Nx, Ny);
      xacc   = gsl_interp_accel_alloc();
      yacc   = gsl_interp_accel_alloc();

      inter_ymin = exp(logTGas_t[0]);
      inter_ymax = exp(logTGas_t[Ny-1]);

      /* EPS is to allow to get values at Z=0.0 and Z=0.40*/
      inter_xmin = ZGas_t[0]-EPS;
      inter_xmax = ZGas_t[Nx-1]+EPS;

      /* set interpolation */
      gsl_spline2d_init(spline, ZGas_t, logTGas_t, logLambda_t, Nx, Ny);

   }

   if (inter_ymin < TGas && TGas < inter_ymax && inter_xmin < ZGas && ZGas < inter_xmax){
      return exp(gsl_spline2d_eval(spline, ZGas, log(TGas), xacc, yacc));
   }else{
      return 0.0;
   }

   /* Ettori (2015) */
   // return 1.1995 * 0.85e-23 * pow(TGas, 0.5);
}










# undef EPS

double CRToLx(const Model *model, double z, double TGas, double ZGas)
{
   /*    call python function to ease the 3D interpolation
    *    see http://www.linuxjournal.com/article/8497?page=0,0
    *    to return arrays, see https://github.com/Frogee/PythonCAPI_testing/blob/master/testEmbed.cpp
    *
    *    TODO: directly call CR->Lx code
    */


   static int firstcall = 1;
   static gsl_interp_accel *acc;
   static gsl_spline *spline;
   static double inter_min, inter_max;

   static double z_tmp = NAN;
   static double ZGas_tmp = NAN;

   if(firstcall || assert_float(z_tmp, z) || assert_float(ZGas_tmp, ZGas) ){

      firstcall = 0;
      ZGas_tmp = ZGas;
      z_tmp = z;

      /*    the code below is necessary as xray.c is itself
       *    wrapped into python code
       *    see https://docs.python.org/2/c-api/init.html for details
       */
      PyGILState_STATE gstate;
      gstate = PyGILState_Ensure();

      /*    add python directory to python path to import xray module */
      char cmd[1000] = "";
      PyRun_SimpleString("import os; import sys");
      sprintf(cmd, "dirname = os.path.dirname(\"%s\")", __FILE__);
      PyRun_SimpleString(cmd);
      PyRun_SimpleString("sys.path.insert(0, dirname+'/../python'); ");
      /*    import xray module and function CRToLx function */
   	PyObject *pName = PyString_FromString("xray");  //Get the name of the module
   	PyObject *pModule = PyImport_Import(pName);     //Get the module
      if (pModule == NULL)
      {
         PyErr_Print();
         fprintf(stderr, "CRToLx(): Error happened in (%s:%d)\n", __FILE__, __LINE__);
         exit(EXIT_FAILURE);
      }

      /*    import function pointer */
      PyObject *pFunc = PyObject_GetAttrString(pModule, "CRToLx");
      if (pFunc == NULL)
      {
         PyErr_Print();
         fprintf(stderr, "CRToLx(): Error happened in (%s:%d)\n", __FILE__, __LINE__);
         exit(EXIT_FAILURE);
      }
      PyObject *main_module = PyImport_AddModule("__main__");
      PyObject *global_dict = PyModule_GetDict(main_module);
      PyRun_SimpleString("fileInName = dirname+'/../data/CRtoLx.ascii'");
      PyObject *pFileInName = PyDict_GetItemString(global_dict, "fileInName");
      if (pFileInName == NULL)
      {
         PyErr_Print();
         fprintf(stderr, "CRToLx(): Error happened in (%s:%d)\n", __FILE__, __LINE__);
         exit(EXIT_FAILURE);
      }
      char *fileInName = PyString_AS_STRING(pFileInName);

   	/*    Set up a tuple that will contain the function arguments. */
      PyObject *pArgTuple = PyTuple_New(3);
      PyTuple_SetItem(pArgTuple, 0, PyString_FromString(fileInName));
      PyTuple_SetItem(pArgTuple, 1, PyFloat_FromDouble(z));
      PyTuple_SetItem(pArgTuple, 2, PyFloat_FromDouble(ZGas));

   	/*    Set up a tuple that will contain the result from the function. */
      PyObject *pResTuple = PyTuple_New(2);

   	/*    call to python function */
   	pResTuple = PyObject_CallObject(pFunc, pArgTuple);
      if (pResTuple == NULL)
      {
         PyErr_Print();
         fprintf(stderr, "CRToLx(): Error happened in (%s:%d)\n", __FILE__, __LINE__);
         exit(EXIT_FAILURE);
      }

      /*    see https://docs.python.org/2/c-api/concrete.html for Tuple functions */
      PyObject *logTGas = PyTuple_GetItem(pResTuple, 0);
      PyObject *logConv = PyTuple_GetItem(pResTuple, 1);

      int i, Ninter = PyList_Size(logTGas);
      double *x = (double *)malloc(Ninter*sizeof(double));
      double *y = (double *)malloc(Ninter*sizeof(double));
      for (i=0;i<Ninter;i++){
         //printf("%.8f %.8f\n", PyFloat_AsDouble(PyList_GetItem(logTGas, i)),  PyFloat_AsDouble(PyList_GetItem(conv, i)));
         x[i] = PyFloat_AsDouble(PyList_GetItem(logTGas, i));
         y[i] = PyFloat_AsDouble(PyList_GetItem(logConv, i));
      }

      inter_min = exp(x[0]);
      inter_max = exp(x[Ninter-1]);

      acc     = gsl_interp_accel_alloc();
      spline  = gsl_spline_alloc (gsl_interp_cspline, Ninter);

      gsl_spline_init(spline, x, y, Ninter);

      free(x);
      free(y);

   	Py_DECREF(pArgTuple);
   	Py_DECREF(pResTuple);

      /* Release the thread. No Python API allowed beyond this point. */
      PyGILState_Release(gstate);

   }


   if (TGas < inter_min || TGas > inter_max){
      return 0.0;
   }else{
      return exp(gsl_spline_eval(spline, log(TGas), acc));
   }
}


double nGas(const Model *model, double r, double Mh, double c, double z){
   /*
    *    Returns a 3D gas density profile given a set
    *    of parameters, truncated at r = r_vir
    *    For non-halo-model profiles a mass is needed
    *    for halo truncation. Put NAN for a non-truncated
    *    halo.
    *
    *    If computed using the halo model, the gas
    *    profile is integrated over the halo mass
    *    function times the galaxy HOD.
    */

   double log10Mh, n0, beta, rc, rtrunc;

   if(model->hod){
      log10Mh = log10(Mh);

      n0 = pow(10.0, inter_gas_log10n0(model, log10Mh));
      beta = pow(10.0, inter_gas_log10beta(model, log10Mh));
      rc = pow(10.0, inter_gas_log10rc(model, log10Mh));
   }else{
      n0 = pow(10.0, model->gas_log10n0);
      beta = pow(10.0, model->gas_log10beta);
      rc = pow(10.0, model->gas_log10rc);
   }


   /* truncation radius */
   if (!isnan(Mh)){
      // rtrunc = r_vir(model, Mh, c, z);
      rtrunc = rh(model, Mh, c, z);
   }else{
      rtrunc = RMAX;
   }

   if (r < rtrunc){
      return betaModel(r, n0, beta, rc);
   }else{
      return 0.0;
   }
}


double betaModel(double r, double n0, double beta, double rc){
   /*
    *    Returns a beta-model profile given n0, beta, and rc.
    */
   return n0 * pow(1.0 + pow(r/rc, 2.0), -3.0*beta/2.0);
}


/*
#define LOG10MH_1 12.9478
#define LOG10MH_2 13.2884
#define LOG10MH_3 13.6673
#define LOG10MH_4 14.0646
*/
#define LOG10MH_1 12.00
#define LOG10MH_2 13.00
#define LOG10MH_3 14.00
#define LOG10MH_4 15.00


#define CONCAT2(a, b)  a##b
#define CONCAT(a, b) CONCAT2(a, b)

#define PARA gas_log10n0
double CONCAT(inter_, PARA)(const Model *model, double log10Mh){

   return model->CONCAT(PARA, _1) + model->CONCAT(PARA, _2) * (log10Mh-14.0);

#if 0
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

#endif

}
#undef PARA

#define PARA gas_log10beta
double CONCAT(inter_, PARA)(const Model *model, double log10Mh){

   // return model->CONCAT(PARA, _1) + model->CONCAT(PARA, _2) * (log10Mh-14.0);


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

}
#undef PARA


#define PARA gas_log10rc
double CONCAT(inter_, PARA)(const Model *model, double log10Mh){

   // return model->CONCAT(PARA, _1) + model->CONCAT(PARA, _2) * (log10Mh-14.0);


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



}
#undef PARA

#undef LOG10MH_1
#undef LOG10MH_2
#undef LOG10MH_3
#undef LOG10MH_4

#undef CONCAT2
#undef CONCAT

void copyModelXRay(const Model *from, Model *to){
   /*    Copies model "from" to model "to" */

   to->hod = from->hod;
   to->gas_log10n0 = from->gas_log10n0;
   to->gas_log10beta = from->gas_log10beta;
   to->gas_log10rc = from->gas_log10rc;
   to->gas_log10n0_1 = from->gas_log10n0_1;
   to->gas_log10n0_2 = from->gas_log10n0_2;
   to->gas_log10n0_3 = from->gas_log10n0_3;
   to->gas_log10n0_4 = from->gas_log10n0_4;
   to->gas_log10beta_1 = from->gas_log10beta_1;
   to->gas_log10beta_2 = from->gas_log10beta_2;
   to->gas_log10beta_3 = from->gas_log10beta_3;
   to->gas_log10beta_4 = from->gas_log10beta_4;
   to->gas_log10rc_1 = from->gas_log10rc_1;
   to->gas_log10rc_2 = from->gas_log10rc_2;
   to->gas_log10rc_3 = from->gas_log10rc_3;
   to->gas_log10rc_4 = from->gas_log10rc_4;

   return;

}

int changeModelXRay(const Model *before, const Model *after){
   /* test if any of the X-ray parameters changed */

   int result = 0;

   result += assert_int(before->hod, after->hod);
   result += assert_float(before->gas_log10n0, after->gas_log10n0);
   result += assert_float(before->gas_log10beta, after->gas_log10beta);
   result += assert_float(before->gas_log10rc, after->gas_log10rc);
   result += assert_float(before->gas_log10n0_1, after->gas_log10n0_1);
   result += assert_float(before->gas_log10n0_2, after->gas_log10n0_2);
   result += assert_float(before->gas_log10n0_3, after->gas_log10n0_3);
   result += assert_float(before->gas_log10n0_4, after->gas_log10n0_4);
   result += assert_float(before->gas_log10beta_1, after->gas_log10beta_1);
   result += assert_float(before->gas_log10beta_2, after->gas_log10beta_2);
   result += assert_float(before->gas_log10beta_3, after->gas_log10beta_3);
   result += assert_float(before->gas_log10beta_4, after->gas_log10beta_4);
   result += assert_float(before->gas_log10rc_1, after->gas_log10rc_1);
   result += assert_float(before->gas_log10rc_2, after->gas_log10rc_2);
   result += assert_float(before->gas_log10rc_3, after->gas_log10rc_3);
   result += assert_float(before->gas_log10rc_4, after->gas_log10rc_4);

   return result;

}

#if  0


double betaModelSqProj(double r, double n0, double beta, double rc){
   /*
   Returns a beta-model profile square projected given n0, beta, and rc.
   See Hudson D. S. et al. (2010)
   */

   double I0 = pow(n0, 2.0) * rc * sqrt(M_PI) * gsl_sf_gamma(3.0*beta - 0.5) / gsl_sf_gamma(3.0*beta);
   return I0 * pow(1.0 + pow(r/rc, 2.0), -3.0*beta + 0.5);
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

#endif
