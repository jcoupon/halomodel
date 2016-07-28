/*
 *    xray.c
 *    halomodel library
 *    Jean Coupon 2015-2016
 */

#include "xray.h"
#include <Python.h>

int main()
{
   /*
    *    THIS IS FOR TESTS ONLY
    *    Needs to define PYTHONHOME
    *    $ setenv PYTHONHOME ~/anaconda/
    */

   Py_Initialize();
   PyRun_SimpleString("import numpy as np; print \"Hello world\"");
   Py_Finalize();

   return 0;
}

void SigmaIx(const Model *model, double *R, int N, double Mh, double c, double z, int obs_type, double *result)
{
   /*
    *    Computes projected  X-ray luminosity profiles
    *    See Vikhlinin et al. (2006) and Cavaliere & Fusco-Femiano (1978)
    *
    *    If model.hod is set to 1, it will use the full
    *    HOD parametrisation, and needs a halo mass function. Otherwise
    *    if set to 0, it will only use the three parameters that controls
    *    the shape of a X-ray profile.
    */


   if(obs_type == all){
      SigmaIxAll(model, R, N, Mh, c, z, result);
      return;
   }


   /*    interpolate to speed up integration for projection and PSF convolution */
   int i, j, k, Ninter = 128;

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
   if (!isnan(model->XMM_PSF_A)){

      int N_PSF = 128;
      double Rpp;
      double *theta = (double *)malloc(N_PSF*sizeof(double));
      double *Rp = (double *)malloc(N_PSF*sizeof(double));
      double *intForTheta = (double *)malloc(N_PSF*sizeof(double));
      double *intForRp = (double *)malloc(N_PSF*sizeof(double));

      double *PSF_profile = (double *)malloc(N_PSF*sizeof(double));

      for(i=0;i<N_PSF;i++){
         Rp[i]    = exp(log(rinter_min)+dlogrinter*(double)i);
         theta[i] = 0.0+2.0*M_PI / (double)(N_PSF-1) * (double)i;
         intForTheta[i] = 0.0;
         intForRp[i] = 0.0;
         PSF_profile[i] = King(Rp[i], model->XMM_PSF_A, model->XMM_PSF_rc, model->XMM_PSF_alpha);
      }

      double norm_PSF = 1.0/trapz(Rp, PSF_profile, N_PSF);

      for(i=0;i<N;i++){                                        /*    Main loop */
         for(j=0;j<N_PSF;j++){                                 /*    loop over R' - trapeze integration */
            for(k=0;k<N_PSF;k++){                              /*    loop over theta - trapeze integration  */
               Rpp = sqrt(R[i]*R[i] + Rp[j]*Rp[j] - 2.0*R[i]*Rp[j]*cos(theta[k]));
               if(logrinter[0] < log(Rpp) && log(Rpp) < logrinter[Ninter-1]){
                  intForTheta[k] = gsl_spline_eval(spline, log(Rpp), acc);
               }
            }
            intForRp[j] = King(Rp[j], model->XMM_PSF_A, model->XMM_PSF_rc, model->XMM_PSF_alpha) * trapz(theta, intForTheta, N_PSF);
         }
         result[i] = norm_PSF * 1.0 / (2.0 * M_PI) * trapz(Rp, intForRp, N_PSF);
      }

      free(intForTheta);
      free(intForRp);
      free(theta);
      free(Rp);

   }else{
      /*    ... or simply return result */
      for(i=0;i<N;i++){
         if(logrinter[0] < log(R[i]) && log(R[i]) < logrinter[Ninter-1]){
            result[i] = gsl_spline_eval(spline, log(R[i]), acc);
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

         double Tx, ZGas;
         Tx = MhToTx(model, Mh, z);
         ZGas = MhToZGas(model, Mh, z);

         double fac = CRToLx(model, z, Tx, ZGas);

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

   double Tx, ZGas;
   Tx = MhToTx(model, Mh, z);
   ZGas = MhToZGas(model, Mh, z);

   double fac = CRToLx(model, z, Tx, ZGas);

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
    *    x-ray luminosity power spectrum.
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

      double Tx, ZGas;
      Tx = MhToTx(model, Mh, z);
      ZGas = MhToZGas(model, Mh, z);

      double fac = CRToLx(model, z, Tx, ZGas);
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

   double Tx, ZGas;
   Tx = MhToTx(model, Mh, z);
   ZGas = MhToZGas(model, Mh, z);

   double fac = CRToLx(model, z, Tx, ZGas);
   double Norm = NormIx(model, Mh, c, z);

   if (fac*Norm > 0.0){
      return  Ngal_s(model, Mh, model->log10Mstar_min, model->log10Mstar_max)
         * pow(uIx(model, k, Mh, c, z), 2.0) * Norm
         * dndlnMh(model, Mh, z) / fac * SCALE;

   }else{
      return 0.0;
   }

}

//void uIx(const Model *model, const double *k, int N, double Mh, double c, double z, double *result)
double uIx(const Model *model, double k, double Mh, double c, double z)
{
   /*
    *    Returns the normalised (out to r_vir radius)
    *    fourier transform of the brightness profile.
    *    The constants are set by the wrapper and depend on
    *    halo properties, redshift, etc., so that all the
    *    quantities depending on cosmology are managed by
    *    the wrapper.
    */


   static int firstcall = 1;

   static gsl_spline2d *spline = NULL;
   static gsl_interp_accel *xacc = NULL;
   static gsl_interp_accel *yacc = NULL;

   static double Norm = 1.0, result, inter_xmin, inter_xmax, inter_ymin, inter_ymax;
   static double *x = NULL, *logx = NULL, dlogx = 0.0;
   static double *y = NULL, *logy = NULL, dlogy = 0.0;
   static double *za;

   static int i, j, Nx = 64, Ny = 64;

   static double c_tmp = NAN;
   static double z_tmp = NAN;

   if(firstcall){

      changeModeXRay(model);

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

   if(firstcall || changeModeXRay(model) || assert_float(c_tmp, c) || assert_float(z_tmp, z)){
      /* If first call or model parameters have changed, the table must be recomputed */

      firstcall = 0;
      c_tmp = c;
      z_tmp = z;

      params p;
      p.model = model;
      p.c = c;
      p.z = z;

      for(i=0;i<Nx;i++){    /*      loop over halo mass */
         p.Mh = x[i];
         Norm = NormIx(model, Mh, c, z);
         for(j=0;j<Ny;j++){ /*      loop over k */
            p.k = y[j];
            result = 4.0*M_PI/Norm*int_gsl(intForUIx, (void*)&p, log(1.e-6), log(2.0*M_PI/KMIN), 1.e-3);
            gsl_spline2d_set(spline, za, i, j, result);
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
   /*    Returns the integrated surface brightness  */

   static int firstcall = 1;

   int i, Ninter = 64;

   static gsl_interp_accel *acc;
   static gsl_spline *spline;
   static double inter_min, inter_max;

   static double dx, *x, *y;

   static double c_tmp;
   static double z_tmp;

   if(firstcall){

      dx = (LNMH_MAX-LNMH_MIN)/(double)Ninter;
      x = (double *)malloc(Ninter*sizeof(double));
      y = (double *)malloc(Ninter*sizeof(double));

      for(i=0;i<Ninter;i++){
         x[i]  = LNMH_MIN+dx*(double)i;
      }

      acc = gsl_interp_accel_alloc();
      spline = gsl_spline_alloc (gsl_interp_cspline, Ninter);

   }

   if(firstcall || changeModeXRay(model) || assert_float(c_tmp, c) || assert_float(z_tmp, z)){
      /* If first call or model parameters have changed, the table must be recomputed */

      firstcall = 0;
      c_tmp = c;
      z_tmp = z;

      params p;
      p.model = model;
      p.c = c;
      p.z = z;

      for(i=0;i<Ninter;i++){
         p.Mh = exp(x[i]);
         y[i] = int_gsl_QNG(intForNormIx, (void*)&p, log(RMIN), log(r_vir(model, p.Mh, c, z)), 1.0e-3);
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

   /* Re_XB in Mpc */
   double Re = model->IxXB_Re;

   /* total X-ray luminosity of binary stars in CR */
   double CR = model->IxXB_CR;

   double Ie = CR / (7.215 * M_PI * Re * Re);

   for(i=0;i<N;i++){
      /* De vaucouleur profile */
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

   /* FFTLog config */
   double q = 0.0, mu = 0.5;
   int FFT_N = 64;
   FFTLog_config *fc = FFTLog_init(FFT_N, KMIN, KMAX, q, mu);

   /* parameters to pass to the function */
   params p;
   p.model = model;
   p.z = z;
   p.c = NAN;  /*    for the HOD model, the concentration(Mh) relationship is fixed */
   p.Mh = Mh;

   /* fonction with parameters to fourier transform */
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
      p.ngp = ngal_den(model, p.logMlim, model->log10Mstar_min, model->log10Mstar_max, z, all);

      if(p.ng < 1.0e-14 || p.ngp < 1.0e-14 || r[i] < RMIN2){
         result[i] = 0.0;
      }else{
         result[i] = (p.ngp/p.ng)*pow(bias_fac, 2.0)*xi_from_Pkr(&Pk, r[i], fc);
      }
   }
   FFTLog_free(fc);


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

      return P_m_nonlin(model, k, z)*pow(int_gsl(intForP_twohalo_Ix, p, LNMH_MIN, logMlim, 1.e-3), 2.0)/ngp;

   }else{

      const double Mh = ((params *)p)->Mh;
      const double z = ((params *)p)->z;
      const double c = ((params *)p)->c;

      double Tx, ZGas;
      Tx = MhToTx(model, Mh, z);
      ZGas = MhToZGas(model, Mh, z);

      double fac = CRToLx(model, z, Tx, ZGas);
      double Norm = NormIx(model, Mh, c, z);

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

   double result, Mh = exp(logMh);

   double Tx, ZGas;
   Tx = MhToTx(model, Mh, z);
   ZGas = MhToZGas(model, Mh, z);

   double fac = CRToLx(model, z, Tx, ZGas);
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
    *    in erg s^-1 Mpc-3 / 1e.44 assuming
    *    a 3D gas profile
    *    ix \propto nGas^2
    *    = Ix if non HOD
    *    = ix if HOD (to be integrated over halo mass function)
    */

   double Tx, ZGas;
   Tx = MhToTx(model, Mh, z);
   ZGas = MhToZGas(model, Mh, z);

   return Lambda(Tx, ZGas)*pow(1.21*nGas(model, r, Mh, c, z), 2.0) / CM3TOMPC3;

}



double MhToTx(const Model *model, double Mh, double z)
{

   double log10Mh = log10(Mh);
   return pow(10.0, (log10Mh - 13.56) / 1.69);
}

double TxToMh(const Model *model, double Tx, double z)
{

   double log10Tx = log10(Tx);
   return pow(10.0, 1.69*log10Tx + 13.56);
}


double MhToZGas(const Model *model, double Mh, double z)
{
   /*
    *    Metallicity
    *
    */
   return 0.25;
}


# define CONST_MGAS (3.33054952367e+16) /* = mu * mp * 2.21 / (cm3ToMpc3 * Msun) */

double MGas(const Model *model, double r, double Mh, double c, double z){
/*
 *    Returns the total Mgas(<r) out to r
 */

   params p;
   p.model = model;
   p.Mh = Mh;
   p.c = c;
   p.z = z;

   return CONST_MGAS*int_gsl(intForMGas, (void*)&p, log(RMIN), log(r), 1.e-4);
}

double intForMGas(double logr, void *p){

   const Model *model = ((params *)p)->model;
   double r = exp(logr);
   double Mh = ((params *)p)->Mh;
   double c = ((params *)p)->c;
   double z = ((params *)p)->z;

   return r * r * 4.0 * M_PI * nGas(model, r, Mh, c, z) * r;
}

# undef CONST_MGAS


# define EPS 1.e-8
double Lambda(double Tx, double ZGas)
{
   /*
    *    Cooling function in erg s^-1 cm^3
    *
    *    Tx: gas temperature
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
      double logLambda_t[148] = {-53.56615510,-53.61945972,-53.61586553,-53.59627881,-53.56493050,-53.52508044,
         -53.48320760,-53.43961778,-53.36267128,-53.29823042,-53.24167152,-53.18988214,-53.14465362,-53.10772864,
         -53.07174493,-53.03484686,-52.99926298,-52.96619781,-52.93665900,-52.90792787,-52.85530435,-52.80876386,
         -52.76511644,-52.70837163,-52.65604690,-52.62527323,-52.59539058,-52.55359877,-52.49484301,-52.44045503,
         -52.39076269,-52.34412016,-52.30415722,-52.26570514,-52.22993946,-52.19805803,-52.16713906,-51.93329010,
         -52.00849062,-52.12096796,-52.41129527,-52.64891828,-52.78071699,-52.85041441,-52.83618696,-52.79279929,
         -52.74925294,-52.71332389,-52.68972012,-52.68080320,-52.69498740,-52.70914988,-52.72212670,-52.73527413,
         -52.74262144,-52.73855610,-52.73450722,-52.71968462,-52.69527914,-52.67020380,-52.62571720,-52.58335905,
         -52.55662215,-52.53055549,-52.49361214,-52.43968763,-52.38914448,-52.34208470,-52.29783159,-52.25983099,
         -52.22322173,-52.18904804,-52.15853237,-52.12892039,-51.50383030,-51.58091273,-51.70412319,-52.03091773,
         -52.31246909,-52.48061147,-52.57839402,-52.57225105,-52.53847964,-52.50112764,-52.47146012,-52.45674434,
         -52.45972482,-52.49169978,-52.52451737,-52.55742458,-52.59145163,-52.61724427,-52.62540927,-52.63364148,
         -52.63851197,-52.62616137,-52.61159996,-52.57416073,-52.53766603,-52.51333645,-52.48954726,-52.45553529,
         -52.40453234,-52.35631673,-52.31092037,-52.26811158,-52.23131933,-52.19585149,-52.16269797,-52.13302547,
         -52.10421237,-51.08269344,-51.16099684,-51.29117124,-51.64252357,-51.95584075,-52.15212896,-52.27295197,
         -52.27365184,-52.24816557,-52.21623015,-52.19214449,-52.18537073,-52.19926365,-52.24799864,-52.29909145,
         -52.35222739,-52.40834598,-52.45446651,-52.47656813,-52.49913775,-52.52792151,-52.53074436,-52.52968974,
         -52.50149446,-52.47283610,-52.45173633,-52.43104310,-52.40103063,-52.35399429,-52.30905046,-52.26589504,
         -52.22513922,-52.19006677,-52.15616043,-52.12444154,-52.09594712,-52.06826270};
      double logTx_t[37] = {-2.30258509,-2.12026354,-1.89711998,-1.60943791,-1.38629436,-1.20397280,-1.04982212,
         -0.91629073,-0.69314718,-0.51082562,-0.35667494,-0.22314355,-0.10536052,0.00000000,0.09531018,0.18232156
         ,0.26236426,0.33647224,0.40546511,0.47000363,0.58778666,0.69314718,0.78845736,0.91629073,1.02961942,1.09861229,
         1.16315081,1.25276297,1.38629436,1.50407740,1.60943791,1.70474809,1.79175947,1.87180218,1.94591015,2.01490302,2.07944154};
      double ZGas_t[4] = { 0.00,0.15,0.25,0.40 };

      int Nx = sizeof(logTx_t)/sizeof(logTx_t[0]);
      int Ny = sizeof(ZGas_t)/sizeof(ZGas_t[0]);

      ZGas_t[0] = ZGas_t[0];
      ZGas_t[Ny-1] = ZGas_t[Ny-1];

      /* initialize interpolation */
      const gsl_interp2d_type *T = gsl_interp2d_bilinear;
      spline = gsl_spline2d_alloc(T, Nx, Ny);
      xacc   = gsl_interp_accel_alloc();
      yacc   = gsl_interp_accel_alloc();

      inter_xmin = exp(logTx_t[0]);
      inter_xmax = exp(logTx_t[Nx-1]);

      /* EPS is to allow to get values at Z=0.0 and Z=0.40*/
      inter_ymin = ZGas_t[0]-EPS;
      inter_ymax = ZGas_t[Ny-1]+EPS;

      /* set interpolation */
      gsl_spline2d_init(spline, logTx_t, ZGas_t, logLambda_t, Nx, Ny);

   }

   if (inter_xmin < Tx && Tx < inter_xmax && inter_ymin < ZGas && ZGas < inter_ymax){
      return exp(gsl_spline2d_eval(spline, log(Tx), ZGas, xacc, yacc));
   }else{
      return 0.0;
   }

   /* Ettori (2015) */
   // return 1.1995 * 0.85e-23 * pow(Tx, 0.5);
}
# undef EPS

double CRToLx(const Model *model, double z, double Tx, double ZGas)
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
      PyObject *logTx = PyTuple_GetItem(pResTuple, 0);
      PyObject *logConv = PyTuple_GetItem(pResTuple, 1);

      int i, Ninter = PyList_Size(logTx);
      double *x = (double *)malloc(Ninter*sizeof(double));
      double *y = (double *)malloc(Ninter*sizeof(double));
      for (i=0;i<Ninter;i++){
         //printf("%.8f %.8f\n", PyFloat_AsDouble(PyList_GetItem(logTx, i)),  PyFloat_AsDouble(PyList_GetItem(conv, i)));
         x[i] = PyFloat_AsDouble(PyList_GetItem(logTx, i));
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


   if (Tx < inter_min || Tx > inter_max){
      return 0.0;
   }else{
      return exp(gsl_spline_eval(spline, log(Tx), acc));
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
      rtrunc = r_vir(model, Mh, c, z);
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


#define LOG10MH_1 12.9478
#define LOG10MH_2 13.2884
#define LOG10MH_3 13.6673
#define LOG10MH_4 14.0646

#define CONCAT2(a, b)  a##b
#define CONCAT(a, b) CONCAT2(a, b)

#define PARA gas_log10n0
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


#define PARA gas_log10rc
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

#undef LOG10MH_1
#undef LOG10MH_2
#undef LOG10MH_3
#undef LOG10MH_4

#undef CONCAT2
#undef CONCAT



int changeModeXRay(const Model *model){
   /* test if any of the X-ray parameters changed */

   static Model model_tmp;
   static int firstcall = 1;
   int result;

   if (firstcall) {
      firstcall = 0;

        model_tmp.gas_log10n0 = model->gas_log10n0;
        model_tmp.gas_log10beta = model->gas_log10beta;
        model_tmp.gas_log10rc = model->gas_log10rc;
        model_tmp.gas_log10n0_1 = model->gas_log10n0_1;
        model_tmp.gas_log10n0_2 = model->gas_log10n0_2;
        model_tmp.gas_log10n0_3 = model->gas_log10n0_3;
        model_tmp.gas_log10n0_4 = model->gas_log10n0_4;
        model_tmp.gas_log10beta_1 = model->gas_log10beta_1;
        model_tmp.gas_log10beta_2 = model->gas_log10beta_2;
        model_tmp.gas_log10beta_3 = model->gas_log10beta_3;
        model_tmp.gas_log10beta_4 = model->gas_log10beta_4;
        model_tmp.gas_log10rc_1 = model->gas_log10rc_1;
        model_tmp.gas_log10rc_2 = model->gas_log10rc_2;
        model_tmp.gas_log10rc_3 = model->gas_log10rc_3;
        model_tmp.gas_log10rc_4 = model->gas_log10rc_4;

   }

   result = 0;
   if (assert_float(model_tmp.gas_log10n0, model->gas_log10n0)){
      model_tmp.gas_log10n0 = model->gas_log10n0;
      result = 1;
   }
   if (assert_float(model_tmp.gas_log10beta, model->gas_log10beta)){
      model_tmp.gas_log10beta = model->gas_log10beta;
      result = 1;
   }
   if (assert_float(model_tmp.gas_log10rc, model->gas_log10rc)){
      model_tmp.gas_log10rc = model->gas_log10rc;
      result = 1;
   }
   if (assert_float(model_tmp.gas_log10n0_1, model->gas_log10n0_1)){
      model_tmp.gas_log10n0_1 = model->gas_log10n0_1;
      result = 1;
   }
   if (assert_float(model_tmp.gas_log10n0_2, model->gas_log10n0_2)){
      model_tmp.gas_log10n0_2 = model->gas_log10n0_2;
      result = 1;
   }
   if (assert_float(model_tmp.gas_log10n0_3, model->gas_log10n0_3)){
      model_tmp.gas_log10n0_3 = model->gas_log10n0_3;
      result = 1;
   }
   if (assert_float(model_tmp.gas_log10n0_4, model->gas_log10n0_4)){
      model_tmp.gas_log10n0_4 = model->gas_log10n0_4;
      result = 1;
   }
   if (assert_float(model_tmp.gas_log10beta_1, model->gas_log10beta_1)){
      model_tmp.gas_log10beta_1 = model->gas_log10beta_1;
      result = 1;
   }
   if (assert_float(model_tmp.gas_log10beta_2, model->gas_log10beta_2)){
      model_tmp.gas_log10beta_2 = model->gas_log10beta_2;
      result = 1;
   }
   if (assert_float(model_tmp.gas_log10beta_3, model->gas_log10beta_3)){
      model_tmp.gas_log10beta_3 = model->gas_log10beta_3;
      result = 1;
   }
   if (assert_float(model_tmp.gas_log10beta_4, model->gas_log10beta_4)){
      model_tmp.gas_log10beta_4 = model->gas_log10beta_4;
      result = 1;
   }
   if (assert_float(model_tmp.gas_log10rc_1, model->gas_log10rc_1)){
      model_tmp.gas_log10rc_1 = model->gas_log10rc_1;
      result = 1;
   }
   if (assert_float(model_tmp.gas_log10rc_2, model->gas_log10rc_2)){
      model_tmp.gas_log10rc_2 = model->gas_log10rc_2;
      result = 1;
   }
   if (assert_float(model_tmp.gas_log10rc_3, model->gas_log10rc_3)){
      model_tmp.gas_log10rc_3 = model->gas_log10rc_3;
      result = 1;
   }
   if (assert_float(model_tmp.gas_log10rc_4, model->gas_log10rc_4)){
      model_tmp.gas_log10rc_4 = model->gas_log10rc_4;
      result = 1;
   }

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
