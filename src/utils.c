/* ------------------------------------------------------------ *
 * utils.c                                                      *
 * halomodel library                                            *
 * Jean Coupon 2016                                             *
 * ------------------------------------------------------------ */

#include "utils.h"

double xi_from_Pkr(gsl_function *ak, double r_prime, FFTLog_config *fc){

   /* The fast FFT algorithm is no longer correct if P(k) = P(k,r), that's
   why the algorithm above is only valid at the point r = r' and requires
   an interpolation. It means that you have to run this routine as many times
   as you need to evaluate xi(r'). The number of point fc->N below is the
   parameter to play with in order to get accurate and fast evaluation of
   xi(r').
   */

   int i;

   double *r    = malloc(fc->N*sizeof(double));
   double *ar   = malloc(fc->N*sizeof(double));
   double *logr = malloc(fc->N*sizeof(double));

   FFTLog(fc,ak,r,ar,-1);

   for(i=0;i<fc->N;i++) logr[i] = log(r[i]);

   /* interpolation */
   gsl_interp_accel *acc = gsl_interp_accel_alloc ();
   gsl_spline *spline    = gsl_spline_alloc (gsl_interp_cspline,fc->N);

   gsl_spline_init (spline, logr, ar, fc->N);

   double inter_min = r[0];
   double inter_max = r[fc->N-1];

   double result = 0.0;

   if (inter_min < r_prime && r_prime < inter_max){
      result = gsl_spline_eval (spline,log(r_prime), acc);
   }

   free(r);
   free(ar);
   free(logr);
   gsl_spline_free (spline);
   gsl_interp_accel_free (acc);

   return result*pow(2.0*pi*r_prime,-1.5);
}


void FFTLog(FFTLog_config *fc, const gsl_function *ar_in, double *k, double *ak_out, int dir){
  /* Hamilton 2000. http://casa.colorado.edu/~ajsh/FFTLog/

     The FFTLog algorithm for taking the discrete Hankel transform, equation (22),
     of a sequence an of N logarithmically spaced points is:

     * FFT an to obtain the Fourier coefficients cm, equation (13);
     * multiply by um given by equations (18) and (19) to obtain cm um;
     * FFT cm um back to obtain the discrete Hankel transform ãn, equation (21).

     A variant of the algorithm is to sandwich the above operations with power
     law biasing and unbiasing operations. For example, one way to take the
     unbiased continuous Hankel transform Ã(k) of a function A(r), equation (4),
     is to bias A(r) and Ã(k) with power laws, equation (3), and take a biased Hankel transform,
     equation (2). The discrete equivalent of this is:

     * Bias An with a power law to obtain an = An rn-q, equation (3);
     * FFT an to obtain the Fourier coefficients cm, equation (13);
     * multiply by um given by equations (18) and (19) to obtain cm um;
     * FFT cm um back to obtain the discrete Hankel transform ãn, equation (21);
     * Unbias ãn with a power law to obtain Ãn = ãnkn-q, equation (3).

     In order to get xi(r) from FFTLog we need to re-write
     the 2-D Hankel transform into a 3-D one.
     We know that

     sqrt(2/pi) sin(x) = sqrt(x) J_(1/2)(x)

     so

           infinity
             /
     xi(r) = | P(k,r) sin(kr)/(kr) k^2/(2 PI^2) dr
            /
            0

     becomes

                                 infinity
                                   /
     xi(r) r^(3/2) (2 PI)^(3/2) =  | P(k,r) k^(3/2) J_(1/2) r dr
                                  /
                                  0
  */


  int i;

  double logrmin  = log(fc->min);
  double logrmax  = log(fc->max);
  double r, dlogr = (logrmax - logrmin)/(double)fc->N;
  double logrc    = (logrmax + logrmin)/2.0;
  double nc       = (double)(fc->N+1)/2.0-1;
  double logkc    = log(fc->kr)-logrc;

  /* write signal */
  for(i=0; i<fc->N; i++){
    k[i] = exp(logkc+((double)i-nc)*dlogr);
    r  = exp(logrc+((double)i-nc)*dlogr);
    fc->an[i][0] = ar_in->function(r,(void*)(ar_in->params))*pow(r,-(double)dir*fc->q);
    fc->an[i][1] = 0.0;
  }

  /* cm's: FFT forward */
  fftw_execute(fc->p_forward);

  /* um*cm */
  fc->cmum[0][0] = fc->cm[0][0]*fc->um[0][0] - fc->cm[0][1]*fc->um[0][1];
  fc->cmum[0][1] = fc->cm[0][0]*fc->um[0][1] + fc->cm[0][1]*fc->um[0][0];
  for(i=1;i<fc->N/2+1;i++){
    fc->cmum[i][0] = fc->cm[i][0]*fc->um[i][0] - fc->cm[i][1]*fc->um[i][1];
    fc->cmum[i][1] = fc->cm[i][0]*fc->um[i][1] + fc->cm[i][1]*fc->um[i][0];
    /* Hermitian symetry (i.e. to get a real signal after FFT back) */
    fc->cmum[fc->N-i][0] = fc->cmum[i][0];
    fc->cmum[fc->N-i][1] = -fc->cmum[i][1];
  }

  /* ak's: FFT backward */
   fftw_execute(fc->p_backward);

   /* reverse array... */
  for(i=0;i<fc->N/2;i++) FFTLog_SWAP(fc->ak[i][0],fc->ak[fc->N-i-1][0]);

  /* ...and write ak(k) */
  for(i=0;i<fc->N;i++) ak_out[i] = fc->ak[i][0]*pow(k[i],-(double)dir*fc->q)/(double)fc->N;


  return;
}

FFTLog_config *FFTLog_init(int N, double min, double max, double q, double mu){
  /* Initializes what FFTLog needs. */


  FFTLog_config *fc = (FFTLog_config*)malloc(sizeof(FFTLog_config));

  /* FFTW3 Initialization */
  fc->min        = min;
  fc->max        = max;
  fc->q          = q;
  fc->mu         = mu;
  fc->N          = N;
  fc->an         = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
  fc->ak         = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
  fc->cm         = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
  fc->um         = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
  fc->cmum       = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*N);
  fc->p_forward  = fftw_plan_dft_1d(N,fc->an,fc->cm,FFTW_FORWARD,FFTW_ESTIMATE);
  fc->p_backward = fftw_plan_dft_1d(N,fc->cmum,fc->ak,FFTW_BACKWARD,FFTW_ESTIMATE);

  /* um's */
  FFTLog_complex z,result;
  double L = log(max)-log(min);
  fc->kr   = 1.0;
  int i;

  for(i=0;i<fc->N/2+1;i++){
    z.re   = fc->q;
    z.im   = 2.0*M_PI*(double)i/L;
    result = FFTLog_U_mu(mu,z);

    /* Multiply by (kr)^-2PIim/L */
    result.amp *= 1.0;
    result.arg += -2.0*M_PI*(double)i*log(fc->kr)/L;

    fc->um[i][0] = result.amp*cos(result.arg);
    fc->um[i][1] = result.amp*sin(result.arg);
  }

  /* If N even, mutiply by real part only */
  if(PARITY(fc->N) == EVEN) fc->um[fc->N/2][1] = 0.0;

  return fc;
}

void FFTLog_free(FFTLog_config *fc){

  fftw_destroy_plan(fc->p_forward);
  fftw_destroy_plan(fc->p_backward);
  fftw_free(fc->an);
  fftw_free(fc->ak);
  fftw_free(fc->cm);
  fftw_free(fc->um);
  fftw_free(fc->cmum);
  free(fc);

  return;
}

FFTLog_complex FFTLog_U_mu(double mu, FFTLog_complex z){
  /*Computes 2^z Gamma[(mu + 1 - z)/2]/Gamma[(mu + 1 - z)/2]
              1                2                 3
  */
  double amp1,arg1;
  gsl_sf_result lnamp2,arg2,lnamp3,arg3;

  FFTLog_complex result;

  /* 2^z */
  amp1 = exp(z.re*log(2.0));
  arg1 = z.im*log(2.0);

  /* Gamma 1 */
  FFTLog_complex zplus;
  zplus.re = (mu + 1.0 + z.re)/2.0;
  zplus.im = z.im/2.0;
  gsl_sf_lngamma_complex_e(zplus.re,zplus.im,&lnamp2,&arg2);

  /* Gamma 2 */
  FFTLog_complex zminus;
  zminus.re = (mu + 1.0 - z.re)/2.0;
  zminus.im = - z.im/2.0;
  gsl_sf_lngamma_complex_e(zminus.re,zminus.im,&lnamp3,&arg3);

  /* Result */
  result.amp = amp1*exp(lnamp2.val)*exp(-lnamp3.val);
  result.arg = arg1 + arg2.val - arg3.val;
  result.re = result.amp*cos(result.arg);
  result.im = result.amp*sin(result.arg);

  return result;
}

double int_gsl_FFT(funcwithparsHalomodel func, void *params, double a, double b, double eps)
{
  int n = 1000, status;
  gsl_integration_workspace * w = gsl_integration_workspace_alloc (n);
  double result, result_err;

  gsl_function F;
  F.function = &integrand_gsl;

  gsl_int_params p;
  p.func   = func;
  p.params = params;
  F.params = &p;

  gsl_set_error_handler_off();
 //  status = gsl_integration_qag (&F, a, b, eps, eps, n, GSL_INTEG_GAUSS51, w, &result, &result_err);
  status = gsl_integration_qag (&F, a, b, 0.0, eps, n, GSL_INTEG_GAUSS51, w, &result, &result_err);

  gsl_integration_workspace_free (w);

  return result;
}



double int_gsl(funcwithparsHalomodel func, void *params, double a, double b, double eps)
{
  int n = 1000, status;
  gsl_integration_workspace * w = gsl_integration_workspace_alloc (n);
  double result, result_err;

  gsl_function F;
  F.function = &integrand_gsl;

  gsl_int_params p;
  p.func   = func;
  p.params = params;
  F.params = &p;

  gsl_set_error_handler_off();
  status = gsl_integration_qag (&F, a, b, eps, eps, n, GSL_INTEG_GAUSS51, w, &result, &result_err);

  gsl_integration_workspace_free (w);

  return result;
}

double integrand_gsl(double x, void *p)
{
  double res;
  void *params =  ((gsl_int_params *)p)->params;

  res = ((gsl_int_params *)p)->func(x,params);

  return res;
}

/*
double sinc(double x)
{
   if (x<0.001 && x>-0.001) return 1. - x*x/6.;
   else return sin(x)/x;
}

*/


double trapz(double *x, double *y, int N){

  int i;
  double result = 0.0;

  for(i=0;i<N-1;i++){
    result += (x[i+1] - x[i]) * (y[i] + y[i+1]) / 2.0;
  }
  return result;
}

double King(double r, double A, double r_c, double alpha){
  /*
    Returns King's profile.
  */
  return A * pow(1.0+(r/r_c)*(r/r_c), -alpha);
}

int assert_float(double before, double after){
   /*
    *    asserts two floats
    */

   if (isnan(before) && isnan(after)){
      return 0;
   }else if (isnan(before) && !isnan(after)){
      return 1;
   }else if(!isnan(before) && isnan(after)){
      return 1;
   }else if(fabs(before - after) > 1.e-8){
      return 1;
   }else{
      return 0;
   }


}
