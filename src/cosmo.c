/*
 *    cosmo.c
 *    halomodel library
 *    Jean Coupon 2016 - 2017
 */

#include "cosmo.h"

/*
 *    cosmological parameters
 */

cosmo *initCosmo(const Model *model){
/*    nicaea library */

   static int firstcall = 1;
   static cosmo *result=NULL;

   if(firstcall){

      firstcall = 0;

      error *myerr = NULL, **err;
      err = &myerr;

      result = init_parameters(
         model->Omega_m,         // Omega_m		0.27	# Matter density, cold dark matter + baryons
         model->Omega_de,        // Omega_de	0.73		# Dark-energy density
         -1.0,                   // w0_de		-1.0		# Dark-energy equation of state parameter (constant term)
         0.0,                    // w1_de		 0.0	  	# Dark-energy equation of state parameter (linear term)
         NULL,                   // W_POLY_DE
         0,                      // N_POLY_DE
         model->H0/100.0,        // h_100		0.70		   # Dimensionless Hubble parameter
         model->Omega_b,         // Omega_b		0.045		# Baryon density
         0.0,                    // Omega_nu_mass	0.0	# Massive neutrino density (so far only for CMB)
         0.0,                    // Neff_nu_mass	0.0	# Effective number of massive neutrinos (only CMB)
         model->sigma_8,                  // normalization	0.8	# This is sigma_8 if normmode=0 below
         model->n_s,                  // n_spec		0.96 		# Scalar power-spectrum index

         /*
         Power spectrum prescription
         linear	Linear power spectrum
         pd96		Peacock&Dodds (1996)
         smith03	Smith et al. (2003)
         smith03_de	Smith et al. (2003) + dark-energy correction from icosmo.org
         smith03_revised
         Takahashi et al. 2012, revised halofit parameters
         coyote10	Coyote emulator v1, Heitmann, Lawrence et al. 2009, 2010
         coyote13	Coyote emulator v2, Heitmann et al. 2013
         */
         // smith03_revised,
         smith03_de, // DEBUGGING: matches Coupon et al. (2015)

         /*
         Transfer function
         bbks		Bardeen, Bond, Kaiser & Szalay (1986)
         eisenhu	Eisenstein & Hu (1998) 'shape fit'
         eisenhu_osc  Eisenstein & Hu (1998) with baryon wiggles
         */
         eisenhu_osc,

         /*
         Linear growth factor
         heath	Heath (1977) fitting formula
         growth_de	Numerical integration of density ODE (recommended)
         */
         growth_de,

         /*
         Dark-energy parametrisation
         jassal	w(a) = w_0 + w_1*a*(1-a)
         linder	w(a) = w_0 + w_1*(1-a)
         */
         linder,

         norm_s8,     // normalisation mode

         0.1,         // a_min		0.2		# For late Universe stuff
         err);
      quitOnError(*err, __LINE__, stderr);

   }

   return result;
}


/*
 *    Angular correlation function of dark matter
 */

void xi_m(const Model *model, double *r, int N, double z, double *result){

   /*
    *    Returns the dark matter correlation function.
    *    i.e. the fourier transform P_m_nonlin.
    */

   static int firstcall = 1;
   static gsl_interp_accel *acc;
   static gsl_spline *spline;
   static double inter_min, inter_max;

   int i;

   if(firstcall){

      firstcall = 0;

      /*    FFTLog config */
      double q = 0.0, mu = 0.5;
      int j, FFT_N = 64;
      FFTLog_config *fc = FFTLog_init(FFT_N, KMIN, KMAX, q, mu);
      double *r_FFT     = (double *)malloc(FFT_N*sizeof(double));
      double *ar        = (double *)malloc(FFT_N*sizeof(double));
      double *logr_FFT  = (double *)malloc(FFT_N*sizeof(double));

      /*    parameters to pass to the function */
      params p;
      p.model = model;
      p.z = z;

      /*    fonction with parameters to fourier transform */
      gsl_function Pk;
      Pk.function = &intForxi_m;
      Pk.params   = &p;

      /*    fourier transform... */
      FFTLog(fc, &Pk, r_FFT, ar, -1);

      /*    return values through interpolation */
      acc    = gsl_interp_accel_alloc ();
      spline = gsl_spline_alloc (gsl_interp_cspline, FFT_N);

      /*    attention: N and FFT_N are different */
      for(j=0;j<FFT_N;j++) logr_FFT[j] = log(r_FFT[j]);
      gsl_spline_init (spline, logr_FFT, ar, FFT_N);

      inter_min = logr_FFT[0];
      inter_max = logr_FFT[FFT_N-1];

      /*    free memory */
      free(r_FFT);
      free(ar);
      free(logr_FFT);
      FFTLog_free(fc);

   }

   // && r[i] < RMAX

   for(i=0;i<N;i++){
      if (inter_min < log(r[i]) && log(r[i]) <  inter_max ){
         result[i] = gsl_spline_eval(spline, log(r[i]), acc)*pow(2.0*M_PI*r[i],-1.5);
      }else{
         result[i] = 0.0;
      }
   }

   return;
}



double intForxi_m(double k, void *p)
{
   const Model *model = ((params *)p)->model;
   const double z  = ((params *)p)->z;

   return pow(k, 1.5)*P_m_nonlin(model, k, z);

}

/*
 *    halo bias
 */

double bias_h(const Model *model, double Mh, double z){
  /*
   *     Halo bias with respect to matter density.
   *     Units: Mh in M_sol^-1 h.
   */

   static int firstcall = 1;
   static gsl_interp_accel *acc;
   static gsl_spline *spline;
   static double inter_min, inter_max;

   static double z_tmp = NAN;

   if(firstcall || assert_float(z_tmp, z)){
      // TODO: also check that model didn't change

      firstcall = 0;
      z_tmp = z;

      double sigma;

      int i, Ninter  = 64;
      double dx = (LNMH_MAX-LNMH_MIN)/(double)Ninter;
      double *x = (double *)malloc(Ninter*sizeof(double));
      double *y = (double *)malloc(Ninter*sizeof(double));

      for(i=0;i<Ninter;i++){
         x[i] = LNMH_MIN+dx*(double)i;
         sigma = sqrt(sigma2M(model, exp(x[i])));
         y[i] = b_sigma(model, sigma, z);
      }

      inter_min = exp(x[0]);
      inter_max = exp(x[Ninter-1]);

      acc = gsl_interp_accel_alloc();
      spline = gsl_spline_alloc (gsl_interp_cspline, Ninter);
      gsl_spline_init(spline, x, y, Ninter);

      free(x);
      free(y);
   }

   if (Mh < inter_min || Mh > inter_max){
      return 0.0;
   }else{
      return gsl_spline_eval(spline, log(Mh), acc);
   }

}

/*
 *    halo mass function
 */

double dndlnMh(const Model *model, double Mh, double z){
   /*
    *    Mass function: halo number density per log unit mass.
    *    dn / dlnM = dn / d ln sigma^{-1} * d ln sigma^{-1} / dlnM
                   = rho_0 / M * f(sigma) * dsigma^{-1} / dln M
    *    Units: [dn/dlnM] = h^3 Mpc^-3  Msun^-1 h.
    *    if unset, it will take input as Msol and
    *    outputs dn/dlnM in Mpc^-3  Msun.
    */

   static int firstcall = 1;
   static gsl_interp_accel *acc;
   static gsl_spline *spline;
   static double inter_min, inter_max;

   if(firstcall){

      firstcall = 0;

      double sigma;

      double rhobar = rho_bar(model, 0.0);

      int i, Ninter = 128;
      double dx = (LNMH_MAX-LNMH_MIN)/(double)Ninter;
      double *x = (double *)malloc(Ninter*sizeof(double));
      double *y = (double *)malloc(Ninter*sizeof(double));

      for(i=0;i<Ninter;i++){
         x[i]  = LNMH_MIN+dx*(double)i;
         sigma = sqrt(sigma2M(model, exp(x[i])));
         y[i]  = log(f_sigma(model, sigma, z) * rhobar / exp(x[i]) * dsigmaM_m1_dlnM(model, exp(x[i])));
      }

      inter_min = exp(x[0]);
      inter_max = exp(x[Ninter-1]);

      acc = gsl_interp_accel_alloc();
      spline = gsl_spline_alloc (gsl_interp_cspline, Ninter);

      gsl_spline_init(spline, x, y, Ninter);

      free(x);
      free(y);
   }

   if (Mh < inter_min || Mh > inter_max){
      return 0.0;
   }else{
      return exp(gsl_spline_eval(spline, log(Mh), acc));
   }

}


/*
 *    halo profile functions
 */


double uHalo(const Model *model, double k, double Mh, double c, double z)
{
   /*
    *    Fourier transform of the halo profile.
    *    To assume c(Mh) relation, set c = -1.0;
    *    uHalo is interpolated in Mh k space if c(Mh) relation
    *    is assumed.
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

   if(firstcall){

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

   if(firstcall || assert_float(c_tmp, c)){
      /* If first call or if c has changed, the table must be recomputed */

      firstcall = 0;
      c_tmp = c;

      params p;
      p.model = model;
      p.c = c;
      p.z = z;

      for(i=0;i<Nx;i++){    /*      loop over halo mass */
         p.Mh = x[i];
         Norm = x[i];
         for(j=0;j<Ny;j++){ /*      loop over k */
            p.k = y[j];
            result = 4.0*M_PI/Norm*int_gsl(intForUHalo, (void*)&p, log(1.e-6), log(2.0*M_PI/KMIN), 1.e-3);
            //result = uHaloClosedFormula(model, p.k, p.Mh, p.c, p.z); // <- exact for truncated NFW profile
            gsl_spline2d_set(spline, za, i, j, result);
         }
      }

      /*    fill in interpolation space */
      gsl_spline2d_init(spline, logx, logy, za, Nx, Ny);

#if 0
      free(x);
      free(logx);
      free(y);
      free(logy);
      free(za);
#endif

   }

   if (inter_xmin < Mh && Mh < inter_xmax && inter_ymin < k && k < inter_ymax){
      return gsl_spline2d_eval(spline, log(Mh), log(k), xacc, yacc);
   }else{
      return 0.0;
   }

}

double intForUHalo(double logr, void *p)
{
   /*    Integrand for uHalo(). */
   const Model *model = ((params *)p)->model;
   const double Mh = ((params *)p)->Mh;
   const double c = ((params *)p)->c;
   const double z = ((params *)p)->z;
   const double k = ((params *)p)->k;

   double r = exp(logr);

   return rhoHalo(model, r, Mh, c, z) * sinc(k*r) * r * r * r;
}


double uHaloClosedFormula(const Model *model, double k, double Mh, double c, double z)
{
   double  f, eta, cieta, sieta, cieta1pc, sieta1pc;

   if(isnan(c)) c = concentration(model, Mh, z, model->concenDef);

   f = 1.0/(log(1.0+c) - c/(1.0+c));
   eta = k * rh(model, Mh, NAN, z)/c;

   cieta = gsl_sf_Ci (eta);
   sieta = gsl_sf_Si (eta);

   cieta1pc = gsl_sf_Ci (eta*(1+c));
   sieta1pc = gsl_sf_Si (eta*(1+c));

   /*    TJ03 (17) */
   return f*(sin(eta)*(sieta1pc - sieta) + cos(eta)*(cieta1pc - cieta)
      - sin(eta*c)/(eta*(1.0+c)));
}


double rhoHalo(const Model *model, double r, double Mh, double c, double z)
{
   /*
    *    Returns a halo profile given a set
    *    of parameters.
    */

   static double r_s, rho_s, Mh_tmp, c_tmp;
   static int firstcall = 1;

   /*
    *    ATTENTION: the halo profile must be truncated otherwise the
    *    normalisation is wrong.
    */
   if(r > rh(model, Mh, NAN, z)){
      return 0.0;
   }

   if (firstcall || assert_float(Mh_tmp, Mh) || assert_float(c_tmp, c)){

      if(isnan(c)) c = concentration(model, Mh, z, model->concenDef);

      r_s = rh(model, Mh, NAN, z)/c;
      rho_s = rho_crit(model, 0.0)*Delta(model, z, model->massDef)/3.0*pow(c,3.0)/(log(1.0+c)-c/(1.0+c));

      Mh_tmp = Mh;
      c_tmp = c;
      firstcall = 0;

   }

   return NFW(r, r_s, rho_s);
}

double NFW(double r, double r_s, double rho_s){
   /*
    *    Returns a NFW profile given r_s and rho_s.
    *    Both r_s and rho_s depend on cosmology and
    *    redshift but are computed by the wrapper.
    */

   double x = r / r_s;
   return rho_s / (x * pow(1.0+x, 2.0));
}

double concentration(const Model *model, double Mh, double z, char *concenDef)
{
   /*
    *    concentration for clusters of mass Mh
    *    at redshift z
    *
    *    Mh in Mpc/h
    *    ATTENTION, each c-M relation is valid
    *    only with the same mass defintion:
    *    c = wrt to critical density
    *    m = wrt to mean density
    */

   double result;
   double a, b, c0, beta, nu;

   if( !strcmp(concenDef, "D11")){

      /*    Duffy et al. (2011), WMAP5 cosmology, M200c, TODO add M200m and Mvir */
      result = 5.71 * pow(Mh/2.0e12, -0.084) * pow(1.0+z, -0.47);

   }else if ( !strcmp(concenDef, "M11")){

      /*    Munoz-Cuartas et al. (2011), Mvir (r_vir from Bryan & Norman, 1998) */
      a = 0.029*z - 0.097;
      b = -110.001/(z+16.885) + 2469.720/pow(z+16.885, 2.0);
      result = pow(10.0, a*log10(Mh) + b);

   }else if ( !strcmp(concenDef, "TJ03")){

      /*
       *    Takada & Jain (2003), concentration for clusters, Mvir, WMAP5 cosmology and
       *    c0, beta parameters used in Coupon et al. (2015)
       */
      c0 = 11.0;
      beta = 0.13;
      result = c0*pow(Mh/2.705760e+12, -beta)/(1.0+z);

   }else if ( !strcmp(concenDef, "B12_F")){

      /*    Bhattacharya et al. (2012) full sample, M200c, concentration for all clusters */
      nu = 1.0/Dplus(model, z) * (1.12 * pow(Mh/5.0e13, 0.3) + 0.53);
      result = pow(Dplus(model, z),1.15) * 9.0*pow(nu, -0.29);

   }else if ( !strcmp(concenDef, "B12_R")){

      /*    Bhattacharya et al. (2012) relaxed sample, M200c, concentration for all clusters */
      nu = 1.0/Dplus(model, z) * (1.12 * pow(Mh/5.0e13, 0.3) + 0.53);
      result = pow(Dplus(model, z), 0.53) * 6.6*pow(nu, -0.41);

   }else if ( !strcmp(concenDef, "B01")){

      /*    Bullock et al. (2001) TODO */
      result = 0.0;

   }else {
      fprintf(stderr, "concentration(): concentration definition \"%s\" is not defined (%s:%d). Exiting...\n", concenDef, __FILE__, __LINE__);
      exit(EXIT_FAILURE);

   }

   return result;

}


/*
 *    mass definition
 */

double Delta(const Model *model, double z, char *massDef)
{
   /*
    *    Set delta wrt critical density depending on
    *    adopted mass definition.
    *    The correction applied is to
    *    take into account the comoving
    *    units
    */

   double result;

   if( !strcmp(massDef, "M200m")){

       result = 200.0 * Omega_m_z(model, 0.0, -1.0);

   }else if( !strcmp(massDef, "M200c")){

      result = 200.0 * E2(model, z, 0)/pow(1.0+z, 3.0);

   }else if( !strcmp(massDef, "M500m")){

      result = 500.0 * Omega_m_z(model, 0.0, -1.0);

   } else if( !strcmp(massDef, "M500c")){

      result = 500.0 * E2(model, z, 0)/pow(1.0+z, 3.0);

   } else if( !strcmp(massDef, "Mvir")){

      result = Delta_vir(model, z) * E2(model, z, 0)/pow(1.0+z, 3.0);

   } else if( !strcmp(massDef, "MvirC15")){

      result = Delta_vir(model, z) * Omega_m_z(model, 0.0, -1.0); // matches Coupon et al. (2015), i.e. Delta_vir defined wrt mean density

   } else {

      fprintf(stderr, "Delta(): mass definition \"%s\" is not defined (%s:%d). Exiting...\n", massDef, __FILE__, __LINE__);
      exit(EXIT_FAILURE);

   }

   return result;
}

double r_vir(const Model *model, double Mh, double c, double z)
{
   /*    Virial radius for a given mass */

   static int firstcall = 1;
   static gsl_interp_accel *acc;
   static gsl_spline *spline;
   static double inter_min, inter_max;

   static double c_tmp = NAN;
   static double z_tmp = NAN;

   if(firstcall || assert_float(c_tmp, c) || assert_float(z_tmp, z)){

      firstcall = 0;
      c_tmp = c;
      z_tmp = z;

      int i, Ninter = 64;
      double dx = (LNMH_MAX-LNMH_MIN)/(double)Ninter;
      double *x = (double *)malloc(Ninter*sizeof(double));
      double *y = (double *)malloc(Ninter*sizeof(double));

      for(i=0;i<Ninter;i++){
         x[i] = LNMH_MIN+dx*(double)i;
         y[i] = pow(3.0*M_vir(model, exp(x[i]), model->massDef, c, z)/(4.0*M_PI*rho_crit(model, 0.0)*Delta_vir(model, z)), 1.0/3.0);
      }

      inter_min = exp(x[0]);
      inter_max = exp(x[Ninter-1]);

      acc = gsl_interp_accel_alloc();
      spline = gsl_spline_alloc (gsl_interp_cspline, Ninter);

      gsl_spline_init(spline, x, y, Ninter);

      free(x);
      free(y);
   }

   if (Mh < inter_min || Mh > inter_max){
      return 0.0;
   }else{
      return gsl_spline_eval(spline, log(Mh), acc);
   }

}


double M_vir(const Model *model, double Mh, char *massDef, double c, double z)
{

   double c2, M2;

   if( !strcmp(massDef, "Mvir")){
      return Mh;
   }

   M1_to_M2(model, Mh, c, Delta(model, z, massDef), Delta_vir(model, z), z, &M2, &c2);

   return M2;
}


void M1_to_M2(const Model *model, double M1, double c1, double Delta1, double Delta2, double z, double *M2, double *c2)
{
   /*
    *    Convert M1 into M2
    *    See Hu & Kravtsov - appendix C
    *    NFW_sum(c1)/Delta1 = NFW_sum(c2)/Delta2
    *    Delta1 and Delta2 are overdensities with respect to
    *    the critical matter density
    */

   if(isnan(c1)) c1 = concentration(model, M1, z, model->concenDef);

   double f, p, x;
   double NFW_sum = (log(1.0+c1) - c1/(1.0+c1))/(c1*c1*c1);

   f = NFW_sum*Delta2/Delta1;
   p = -0.4283-3.13e-3*log(f)-3.52e-5*log(f)*log(f);
   x = pow(0.5116 * pow(f, 2.0*p) + pow(3.0/4.0, 2.0),-1.0/2.0) + 2.0*f;

   *c2 = 1.0/x;
   *M2 = M1*Delta2/Delta1*pow(*c2/c1, 3.0);

   return;

}


#define KS96 0
#define WK03 1

#define TYPE WK03

double Delta_vir(const Model *model, double z)
{

   /*      Virial overdensity */

   double result = 0.0;
   double w_vir, theta, w, a, b;

   int type = TYPE; // TODO: set as a model parameter

   switch (type) {

      case KS96:
         /*    Kitayama & Suto (1996) */

         w_vir = 1.0/Omega_m_z(model, z, -1.0) - 1.0;
         result = 18.0*M_PI*M_PI*(1.0 + 0.4093*pow(w_vir, 0.9052));
         break;

      case WK03:
         /*    Weinberg & Kamionkowsky (2003), used in Coupon et al. (2015) */

         theta = 1.0/Omega_m_z(model, z, -1.0) - 1.0;

         /*    dark energy equation of state */
         w = -1;

         a = 0.399 - 1.309*(pow(fabs(w), 0.426) - 1.0);
         b = 0.941 - 0.205*(pow(fabs(w), 0.938) - 1.0);

         result = 18.0*M_PI*M_PI*(1.0 + a * pow(theta, b));
         break;
   }

   return result;
}

#undef TYPE

#undef KS96
#undef WK03


/*
 *    halo model functions
 */

double f_sigma(const Model *model, double sigma, double z)
{
   /*    f(sigma) */

   double result;
   double a, p, A, b, c, log10_Delta, log10_alpha;

   double x  = Dplus(model, z)*sigma;
   double nu = delta_c(model, z)/x;

   if( !strcmp(model->hmfDef, "PS74")){
      /*    Press & Schechter (1974) */
      p = 0.0; a = 1.0;
      A = 1.0/(1.0 + pow(2.0, -p)*exp(gammln(0.5-p))/sqrt(M_PI));
      result = A * nu * sqrt(2.0*a/M_PI) * (1.0 + pow(a*nu*nu, -p)) * exp(-a*nu*nu/2.0);

   }else if( !strcmp(model->hmfDef, "ST99")){
      /*    Sheth & Tormen (1999) */
      p = 0.3; a = 0.75;
      A = 1.0/(1.0 + pow(2.0, -p)*exp(gammln(0.5-p))/sqrt(M_PI));
      result = A * nu * sqrt(2.0*a/M_PI) * (1.0 + pow(a*nu*nu, -p)) * exp(-a*nu*nu/2.0);

   }else if( !strcmp(model->hmfDef, "ST02")){
      /*    Sheth & Tormen (2002) */
      p = 0.3; a = 1.0/sqrt(2.0);
      A = 1.0/(1.0 + pow(2.0, -p)*exp(gammln(0.5-p))/sqrt(M_PI));
      result = A * nu * sqrt(2.0*a/M_PI) * (1.0 + pow(a*nu*nu, -p)) * exp(-a*nu*nu/2.0);

   } else if( !strcmp(model->hmfDef, "J01")){
      result = 0.315 * exp(-pow(fabs(log(1.0/(x)) + 0.61), 3.8));

   } else if( !strcmp(model->hmfDef, "T08")){
      /*    Tinker et al. (2008) */

      /*
       * In Tinker et al., Delta is defined wrt to critical density
       * hence the normalisation by Omega_m
       */
      log10_Delta = log10(Delta(model, z, model->massDef) / Omega_m_z(model, 0.0, -1.0) );

      A = 0.1*log10_Delta - 0.05;
      a = 1.43 + pow(log10_Delta - 2.30,  1.5);
      b = 1.00 + pow(log10_Delta - 1.60, -1.5);
      c = 1.20 + pow(MAX(log10_Delta - 2.35, 0.0),  1.6);

      // redshift evolution
      log10_alpha = - pow(0.75/(log10_Delta - log10(75.0)), 1.2);
      A *= pow(1.+z, -0.14);
      a *= pow(1.+z, -0.06);
      b *= pow(1.+z, -pow(10.0, log10_alpha));

      result = A * (pow(x/b, -a)+1.0)*exp(-c/(x*x));


   } else {
      fprintf(stderr, "f_sigma(): halo mass definition \"%s\" is not defined (%s:%d). Exiting...\n", model->hmfDef, __FILE__, __LINE__);
      exit(EXIT_FAILURE);
   }

   return result;

}


double b_sigma(const Model *model, double sigma, double z)
{
   /*    bias(sigma) */

   double result;
   double a, p, A, b, c, log_Delta, B, C;

   double dc = delta_c(model, z);
   double x  = Dplus(model, z)*sigma;
   double nu = dc/x;

   static int firstcall = 1;

   if( strcmp(model->hmfDef, model->biasDef) && firstcall){
      fprintf(stderr, "b_sigma(): WARNING halo mass function definition \"%s\" does not match bias definition \"%s\" (%s:%d).\n", model->hmfDef, model->biasDef, __FILE__, __LINE__);
      firstcall = 0;
   }

   if( !strcmp(model->biasDef, "PS74")){
      /*    Press & Schechter (1974) */
      p = 0.0; a = 1.0;
      /*    Cooray and Sheth (2002) Eq. (68) */
      result = 1.0 + (a*nu*nu - 1.0)/(dc) + (2.0*p/dc)/(1.0+pow(a*nu*nu,p));

   }else if( !strcmp(model->biasDef, "ST99")){
      /*    Sheth & Tormen (1999) */
      p = 0.3; a = 0.75;
      /* Cooray and Sheth (2002) Eq. (68) */
      result = 1.0 + (a*nu*nu - 1.0)/(dc) + (2.0*p/dc)/(1.0+pow(a*nu*nu,p));

   }else if( !strcmp(model->biasDef, "ST02")){
      /*    Sheth & Tormen (2002) */
      p = 0.3; a = 1.0/sqrt(2.0);
      /* Cooray and Sheth (2002) Eq. (68) */
      result = 1.0 + (a*nu*nu - 1.0)/(dc) + (2.0*p/dc)/(1.0+pow(a*nu*nu,p));

   }else if( !strcmp(model->biasDef, "J01")){
      /*       Jenkins et al. (2001) */
      fprintf(stderr, "b_sigma(): \"%s\" is not implemented yet (%s:%d). Exiting...\n", model->biasDef, __FILE__, __LINE__);
      exit(EXIT_FAILURE);

   }else if( !strcmp(model->biasDef, "T08")){
      /*    Tinker et al. (2008, 2010) */

      log_Delta = log10(Delta(model, z, model->massDef) / Omega_m_z(model, 0.0, -1.0) );

      A = 1.0+0.24*log_Delta*exp(-pow(4.0/log_Delta,4.0));
      a = 0.44*log_Delta-0.88;
      B = 0.183;
      b = 1.5;
      C = 0.019+0.107*log_Delta+0.19*exp(-pow(4.0/log_Delta,4.0));
      c = 2.4;

      result = 1.0-A*pow(nu, a)/(pow(nu, a)+pow(dc, a))+B*pow(nu, b)+C*pow(nu, c);

   } else {
      fprintf(stderr, "b_sigma(): bias definition \"%s\" is not defined (%s:%d). Exiting...\n", model->biasDef, __FILE__, __LINE__);
      exit(EXIT_FAILURE);
   }

   return result;
}

double dsigmaM_m1_dlnM(const Model *model, double Mh)
{
   /*    Returns dsigma^-1/dlnM. M in M_sol/h */
   double lnMh = log(Mh);

   /*    Numerical derivative */
   double h, a, b;

   h = lnMh / 20.0;
   a = log(pow(sigma2M(model, exp(lnMh+h)),-0.5));
   b = log(pow(sigma2M(model, exp(lnMh-h)),-0.5));

   return (a - b) / (2.0 * h);
}


double sigma2M(const Model *model, double Mh)
{
   double rhobar = rho_bar(model, 0.0);
   double R = pow(3.0*Mh/(4.0*M_PI*rhobar), 1.0/3.0);

   return sigma2R(model, R);
}


double sigma2R(const Model *model, double R)
{

   params p;
   p.model = model;
   p.R = R;

   return 1.0/(2.0*M_PI*M_PI) * int_gsl(int_for_sigma2R, (void*)&p, log(KMIN), log(KMAX), 1.e-5);
}


double int_for_sigma2R(double lnk, void *p)
{

   const Model *model = ((params *)p)->model;
   double R = ((params *)p)->R;

   double x, k, W;

   k  = exp(lnk);

   x = k*R;

   /*    Fourier transform of tophat filter */
   if (x < 1.e-8){
      W = 1.0 - x*x/10.0 + pow(x, 4.0)/280.0;
   }else{
      W = 3.0/(x*x*x) * (sin(x) - x * cos(x) );
   }

   return k*k*k*P_m_lin(model, k, 0.0)*W*W;

}

double rh(const Model *model, double Mh, double D, double z){
   /*
    *    Returns the radius rh enclosing Delta (D) times the CRITICAL
    *    density of the Universe at redshift z. If Delta = Delta_vir,
    *    and Mh virial mass, this is the virial radius.
    *
    *    Attention: rh in comoving coordinates
    */

   if (isnan(D)){
      D = Delta(model, z, model->massDef);
   }

   return pow(3.0*Mh/(4.0*M_PI*rho_crit(model, 0.0)*D), 1.0/3.0);
}

double Mh_rh(const Model *model, double r, double z)
{
  /*
   *     Mass of a halo with radius rh. If Delta = Delta_vir, this
   *     is virial mass. This is NOT the mass integrated within r, Mh(r).
   */

  return (4.0/3.0)*M_PI*r*r*r*rho_crit(model, 0.0)*Delta(model, z, model->massDef);
}

/*
 *    cosmology functions
 */


double cm3toMpc3_como(const Model *model, double z)
{
   /*    cm3 to Mpc3 conversion function. Returns
    *    result in h^-3 comoving Mpc. */

   static double result = 0.0, z_tmp = -1, H0_tmp = -1;

   if(assert_float(z_tmp, z) || assert_float(H0_tmp, model->H0)){
      z_tmp = z;
      H0_tmp = model->H0;
      result = pow(model->H0/100.0, 3.0) * pow(1.0+z, 3.0) * CM3TOMPC3;
   }

   return result;
}




double delta_c(const Model *model, double z)
{
   /*    Critical collapse overdensity */

   double delta_EdS, alpha;

   cosmo *cosmo = initCosmo(model);

   delta_EdS = 1.68647;

   /*    WK03 (18) */
   alpha = 0.131 + cosmo->w0_de*(0.555 + cosmo->w0_de*(1.128 + cosmo->w0_de*(1.044 + cosmo->w0_de*0.353)));

   /*    KS96 (A.6). Note: typo (0.123) in astro-ph version. Correct in ApJ. (alpha = 0.0123)*/

   return delta_EdS*(1. + alpha*log10(Omega_m_z(model, z, -1.0)));

}


double Dplus(const Model *model, double z)
{
   /*    wrapper for D+ function */

   static double Dp = 0.0, z_tmp = -1.0;

   if(fabs(z-z_tmp) > 1.e-5){

      z_tmp = z;

      cosmo *cosmo = initCosmo(model);

      error *myerr = NULL, **err;
      err = &myerr;

      Dp = D_plus(cosmo, 1./(1.+z), 1, err);
      quitOnError(*err, __LINE__, stderr);

   }

   return Dp;

}

double P_m_nonlin(const Model *model, double k, double z)
{
   /*
    *    Wrapper for the non-linear power spectrum
    *    at redshift z from the NICAEA library.
    *    Input k must be in Mpc and outputs P(k) in Mpc-3 .
    */

   static int firstcall = 1;
   static gsl_interp_accel *acc;
   static gsl_spline *spline;
   static double inter_min, inter_max;

   if(firstcall){

      firstcall = 0;

      cosmo *cosmo = initCosmo(model);

      error *myerr = NULL, **err;
      err = &myerr;

      int i, Ninter  = 256;
      double dlnk    = log(KMAX/KMIN)/(double)Ninter;
      double *lnk    = (double *)malloc(Ninter*sizeof(double));
      double *logP   = (double *)malloc(Ninter*sizeof(double));

      for(i=0;i<Ninter;i++){
         lnk[i]  = log(KMIN)+dlnk*(double)i;
         logP[i] = log(P_NL(cosmo, 1.0/(1.0+z), exp(lnk[i]), err));
         quitOnError(*err, __LINE__, stderr);
      }

      inter_min = exp(lnk[0]);
      inter_max = exp(lnk[Ninter-1]);

      acc = gsl_interp_accel_alloc();
      spline    = gsl_spline_alloc (gsl_interp_cspline, Ninter);
      gsl_spline_init(spline, lnk, logP, Ninter);

      free(lnk);
      free(logP);

   }

   if (k < inter_min || k > inter_max){
      return 0.0;
   }else{
      return exp(gsl_spline_eval(spline, log(k), acc));
   }

}


double P_m_lin(const Model *model, double k, double z)
{
   /*
    *    Wrapper for the linear power spectrum
    *    at redshift z from the NICAEA library.
    *    Input k must be in Mpc and outputs P(k) in Mpc-3 .
    */

   static int firstcall = 1;
   static gsl_interp_accel *acc;
   static gsl_spline *spline;
   static double inter_min, inter_max;

   if(firstcall){

      firstcall = 0;

      cosmo *cosmo = initCosmo(model);

      error *myerr = NULL, **err;
      err = &myerr;

      int i, Ninter  = 256;
      double dlnk    = log(KMAX/KMIN)/(double)Ninter;
      double *lnk    = (double *)malloc(Ninter*sizeof(double));
      double *logP   = (double *)malloc(Ninter*sizeof(double));

      for(i=0;i<Ninter;i++){
         lnk[i]  = log(KMIN)+dlnk*(double)i;
         logP[i] = log(P_L(cosmo, 1.0/(1.0+z), exp(lnk[i]), err));
         quitOnError(*err, __LINE__, stderr);
      }

      inter_min = exp(lnk[0]);
      inter_max = exp(lnk[Ninter-1]);

      acc = gsl_interp_accel_alloc();
      spline    = gsl_spline_alloc (gsl_interp_cspline, Ninter);
      gsl_spline_init(spline, lnk, logP, Ninter);

      free(lnk);
      free(logP);

   }

   if (k < inter_min || k > inter_max){
      return 0.0;
   }else{
      return exp(gsl_spline_eval(spline, log(k), acc));
   }

}

double E2(const Model *model, double z,  int wOmegar){
   /*    wrapper for E^2 */
   double result;

   cosmo *cosmo = initCosmo(model);

   error *myerr = NULL, **err;
   err = &myerr;

   result = Esqr(cosmo, 1.0/(1.0+z), wOmegar, err);
   quitOnError(*err, __LINE__, stderr);

   return result;

}

/*    Per05 (6) */
double Omega_m_z(const Model *model, double z, double E2pre)
{
   /*    returns Omega_m is z has changed */

   static double Om = 0.0, z_tmp = -1.0;

   if(fabs(z-z_tmp) > 1.e-5){

      z_tmp = z;

      cosmo *cosmo = initCosmo(model);

      if (E2pre>0){
         Om = cosmo->Omega_m*pow(1.+z, 3.0)/E2pre;
      } else {
         Om = cosmo->Omega_m*pow(1.+z, 3.0)/E2(model, z, 0);
      }

   }

   return Om;
}


double rho_crit(const Model *model, double z){
   /*    returns rho_crit at redshift z in [M_sol h^2 / Mpc^3] */
   /*    present critical density is  rho_c0 = 2.7754e11 */

   static double H, H2, G, rc = 0.0, z_tmp = -1.0;

   if(fabs(z-z_tmp) > 1.e-5){

      z_tmp = z;
      // cosmo *cosmo = initCosmo(model);
      // double H = cosmo->h_100 * 100.0;
      H = 100.0; // <- Hubble units

      H2 = E2(model, z, 0) * pow(H, 2.0); /* H(z) in h km s^-1 Mpc^-1 */
      G  = 4.302e-9;                      /* in  [km^2 s^-2 M_sun^-1 Mpc^1]  */

      rc = 3.0*H2/(8.0*pi*G);

   }

   return rc;
}


double rho_bar(const Model *model, double z)
{
   return rho_crit(model, z) * Omega_m_z(model, z, -1.0);
}

double dr_dz(const Model *model, double z)
{
   /*
    *    wrapper for dr/dz, r=comoving distance, see Hamana et al. (2004), eq. (10)
    */

   double a, result;

   error *myerr = NULL, **err;
   err = &myerr;

   cosmo *cosmo = initCosmo(model);

   a       = 1.0/(1.0 + z);
   result = drdz(cosmo, a, err);
   quitOnError(*err, __LINE__, stderr);

   return result;

}


double lookbackTimeInv(const Model *model, double tL){
   /*
    *    Returns the redshift given a lookback time t
    *    in Gyr (z in between 0 and 10000).
    */

   int i, N = 256;
   double log10OnePlusZmin = 0.0, log10OnePlusZmax = 4.0; // x = log10(1+z))

   static int firstcall = 1;
   static gsl_interp_accel *acc;
   static gsl_spline *spline;

   static double *t_log10OnePlusZ;
   static double *t_tL;
   static double dlog10OnePlusZ;

   if (firstcall) {
      firstcall = 0;

      /*    tabulate t = f(log10z) */
      t_log10OnePlusZ = (double *)malloc(N*sizeof(double));
      t_tL = (double *)malloc(N*sizeof(double));
      dlog10OnePlusZ = (log10OnePlusZmax - log10OnePlusZmin)/(double)N;

      for(i=0;i<N;i++){
         t_log10OnePlusZ[i] = log10OnePlusZmin + dlog10OnePlusZ*(double)i;
         t_tL[i] = lookbackTime(model, pow(10.0, t_log10OnePlusZ[i])-1.0);
      }

      acc = gsl_interp_accel_alloc();
      spline = gsl_spline_alloc (gsl_interp_cspline, N);

      gsl_spline_init(spline, t_tL, t_log10OnePlusZ, N);

   }

   if (t_tL[0] < tL && tL < t_tL[N-1]){
      return pow(10.0, gsl_spline_eval(spline, tL, acc))-1.0;
   }else{
      return 1.0/0.0;
   }

#if 0
   free(t_log10Mh);
   free(t_log10Mstar);
   gsl_spline_free (spline);
   gsl_interp_accel_free (acc);
#endif

}


double lookbackTime(const Model *model, double z)
{
   /*
    *    lookback time in Gyr
    */

   double result;

   params p;
   p.model = model;

   result = 9784619482.496195 / model->h  * int_gsl(intForLookbackTime, (void*)&p, 0.0, z, 1.e-3) / 1.e9;

   return result;
}

double intForLookbackTime(double zp, void *p)
{
   const Model *model = ((params *)p)->model;
   return 1.0/((1.0+zp)*sqrt(E2(model, zp,  0)));
}

/*
 *    From NICAEA doc:
 *
 *    If wOmegar=1, Omega_radiation>0 is included (photons +
 *    neutrinos), needed for high-z quantities such as the sound
 *    horizon at the drag epoch. Note: For low redshift, wOmega=0
 *    should be used, otherwise the nonlinear power-spectrum
 *    fitting formulae might not work.
 */

double DC(const Model *model, double z, int wOmegar)
{
   /*
    *    wrapper for comoving distance in Mpc/h
    */

   double a, result;

   error *myerr = NULL, **err;
   err = &myerr;

   cosmo *cosmo = initCosmo(model);

   a = 1.0/(1.0 + z);
   result = w(cosmo, a, wOmegar, err);
   quitOnError(*err, __LINE__, stderr);

   return result;

}


double DM(const Model *model, double z, int wOmegar)
{
   /*
    *    wrapper for radial (transverse) comoving distance in Mpc/h
    */

   double a, ww, result;

   error *myerr = NULL, **err;
   err = &myerr;

   cosmo *cosmo = initCosmo(model);

   a = 1.0/(1.0 + z);
   ww = w(cosmo, a, wOmegar, err);
   quitOnError(*err, __LINE__, stderr);
   result = f_K(cosmo, ww, err);
   quitOnError(*err, __LINE__, stderr);

   return result;
}


double DA(const Model *model, double z, int wOmegar)
{
   /*
    *    Angular diameter distance in Mpc/h
    */

   return DM(model, z, wOmegar)/(1.0+z);

}


double DL(const Model *model, double z, int wOmegar)
{
   /*
    *    Luminosity distance in Mpc/h
    */

   return DM(model, z, wOmegar)*(1.0+z);

}
