#!/usr/bin/env python

"""
Jean coupon - 2016
script to run wrapped halomodel routines in c

Required librairies:

for c (if not in /usr/local/, set path in Makefile):
- nicaea 2.5 (http://www.cosmostat.org/software/nicaea/)
- fftw3 3.3.4 (http://www.fftw.org/)
- gsl 2.1 (https://www.gnu.org/software/gsl/)

for python:
- numpy 1.10.2 (http://www.numpy.org/)
- scipy 0.17.1 (https://www.scipy.org/scipylib/download.html)

"""

import os
import numpy as np
import ctypes
import sys


"""

-------------------------------------------------------------
path to c library (absolute path)
-------------------------------------------------------------

"""

# c_halomodel = ctypes.cdll.LoadLibrary(os.path.dirname(os.path.realpath(__file__))+"/lib/libhalomodel.so")
c_halomodel = ctypes.cdll.LoadLibrary("/Users/coupon/local/source/GitHub/halomodel/lib/libhalomodel.so")

"""

-------------------------------------------------------------
classes
-------------------------------------------------------------

"""

class Model(ctypes.Structure):
    """ Structure to serve at interface between
    python wrapper and c routines.

    IMPORTANT: when adding a new field,
    add corresponding field IN SAME ORDER
    in include/utils.h
    """

    _fields_ = [

        # cosmology
        ("Omega_m", ctypes.c_double),
        ("Omega_de", ctypes.c_double),
        ("H0", ctypes.c_double),
        # ("como", ctypes.c_int), # TODO-> change strategy leave everything in comoving units only convert DATA
        ("massDef", ctypes.c_char_p),
        ("concenDef", ctypes.c_char_p),
        ("hmfDef", ctypes.c_char_p),
        ("biasDef", ctypes.c_char_p),

         # halo model / HOD parameters
        ("log10M1", ctypes.c_double),
        ("log10Mstar0", ctypes.c_double),
        ("beta", ctypes.c_double),
        ("delta", ctypes.c_double),
        ("gamma", ctypes.c_double),
        ("log10Mstar_min", ctypes.c_double),
        ("log10Mstar_max", ctypes.c_double),
        ("sigma_log_M0", ctypes.c_double),
        ("sigma_lambda", ctypes.c_double),
        ("B_cut", ctypes.c_double),
        ("B_sat", ctypes.c_double),
        ("beta_cut", ctypes.c_double),
        ("beta_sat", ctypes.c_double),
        ("alpha", ctypes.c_double),
        ("fcen1", ctypes.c_double),
        ("fcen2", ctypes.c_double),

        # if using hod model
        ("hod", ctypes.c_int),

        # for X-ray binaries
        ("Ix_XB_Re", ctypes.c_double),
        ("Ix_XB_L", ctypes.c_double),

        # X-ray, if hod = 0
        ("gas_log10n0", ctypes.c_double),
        ("gas_log10beta", ctypes.c_double),
        ("gas_log10rc", ctypes.c_double),

        # X-ray, if hod = 1
        ("gas_log10n0_1", ctypes.c_double),
        ("gas_log10n0_2", ctypes.c_double),
        ("gas_log10n0_3", ctypes.c_double),
        ("gas_log10n0_4", ctypes.c_double),
        ("gas_log10beta_1", ctypes.c_double),
        ("gas_log10beta_2", ctypes.c_double),
        ("gas_log10beta_3", ctypes.c_double),
        ("gas_log10beta_4", ctypes.c_double),
        ("gas_log10rc_1", ctypes.c_double),
        ("gas_log10rc_2", ctypes.c_double),
        ("gas_log10rc_3", ctypes.c_double),
        ("gas_log10rc_4", ctypes.c_double),

        # for gg lensing, if hod = 0
        ("ggl_pi_max", ctypes.c_double),
        ("ggl_log10c", ctypes.c_double),
        ("ggl_log10Mh", ctypes.c_double),
        ("ggl_log10Mstar", ctypes.c_double),

        # for wtheta, if hod = 1
        ("wtheta_nz_N", ctypes.c_int),
        ("wtheta_nz_z", ctypes.POINTER(ctypes.c_double)),
        ("wtheta_nz", ctypes.POINTER(ctypes.c_double)),

        # XMM PSF, King function parameters
        ("XMM_PSF_A", ctypes.c_double),
        ("XMM_PSF_rc", ctypes.c_double),
        ("XMM_PSF_alpha", ctypes.c_double),
        ]

    # default parameters
    def __init__(self, Omega_m=0.258, Omega_de=0.742, H0=72.0, hod=0, massDef="M500c", concenDef="TJ03", hmfDef="T08", biasDef="T08"):

        # cosmology
        self.Omega_m = Omega_m
        self.Omega_de = Omega_de
        self.H0 = H0
        # self.como = 1          # comoving coordinates (1) or physical (0)
        self.massDef = massDef    # halo mass definition: M500c, M500m, M200c, M200m, Mvir, MvirC15
        self.concenDef = concenDef  # mass/concentration relation: D11, M11, TJ03, B12_F, B12_R, B01
        self.hmfDef = hmfDef     # halo mass defintion: PS74, ST99, ST02, J01, T08
        self.biasDef = biasDef    # mass/bias relation:  PS74, ST99, ST02, J01, T08

        # halo model / HOD parameters
        self.log10M1 = 12.529 # in Msun h^-1
        self.log10Mstar0 = 10.748 # in Msun h^-1
        self.beta = 0.356
        self.delta = 0.749
        self.gamma = 0.807
        self.log10Mstar_min = 10.00
        self.log10Mstar_max = 11.00
        self.sigma_log_M0 = 0.394
        self.sigma_lambda = 0.253
        self.B_cut = -1
        self.B_sat = 9.956
        self.beta_cut = -1
        self.beta_sat = 0.872
        self.alpha = 1.139
        self.fcen1 = -1
        self.fcen2 = -1

        # if using hod model
        self.hod = hod

        # for X-ray binaries
        self.Ix_XB_Re = -1.0 # in Mpc
        self.Ix_XB_L = -1.0 # in CR Mpc-2

        # X-ray, if hod = 0
        self.gas_log10n0 = -3.0
        self.gas_log10beta = -1.0
        self.gas_log10rc = -1.0

        # X-ray, if hod = 1
        self.gas_log10n0_1 = -2.11726021929 # log10n0 = gas_log10n0_1  + gas_log10n0_2 * (log10Mh-14.0)
        self.gas_log10n0_2 = -0.29693164    # n0 in [h^3 Mpc^-3], Mpc in comoving coordinate.
        self.gas_log10n0_3 = np.nan
        self.gas_log10n0_4 = np.nan
        self.gas_log10beta_1 = -0.32104805 # log10beta = gas_log10beta_1  + gas_log10beta_2 * (log10Mh-14.0)
        self.gas_log10beta_2 = +0.26463453
        self.gas_log10beta_3 = np.nan
        self.gas_log10beta_4 = np.nan
        self.gas_log10rc_1 = -1.12356845357 # log10beta = gas_log10rc_1  + gas_log10rc_2 * (log10Mh-14.0)
        self.gas_log10rc_2 = +0.73917722    # rc in [h^-1 Mpc], Mpc in comoving coordinate.
        self.gas_log10rc_3 = np.nan
        self.gas_log10rc_4 = np.nan

        # for gg lensing
        self.ggl_pi_max = 60.0
        self.ggl_log10c = np.nan
        self.ggl_log10Mh = 14.0
        self.ggl_log10Mstar = np.nan

        # n(z) for wtheta, if hod = 1
        self.wtheta_nz_N = 0
        self.wtheta_nz_z = None
        self.wtheta_nz = None

        # XMM PSF
        self.XMM_PSF_A = np.nan
        self.XMM_PSF_rc = np.nan
        self.XMM_PSF_alpha = np.nan

""" c function prototypes """

c_halomodel.dndlog10Mstar.argtypes = [ctypes.POINTER(Model), np.ctypeslib.ndpointer(dtype = np.float64), ctypes.c_int, ctypes.c_double, ctypes.c_int, np.ctypeslib.ndpointer(dtype = np.float64)]
c_halomodel.DeltaSigma.argtypes = [ctypes.POINTER(Model), np.ctypeslib.ndpointer(dtype = np.float64), ctypes.c_int, ctypes.c_double, ctypes.c_int, np.ctypeslib.ndpointer(dtype = np.float64)]
c_halomodel.wOfTheta.argtypes = [ctypes.POINTER(Model), np.ctypeslib.ndpointer(dtype = np.float64), ctypes.c_int, ctypes.c_double, ctypes.c_int, np.ctypeslib.ndpointer(dtype = np.float64)]
c_halomodel.SigmaIx.argtypes = [ctypes.POINTER(Model), np.ctypeslib.ndpointer(dtype = np.float64), ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, np.ctypeslib.ndpointer(dtype = np.float64)]
c_halomodel.rh.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double, ctypes.c_double]
c_halomodel.rh.restype = ctypes.c_double
c_halomodel.bias_h.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double]
c_halomodel.bias_h.restype = ctypes.c_double
c_halomodel.concentration.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double, ctypes.c_char_p]
c_halomodel.concentration.restype = ctypes.c_double
c_halomodel.M1_to_M2.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
c_halomodel.Omega_m_z.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double]
c_halomodel.Omega_m_z.restype = ctypes.c_double
c_halomodel.Delta.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_char_p]
c_halomodel.Delta.restype = ctypes.c_double
c_halomodel.Delta_vir.argtypes = [ctypes.POINTER(Model), ctypes.c_double]
c_halomodel.Delta_vir.restype = ctypes.c_double
c_halomodel.msmh_log10Mh.argtypes = [ctypes.POINTER(Model), ctypes.c_double]
c_halomodel.msmh_log10Mh.restype = ctypes.c_double
c_halomodel.inter_gas_log10n0.argtypes = [ctypes.POINTER(Model), ctypes.c_double]
c_halomodel.inter_gas_log10n0.restype = ctypes.c_double
c_halomodel.inter_gas_log10beta.argtypes = [ctypes.POINTER(Model), ctypes.c_double]
c_halomodel.inter_gas_log10beta.restype = ctypes.c_double
c_halomodel.inter_gas_log10rc.argtypes = [ctypes.POINTER(Model), ctypes.c_double]
c_halomodel.inter_gas_log10rc.restype = ctypes.c_double
c_halomodel.DA.argtypes =  [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_int]
c_halomodel.DA.restype  = ctypes.c_double


"""

-------------------------------------------------------------
main
-------------------------------------------------------------

"""


def main(args):

    function = getattr(sys.modules[__name__], args.option)(args)
    return

"""

-------------------------------------------------------------
test
-------------------------------------------------------------

"""


def test(args):
    """ tests """

    actions = ["dist"]

    # this model matches Coupon et al. (2015)
    model = Model(Omega_m=0.258, Omega_de=0.742, H0=72.0, hod=1, massDef="MvirC15", concenDef="TJ03", hmfDef="ST02", biasDef="T08")
    z = 0.308898


    if "dist" in actions:
        """ angular diameter distance """

        from   astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=model.H0, Om0=model.Omega_m)
        h = model.H0/100.0

        print c_halomodel.DA(model, z, 0)/h, cosmo.angular_diameter_distance([z])









    if "smf" in actions:
        """ stellar mass function """
        model.hod = 1

        log10Mstar  = np.linspace(np.log10(1.e9), np.log10(1.e12), 100.00)
        result      = dndlog10Mstar(model, log10Mstar, z, obs_type="all")

        for i in range(len(log10Mstar)):
            print log10Mstar[i], result[i]

    if "ggl" in actions:

        model.como = 0 # physical or comoving coordinates
        model.hod = 0 # halo model

        if model.hod == 1:
            model.log10Mstar_min = 11.10 - 0.1549 #- 0.142668
            model.log10Mstar_max = 11.30 - 0.1549 #- 0.142668
        else:
            model.hod = 0
            model.ggl_log10Mh = 12.0520432788 - 0.142668
            model.ggl_log10c  = 1.12299192362

        R   = pow(10.0, np.linspace(np.log10(1.e-3), np.log10(1.e2), 100))

        star    = DeltaSigma(model, R, z, obs_type="star")
        cen     = DeltaSigma(model, R, z, obs_type="cen")
        sat     = DeltaSigma(model, R, z, obs_type="sat")
        twohalo = DeltaSigma(model, R, z, obs_type="twohalo")
        total   = DeltaSigma(model, R, z, obs_type="all")

        for i in range(len(R)):
            print R[i], total[i], star[i], cen[i], sat[i], twohalo[i]

    if "wtheta" in actions:

        from   astropy.io import ascii

        nz = ascii.read("/Users/coupon/data/XXL/HOD/K22_0.20_0.35_Mstar_11.10_11.30_all.nz", format="no_header")

        # n(z)
        model.wtheta_nz_N = len(nz)
        model.wtheta_nz_z = np.ctypeslib.as_ctypes(nz["col1"])
        model.wtheta_nz   = np.ctypeslib.as_ctypes(nz["col2"])

        if True: # parameteric HOD
            model.log10Mstar_min = 11.10 - 0.1549 #- 0.142668
            model.log10Mstar_max = 11.30 - 0.1549 #- 0.142668

        R   = pow(10.0, np.linspace(np.log10(1.e-3), np.log10(1.e2), 100.00))

        censat  = wOfTheta(model, R, z, obs_type="censat")
        satsat  = wOfTheta(model, R, z, obs_type="satsat")
        twohalo = wOfTheta(model, R, z, obs_type="twohalo")
        total   = wOfTheta(model, R, z, obs_type="all")

        for i in range(len(R)):
            print R[i], total[i], censat[i], satsat[i], twohalo[i]

    if "Lambda" in actions:

        c_halomodel.Lambda.argtypes = [ctypes.c_double, ctypes.c_double]
        c_halomodel.Lambda.restype = ctypes.c_double

        Tx = pow(10.0, np.linspace(np.log10(1.01e-1), np.log10(1.e1), 100))
        for i in range(len(Tx)):
            print Tx[i], c_halomodel.Lambda(Tx[i], 0.25), c_halomodel.Lambda(Tx[i], 0.15), c_halomodel.Lambda(Tx[i], 0.40)


    if "CRToLx" in actions:

        c_halomodel.CRToLx.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double, ctypes.c_double]
        c_halomodel.CRToLx.restype = ctypes.c_double

        Tx = 0.2
        res = c_halomodel.CRToLx(model, z, Tx, 0.25)
        print Tx, res

        # dirname = os.path.dirname(os.path.realpath(__file__))
        # sys.path.append(dirname+"/python")
        # import xray
        # print xray.CRToLx(dirname+"/data/CRtoLx.ascii", z, 0.25)

    def powerLaw(x, a, b):
        return a + b*(x-14.0)

    if "Ix" in actions:

        from astropy.io import ascii
        from scipy.optimize import curve_fit

        c_halomodel.MhToTx.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double]
        c_halomodel.MhToTx.restype = ctypes.c_double

        c_halomodel.TxToMh.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double]
        c_halomodel.TxToMh.restype = ctypes.c_double

        if False:
            data = ascii.read("/Users/coupon/projects/Stacked_X_ray/info/betaProfilesEckert2015.ascii", header_start=-1)

            x = [ np.log10(c_halomodel.TxToMh(model, Tx, z)) for Tx in data['Tx'] ]

            print x

            R500 = [c_halomodel.rh(model, pow(10.0, log10Mh), np.nan, z) for log10Mh in x]

            data["rc"] = data["RcOverR500"]*R500
            data["rc_err"] = data["RcOverR500_err"]*R500

            para = 'rc'
            y    = [ np.log10(p) for p in data[para] ]
            yerr = [ p_err/p / np.log(10.0) for (p, p_err) in zip(data[para],data[para+'_err']) ]

            p, pCov = curve_fit(powerLaw,x, y, sigma=yerr )
            print p

            return

        model.como = 0 # physical (0) or comoving coordinates (1)
        model.hod = 0 # halo model

        if model.hod == 1:

            Mh = np.nan
            c  = np.nan #pow(10.0, 0.797955197032)

            model.log10Mstar_min = 11.10 - 0.1549 #- 0.142668
            model.log10Mstar_max = 11.30 - 0.1549 #- 0.142668

            # X-ray, if hod = 1
            model.gas_log10n0_1 = -2.54526273
            model.gas_log10n0_2 = -0.29693164
            model.gas_log10beta_1 = -0.32104805
            model.gas_log10beta_2 =  0.26463453
            model.gas_log10rc_1  = -0.98090095
            model.gas_log10rc_2  = 0.73917722

        else:

            Mh = pow(10.0, 13.8)
            c  = np.nan #pow(10.0, 0.797955197032)

            R500 = c_halomodel.rh(model, Mh, Delta(model, z, "M500c"), z)

            model.gas_log10n0 = np.log10(5.3e-3)
            model.gas_log10beta = np.log10(0.40)
            model.gas_log10rc = np.log10(0.03*R500)

        R   = pow(10.0, np.linspace(np.log10(2.e-3), np.log10(1.e1), 100))

        # PSF=[12.270080, 0.033294, 1.851542]
        PSF=None

        cen = SigmaIx(model, R, Mh, c, z, obs_type="cen", PSF=PSF)

        for i in range(len(R)):
            print R[i], cen[i]


    if "MsMh" in actions:

        c_halomodel.msmh_log10Mstar.argtypes = [ctypes.POINTER(Model), ctypes.c_double]
        c_halomodel.msmh_log10Mstar.restype  = ctypes.c_double

        log10Mh  = np.linspace(np.log10(1.e10), np.log10(1.e15), 100.00)

        for i in range(len(log10Mh)):
            print log10Mh[i], c_halomodel.msmh_log10Mstar(model, log10Mh[i])

    if "uNFW" in actions:

        Mh = 1.e14
        c  = np.nan

        c_halomodel.uHalo.argtypes =  [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        c_halomodel.uHalo.restype  = ctypes.c_double

        c_halomodel.uHaloClosedFormula.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        c_halomodel.uHaloClosedFormula.restype  = ctypes.c_double

        k       = pow(10.0, np.linspace(np.log10(2.e-3), np.log10(1.e4), 100))
        numerical = np.asarray(np.zeros(len(k)), dtype=np.float64)
        analytic  = np.asarray(np.zeros(len(k)), dtype=np.float64)

        for i in range(len(k)):
            numerical[i] = c_halomodel.uHalo(model, k[i], Mh, c, z)
            analytic[i]  = c_halomodel.uHaloClosedFormula(model, k[i], Mh, c, z)
            print k[i], numerical[i], analytic[i]

        #print np.trapz(numerical, k), np.trapz(analytic, k)

    if "xi_dm" in actions:

        c_halomodel.xi_m.argtypes =  [ctypes.POINTER(Model), np.ctypeslib.ndpointer(dtype = np.float64), ctypes.c_int, ctypes.c_double,  np.ctypeslib.ndpointer(dtype = np.float64)]

        R       = pow(10.0, np.linspace(np.log10(2.e-3), np.log10(200.0), 100.00))
        result  = np.asarray(np.zeros(len(R)), dtype=np.float64)

        c_halomodel.xi_m(model, R, len(R), z, result)

        for i in range(len(R)):
            print R[i], result[i]

    if "concen" in actions:
        print concentration(model, 1.e14, z, concenDef="TJ03")

    if "mass_conv" in actions:

        log10h = np.log10(model.H0/100.0)

        log10Mstar = 10.0 # in Msun
        log10Mh    = msmh_log10Mh(model, log10Mstar+log10h)

        print log10M1_to_log10M2(model, log10Mh, None, "MvirC15", "M500c", z)[0] - log10h

    if "change" in actions:

        c_halomodel.changeModelHOD.argtypes = [ctypes.POINTER(Model)]
        c_halomodel.changeModelHOD.restype  = ctypes.c_int

        print c_halomodel.changeModelHOD(model)
        print c_halomodel.changeModelHOD(model)


"""

-------------------------------------------------------------
main functions
-------------------------------------------------------------

"""



def dndlog10Mstar(model, log10Mstar, z, obs_type="all"):
    """
    Wrapper for c-function dndlog10Mstar()

    Returns the stellar mass function in units of (Mpc/h)^-3 dex^-1

    ATTENTION: Mpc in comoving units

    INPUT PARAMETERS:
    log10Mstar: log10(Mstar) array or single value) in log10 Msun/h units
    z: redshift of the sample

    obs_type: [cen, sat, all]

    """

    log10Mstar = np.asarray(log10Mstar, dtype=np.float64)
    result     = np.asarray(np.zeros(len(log10Mstar)), dtype=np.float64)

    # log10Mstar = np.asarray(log10Mstar, dtype=np.float64)
    # result     = np.asarray(np.zeros(len(log10Mstar)), dtype=np.float64)

    if obs_type == "cen":  obs_type = 1
    if obs_type == "sat":  obs_type = 2
    if obs_type == "all":  obs_type = 3

    c_halomodel.dndlog10Mstar(model, log10Mstar, len(log10Mstar), z, obs_type, result)

    return result

def wOfTheta(model, R, zl, obs_type="all"):
    """
    Returns w(theta) for NFW halo mass profile (in h Msun/pc^2) -

    INPUT PARAMETERS:
    R: (array or single value) in physical units (Mpc/h)
    z: mean redshift of the population

    obs_type: [censat, satsat, twohalo, all]
    """


    R      = np.asarray(R, dtype=np.float64)
    result = np.asarray(np.zeros(len(R)), dtype=np.float64)

    if obs_type == "censat":      obs_type = 12
    if obs_type == "satsat":      obs_type = 22
    if obs_type == "twohalo":     obs_type = 33
    if obs_type == "all":         obs_type = 3

    c_halomodel.wOfTheta(model, R, len(R), zl, obs_type, result)

    return result

def DeltaSigma(model, R, zl, obs_type="all"):
    """
    Returns DeltaSigma for NFW halo mass profile (in h Msun/pc^2) -
    PHYSICAL UNITS, unless como=True set

    INPUT PARAMETERS:
    R: (array or single value) in physical units (Mpc/h)
    zl: redshift of the lens

    obs_type: [cen, sat, twohalo, star, all], for "star", set log10Mstar if
    HOD model is not set

    If M given defined wrt the *mean* density
    then give Delta_m X Omega_m(z) as Delta

    REFS: Sec. 7.5.1 of Mo, van den Bosh & White's Galaxy
    formation and evolution Wright & Brainerd (200) after
    correcting the typo in Eq. (7.141)
    """

    R      = np.asarray(R, dtype=np.float64)
    result = np.asarray(np.zeros(len(R)), dtype=np.float64)

    if obs_type == "star":       obs_type = 0
    if obs_type == "cen":        obs_type = 1
    if obs_type == "sat":        obs_type = 2
    if obs_type == "twohalo":    obs_type = 33
    if obs_type == "all":        obs_type = 3

    c_halomodel.DeltaSigma(model, R, len(R), zl, obs_type, result)

    return result

def SigmaIx(model, R, Mh, c, z, obs_type="all", PSF=None):
    """
    Wrapper for c-function SigmaIx()

    Returns the X-ray luminosity profile

    INPUT PARAMETERS:
    R: (array or single value) in physical units (Mpc)
    Mh: mass of the host halo (not used if HOD set)
    c: concentration of the host halo (not used if HOD set)
    z: redshift
    obs_type: [cen, sat]
    """

    R = np.asarray(R, dtype=np.float64)
    result = np.asarray(np.zeros(len(R)), dtype=np.float64)

    # satellite term is a convolution
    # of the profile with itself
    if obs_type == "cen":  obs_type = 1
    if obs_type == "sat":  obs_type = 2
    if obs_type == "XB":   obs_type = 3
    # TODO: add all

    # PSF in model
    if PSF is not None:
        model.XMM_PSF_A     = PSF[0]
        model.XMM_PSF_rc    = PSF[1]
        model.XMM_PSF_alpha = PSF[2]

    c_halomodel.SigmaIx(model, R, len(R), Mh, c, z, obs_type, result)

    return result

"""

-------------------------------------------------------------
Utils
-------------------------------------------------------------

"""

def getCRtoLx_bremss(fileNameIn, redshift):
    from scipy import interpolate
    from   astropy.io import ascii

    # CR to Lx_bolo conversion file:
    data = ascii.read(fileNameIn, format="commented_header", header_start=-1)
    x    = data['z']
    y    = np.log10(data['Lx_bolo']/data['CR'])

    return pow(10.0, interpolate.griddata(x, y, redshift, method='linear'))


def DA(model, z):
    """ Returns the angular diamter distance
    Assumes OmegaR = 0.0
    """

    return c_halomodel.DA(model, z, 0)


def rh(model, Mh, z):
    """ Returns rh in h^-1 Mpc """
    return c_halomodel.rh(model, Mh, z)

def bias_h(model, Mh, z):
    """ Returns halo bias """
    return c_halomodel.bias_h(model, Mh, z)

def concentration(model, Mh, z, concenDef="TJ03"):
    """ Returns the concentration """
    return c_halomodel.concentration(model, Mh, z, concenDef)


def Delta(model, z, massDef):
    """ Returns Delta according to mass definition.
    Matches Delta() in c_halomodel

    Delta defined wrt critical density
    """

    return c_halomodel.Delta(model, z, massDef)


def msmh_log10Mh(model, log10Mstar):
    """
    Mstar - Mh relation.
    Parameterization from
    Behroozi et al. (2010).
    """

    return c_halomodel.msmh_log10Mh(model, log10Mstar)

def M1_to_M2(model, M1, c1, Delta1, Delta2, z):

    Delta1 = Delta(model, z, Delta1)
    Delta2 = Delta(model, z, Delta2)

    if c1 is None:
        c1 = np.nan

    M2 = ctypes.c_double()
    c2 = ctypes.c_double()

    c_halomodel.M1_to_M2(model, M1, c1, Delta1, Delta2, z, ctypes.pointer(M2), ctypes.pointer(c2))

    return M2.value, c2.value

def log10M1_to_log10M2(model, log10M1, log10c1, Delta1, Delta2, z):

    if log10c1 is None:
        c1 = None
    else:
        c1 = pow(10.0, log10c1)

    M2, c2 = M1_to_M2(model, pow(10.0, log10M1), c1, Delta1, Delta2, z)

    return np.log10(M2), np.log10(c2)

def setInterParaGas(model, log10Mh):
    """
    Set log10n0, log10beta and log10rc in model
    from interpolated values, and returns
    the values
    """

    if isinstance(log10Mh, np.ndarray):

        gas_log10n0 = []
        gas_log10beta = []
        gas_log10rc  = []
        for m in log10Mh:
            gas_log10n0.append(c_halomodel.inter_gas_log10n0(model, m))
            gas_log10beta.append(c_halomodel.inter_gas_log10beta(model, m))
            gas_log10rc.append(c_halomodel.inter_gas_log10rc(model, m))

        return [gas_log10n0, gas_log10beta, gas_log10rc]

    else:

        model.gas_log10n0 = c_halomodel.inter_gas_log10n0(model, log10Mh)
        model.gas_log10beta = c_halomodel.inter_gas_log10beta(model, log10Mh)
        model.gas_log10rc  = c_halomodel.inter_gas_log10rc(model, log10Mh)


        return [model.gas_log10n0 , model.gas_log10beta, model.gas_log10rc]





# ----------------------------------------------------- #
# main
# ----------------------------------------------------- #


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('option',          help="Which task to do")
    parser.add_argument('-o', '--output',  default=None, help='output file')

    args = parser.parse_args()

    main(args)


# ----------------------------------------------------- #
# ----------------------------------------------------- #
# ----------------------------------------------------- #
# DEPRECATED
# ----------------------------------------------------- #
# ----------------------------------------------------- #
# ----------------------------------------------------- #



def testNicaea_DEPRECATDED(args):

    model = Model()
    z     = 0.311712 #0.29

    # linear power spectrum
    if True:

        c_halomodel.P_lin.argtypes     = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double]
        c_halomodel.P_lin.restype      = ctypes.c_double
        c_halomodel.P_nonlin.argtypes  = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double]
        c_halomodel.P_nonlin.restype   = ctypes.c_double

        for k in np.exp(np.linspace(np.log(1.e-4), np.log(20.0), 50)):
            # fac  = k*k*k/(2*np.pi*np.pi);
            fac = 1.0
            print k, c_halomodel.P_lin(model, z, k) * fac, c_halomodel.P_nonlin(model, z, k) * fac

    if False:

        c_halomodel.sigmaR2.argtypes  = [ctypes.POINTER(Model), ctypes.c_double]
        c_halomodel.sigmaR2.restype   = ctypes.c_double

        print np.sqrt(c_halomodel.sigmaR2(model, 8.0)) # sigma8

    if False:

        c_halomodel.rho_crit.argtypes  = [ctypes.POINTER(Model), ctypes.c_double]
        c_halomodel.rho_crit.restype   = ctypes.c_double

        print c_halomodel.rho_crit(model, z)

    if False:

        c_halomodel.delta_c.argtypes  = [ctypes.POINTER(Model), ctypes.c_double]
        c_halomodel.delta_c.restype   = ctypes.c_double

        print c_halomodel.delta_c(model, z)

    if False:

        c_halomodel.dndlnMh.argtypes  = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double]
        c_halomodel.dndlnMh.restype   = ctypes.c_double

        for m in np.exp(np.linspace(np.log(1.e3), np.log(2.e16), 100)):
            print m, c_halomodel.dndlnMh(model, z, m)

    if False:

        c_halomodel.bias_h.argtypes  = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double]
        c_halomodel.bias_h.restype   = ctypes.c_double

        for m in np.exp(np.linspace(np.log(1.e3), np.log(2.e16), 100)):
            print m, c_halomodel.bias_h(model, z, m)

    return






def Ix_DEPRECATED(R, model, PSF=None, obs_type="cen", log10Mh=-1.0, z=0.0):
    """
    Wrapper for c-function Ix()

    Returns the X-ray luminosity profile

    INPUT PARAMETERS:
    R: (array or single value) in physical units (Mpc)
    obs_type: [cen, sat]

    """

    R      = np.asarray(R, dtype=np.float64)
    result = np.asarray(np.zeros(len(R)), dtype=np.float64)


    # PSF in model
    if PSF is not None:
        model.XMM_PSF_A     = PSF[0]
        model.XMM_PSF_rc   = PSF[1]
        model.XMM_PSF_alpha = PSF[2]

    # this is used if model.HOD = 0 (non HOD modelling =)
    # to truncate the halo at the viral radius
    if log10Mh > -1.0:
        model.gas_r_h = rh(pow(10.0, log10Mh), Delta(z), rho_crit(z))
        #model.gas_r_h = rh(pow(10.0, log10Mh), Deltavir(z), rho_crit(z))

    else:
        model.gas_r_h = 100.0

    c_halomodel.Ix(R, len(R), model, result, obs_type)

    #if PSF is not None:
    #    result = PSFConvKing(R, result, PSF[1], PSF[2], A=PSF[0])

    return result




def setInterPara_DEPRECATED(model, log10Mh):
    """
    Set log10n0, beta and log10rc in model
    from interpolated values, and returns
    the values
    """

    if isinstance(log10Mh, np.ndarray):

        gas_log10n0 = []
        gas_log10beta = []
        gas_log10rc  = []
        for m in log10Mh:
            gas_log10n0.append(c_halomodel.inter_gas_log10n0(model, m))
            gas_log10beta.append(c_halomodel.inter_gas_log10beta(model, m))
            gas_log10rc.append(c_halomodel.inter_gas_log10rc(model, m))

        return [gas_log10n0, gas_log10beta, gas_log10rc]

    else:

        model.gas_log10n0 = c_halomodel.inter_gas_log10n0(model, log10Mh)
        model.gas_log10beta      = c_halomodel.inter_gas_log10beta(model, log10Mh)
        model.gas_log10rc  = c_halomodel.inter_gas_log10rc(model, log10Mh)


        return [model.gas_log10n0 , model.gas_log10beta, model.gas_log10rc]






def setHaloModel_DEPRECATED(model, fileInName, z):
    """
    Set the halo mass function to model from a file
    created by getHodModel.
    """
    from   astropy.io import ascii

    if fileInName is not None:
        hmf = ascii.read(fileInName, format="no_header")

        # Halo mass function
        h                 = cosmo.H0.value/100.0
        x                 = hmf['col1'] - np.log10(h)
        y                 = hmf['col2'] * h**3
        model.hmf_N       = len(hmf)
        model.hmf_log10Mh = np.ctypeslib.as_ctypes(x)
        model.hmf_dndlnMh = np.ctypeslib.as_ctypes(y)

        # r_h (for the tuncation radius and profile normalisation)
        #r_h = halomodel.rh(pow(10.0, x), halomodel.Delta(z), halomodel.rho_crit(z))
        r_h = rh(pow(10.0, x), z, mass_def="vir")

        # no truncation

        model.hmf_r_h = np.ctypeslib.as_ctypes(r_h)

        return x, y, r_h





def testCamb_DEPRECATED(args):
    ''' test camb python wrapper
    (see http://camb.readthedocs.io/en/latest/CAMBdemo.html)
    '''

    import camb
    from camb import model, initialpower

    #Now get matter power spectra and sigma8 at redshift 0 and 0.8
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=cosmo.H0.value)
    pars.set_dark_energy() #re-set defaults
    pars.InitPower.set_params(ns=0.963)
    #Not non-linear corrections couples to smaller scales than you want
    pars.set_matter_power(redshifts=[0.29], kmax=2.0)

    #Linear spectra
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)
    s8 = np.array(results.get_sigma8())

    #Non-Linear spectra (Halofit)
    pars.NonLinear = model.NonLinear_both
    results.calc_power_spectra(pars)
    kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)

    for i in range(len(kh)):
        print kh[i], pk[0][i], pk_nonlin[0][i]

    return



def DeltaSigma_DEPRECATED(R, zl, model, como=False, Kappa=False, gal_type="cen", c=None, Delta_def="200m", eps=1.e-10, log10Mstar=10.0):
    """
    Returns DeltaSigma for NFW halo mass profile (in h Msun/pc^2) -
    PHYSICAL UNITS, unless como=True set

    INPUT PARAMETERS:
    R: (array or single value) in physical units (Mpc/h)
    zl: redshift of the lens
    c: value, c-M relation (string) or None. If None the default relation is Duffy08
    Delta: overdensity wrt the mean density

    gal_type: [cen, sat, stars], for "stars", set log10Mstar

    If M given defined wrt the *mean* density
    then give Delta_m X Omega_m(z) as Delta

    REFS: Sec. 7.5.1 of Mo, van den Bosh & White's Galaxy
    formation and evolution Wright & Brainerd (200) after
    correcting the typo in Eq. (7.141)
    """

    # Mh = 0.0
    if model.ggl_log10Mh < eps: return 0.0*R

    # Mass - concentration relationship
    # from literature relation
    #if isinstance(c, basestring):
    #    c = concentration(Mh, zl, ref=c)
    #if c is None:
    #    c = concentration(Mh, zl)

    # M: in Msun unit, defined within a radius enclosing
    # an overdensity of Delta times the *critical* density at redshift z


    if c is None:
        c = pow(10.0, model.ggl_log10c)
    Mh = pow(10.0, model.ggl_log10Mh)

    # if comoving coordinates, set z0 = 0.0
    z0 = 0.0 if como else zl

    Delta_h = Delta(z0, mass_def = Delta_def)

    #r_s, rho_s = r_s_rho_s(Mh, c, Delta_h, z0)
    model.ggl_r_s, model.ggl_rho_s = r_s_rho_s(Mh, c, z0, mass_def=Delta_def)
    model.ggl_r_h                  = rh(Mh, z0, mass_def=Delta_def)
    model.ggl_rho_bar              = rhobar(z0)



    R      = np.asarray(R, dtype=np.float64)
    result = np.asarray(np.zeros(len(R)), dtype=np.float64)

    #constants = np.asarray([Mh, c, r_s, rho_s, r_h, rho_bar], dtype=np.float64)

    if gal_type == "cen":
        c_halomodel.DeltaSigma(R, len(R), model, result, 1)

    if gal_type == "sat":
        c_halomodel.DeltaSigma(R, len(R), model, result, 2)

    if gal_type == "stars":
        result = 1.e-12 * pow(10.0, log10Mstar)/(np.pi*R*R)



    return result




def concentration_DEPRECATED(Mh, z, ref="Duffy08"):
    """
    concentration for clusters of mass Mh
    at redshift z

    ATTENTION, each c-M relation is valid
    only with the same mass defintion:

    c = wrt to critical density
    m = wrt to mean density

    Duffy08 -> M200c
    Munoz-Cuartas2011 -> Mvir  (r_vir from Bryan & Norman, 1998)

    """

    if ref == "Duffy08":
        # Duffy et al. (2011)
        # WMAP5 cosmology
        # Mass defintion -> M200c
        # to do add M200m and Mvirc
        h      = cosmo.H0.value/100.0
        Mpivot = 2.0e12 * h

        return 5.71 * pow(Mh/Mpivot, -0.084) * pow(1.0+z, -0.47)


    if ref == "MunozCuartas11":
        # Munoz-Cuartas et al. 2011
        # Mass defintion -> Mvir
        # (r_vir from Bryan & Norman, 1998)

        aa = 0.029*z - 0.097;
        bb = -110.001/(z+16.885) + 2469.720/(z+16.885)**2;
        log10_c = aa*np.log10(Mh) + bb;

        return pow(10.0, log10_c)


    if ref == "TakadaJain02":
        # Takada & Jain et al. 2002
        # concentration for clusters
        # with mass Mvirm in Msun unit
        # With Mvir, WMAP5 cosmology and
        # c0, beta parameters used in Coupon et al. (2015)

        c0 = 11.0
        beta = 0.13

        # WMAP5 cosmology
        Ms = 2.705760e+12/0.72

        return c0*pow(Mh/Ms, -beta)/(1.0+z)

    if ref == "Bhattacharya12_full":
        # BHATTACHARYA et al. 2012
        # concentration for all clusters
        # with mass M200c in Msun unit

        # compute D+ and nu
        D_plus = Dplus(z)

        h     = cosmo.H0.value/100.0
        Mh0   = 5.0e13 / h
        nu    = 1.0/D_plus * (1.12 * pow(Mh/Mh0, 0.3) + 0.53)

        return D_plus**1.15 * 9.0*nu**(-0.29)


    if ref == "Bhattacharya12_relaxed":
        # BHATTACHARYA et al. 2012
        # concentration for relaxed clusters
        # with mass M200c in Msun unit

        # compute D+ and nu
        D_plus = Dplus(z)

        h     = cosmo.H0.value/100.0
        Mh0   = 5.0e13 / h
        nu    = 1.0/D_plus * (1.12 * pow(Mh/Mh0, 0.3) + 0.53)

        return D_plus**0.53 * 6.6*nu**(-0.41)


    if ref == "Bullock01":

        return -1

    print "concentration: {0:s is not defined}".format(ref)
    exit(-1)


def r_s_rho_s_DEPRECATED(Mh, c, z, mass_def="500c"):
    """
    Returns r_s, rho_s for a NFW profile with
    parameters Mh and c at redshift z.

    ATTENTION: Delta is defined as the overdensity
    in a radius rh enclosing Delta times the CRITICAL
    density of the Universe at redshift z

    For masses defined as Delta times the MEAN density
    set Delta = Delta_m * Omega(z)
    """

    rho_crit_z = rho_crit(z)
    r_h        = rh(Mh, z, mass_def=mass_def)

    r_s     = r_h/c
    rho_s   = rho_crit_z* Delta(z, mass_def = mass_def)/3.0*c**3.0/(np.log(1.0+c)-c/(1.0+c))

    return r_s, rho_s

def rhobar_DEPRECATED(z):
    """
    mean matter density of
    the Universe at redhsift z
    """

    return rho_crit(z) * cosmo.Om(z)

def rho_crit_DEPRECATED(z):
    """
    Returns the critical density
    of the Universe at redshift z
    """

    G = 4.302e-9                                   # in [km^2 s^-2 M_sun^-1 Mpc^1]
    return 3.0*cosmo.H(z).value**2.0/(8.0*np.pi*G) # in M_sol Mpc^3


def Dplus_DEPRECATED(z):

    # To be replaced by NICAE DPLUS function

    a    = 1.0/(1.0+z)
    res  = integrate.quad(intForDplus, 0.0, a)
    norm = integrate.quad(intForDplus, 0.0, 1.0)

    return cosmo.efunc(z) * res[0]/norm[0]

def intForDplus_DEPRECATED(a):
    z = 1.0/a - 1.0
    return (a*cosmo.efunc(z))**(-3.0)

def NFW_sum_DEPRECATED(c):
    return (np.log(1.0+c) - c/(1.0+c))/(c*c*c)


def log10M1_to_log10M2_DEPRECATED(z, log10M1, Delta1, Delta2, c1=None):
    """
    See Hu & Kravtsov - appendix C
    NFW_sum(c_200)/Delta_200 = NFW_sum(c_vir)/Delta_vir
    Delta1 and Delta2 are overdensities with respect to
    the critical matter density
    """

    Delta1 = Delta(z, mass_def = Delta1)
    Delta2 = Delta(z, mass_def = Delta2)

    M1 = pow(10.0, log10M1)

    # concentration for M1
    # Mass - concentration relationship
    if isinstance(c1, basestring):
        c1 = concentration(M1, z, ref=c1)
    if c1 is None:
        c1 = concentration(M1, z)

    f  = NFW_sum(c1)*Delta2/Delta1
    p  = -0.4283-3.13e-3*np.log(f)-3.52e-5*np.log(f)*np.log(f)
    x  = (0.5116*f**(2.0*p)+(3.0/4.0)**2.0)**(-1.0/2.0)+2.0*f
    c2 = 1.0/x
    M2 = M1*Delta2/Delta1*(c2/c1)**3.0

    return np.log10(M2), c2


def M1_to_M2_DEPRECATED(z, M1, Delta1, Delta2, c1=None):
    """
    See Hu & Kravtsov - appendix C
    NFW_sum(c_200)/Delta_200 = NFW_sum(c_vir)/Delta_vir
    Delta1 and Delta2 are overdensities with respect to
    the critical matter density
    """

    # concentration for M1
    # Mass - concentration relationship
    if isinstance(c1, basestring):
        c1 = concentration(M1, z, ref=c1)
    if c1 is None:
        c1 = concentration(M1, z)

    f  = NFW_sum(c1)*Delta2/Delta1
    p  = -0.4283-3.13e-3*np.log(f)-3.52e-5*np.log(f)*np.log(f)
    x  = (0.5116*f**(2.0*p)+(3.0/4.0)**2.0)**(-1.0/2.0)+2.0*f
    c2 = 1.0/x
    M2 = M1*Delta2/Delta1*(c2/c1)**3.0

    return M2, c2


def King_DEPRECATED(r, A, rc, alpha):
    """
    Returns King's profile.
    """
    return A * pow(1.0+(r**2/rc**2), -alpha)


def PSFConvKing_DEPRECATED(R, model, rc, alpha, A=None):

    """
    Convolve model with King's profile PSF
    A * pow(1.0+(r**2/rc**2), -alpha)
    """

    rmin = 1.e-3
    rmax = 1.e+3

    # normalise PSF
    if A is None:
        X    = np.exp(np.linspace(np.log(rmin), np.log(rmax), 50))
        Norm = np.trapz(King(X, 1.0, rc, alpha), X)
        A = 1.0/Norm

    # interpolate model
    model_inter = interpolate.interp1d(R, model, fill_value=0.0, bounds_error=False)

    # theta and R prime
    theta    = np.linspace(0.0, 2.0*np.pi, 50)
    Rp       = np.exp(np.linspace(np.log(rmin), np.log(rmax), 50))
    model_Rp = np.zeros(len(Rp))

    result = np.zeros(len(R))
    for i in range(len(R)):
        for j in range(len(Rp)):
            model_Rp[j] = 1.0/(2.0*np.pi) * np.trapz(model_inter(np.sqrt(R[i]**2 + Rp[j]**2 - 2.0*R[i]*Rp[j]*np.cos(theta))), theta)
        result[i] = np.trapz(King(Rp, A, rc, alpha) * model_Rp, Rp)

    return result



def DeltaSigmaSat_DEPRECATED(r, zl, Mh, c="Duffy08", Delta=200.0, como=False, Kappa=False, eps=1.e-10):
    """
    Returns DeltaSigma for NFW halo mass profile (in Msun/pc^2) -
    PHYSICAL UNITS, unless como=True set

    INPUT PARAMETERS:
    r: (array or single value) in physical units (Mpc)
    zl: redshift of the lens
    M: in Msun unit, defined within a radius enclosing
    an overdensity of Delta times the *critical* density at redshift z
    c: value, c-M relation (string) or None. If None the default relation is Duffy08
    Delta: overdensity wrt the mean density

    If M given defined wrt the *mean* density
    then give Delta_m X Omega_m(z) as Delta

    REFS: Sec. 7.5.1 of Mo, van den Bosh & White's Galaxy
    formation and evolution Wright & Brainerd (200) after
    correcting the typo in Eq. (7.141)
    """

    # Mh = 0.0
    if Mh < eps: return 0.0*r

    # Mass - concentration relationship
    if isinstance(c, basestring):
        c = concentration(Mh, zl, ref=c)
    if c is None:
        c = concentration(Mh, zl)

    # if comoving coordinates set z0 = 0.0
    z0 = 0.0 if como else zl

    r_s, rho_s = r_s_rho_s(Mh, c, Delta, z0)
    r_h        = rh(Mh, Delta, rho_crit(z0))

    return (SigmaMeanSat(r/r_s, r_s, rho_s, r_h) - SigmaSat(r/r_s, r_s, rho_s, r_h))/1.0e12


def Sigma_DEPRECATED(x, r_s, rho_s, eps=1.e-10, compute="analytic"):

    result = np.zeros(len(x))

    if compute == "analytic":

        is_one = ((1.0 - eps < x) & (x < 1.0 + eps)).nonzero()
        result[is_one] = 2.0/3.0

        is_inf_one = (x < 1.0).nonzero()
        result[is_inf_one] = 2.0/(x[is_inf_one]**2.0 - 1.0) \
            * (1.0 - 2.0/np.sqrt(1.0-x[is_inf_one]**2.0) \
            * np.arctanh(np.sqrt((1.0-x[is_inf_one])/(1.0+x[is_inf_one]))))

        is_sup_one = (x > 1.0).nonzero()
        result[is_sup_one] = 2.0/(x[is_sup_one]**2.0 - 1.0) \
            * (1.0 - 2.0/np.sqrt(x[is_sup_one]**2.0-1.0) \
            * np.arctan(np.sqrt((x[is_sup_one]-1.0)/(1.0+x[is_sup_one]))))

        return r_s*rho_s*result

    if compute == "bruteForce":

        #logz  = np.arange(np.log(1.e-3), np.log(1.e2), 0.1)
        logz  = np.arange(np.log(1.e-3), np.log(1.e2), 0.1)
        z     = np.exp(logz)
        dlogz = np.log(z[1]/z[0])

        for i in range(len(x)):
            result[i] = np.sum(NFW_x(np.sqrt(x[i]**2 + z**2)) * z * dlogz)

        return 2.0*r_s*rho_s*result


    if compute == "quad":

        logzmin = np.log(1.e-3)
        logzmax = np.log(1.e2)

        result = np.zeros(len(x))
        for i in range(len(x)):
            result[i] = integrate.quad(intForSigma, logzmin, logzmax, args=(x[i]), epsrel=1.e-3)[0]

        return 2.0*r_s*rho_s*result


def intForSigma_DEPRECATED(logz, x):

    z = np.exp(logz)
    X = np.sqrt(x**2 + z**2)

    return NFW_x(X) * z


def SigmaMean_DEPRECATED(x, r_s, rho_s, eps=1.e10, compute="analytic"):

    result = np.zeros(len(x))

    if compute == "analytic":

        is_one = ((1.0 - eps < x) & (x < 1.0 + eps)).nonzero()
        result[is_one] = 4.0*(1.0+np.log(0.5))

        is_inf_one = (x < 1.0).nonzero()
        result[is_inf_one] = 4.0/x[is_inf_one]**2.0 \
            * (2.0/np.sqrt(1.0-x[is_inf_one]**2.0)\
            * np.arctanh(np.sqrt((1.0-x[is_inf_one])/(1.0+x[is_inf_one]))) + np.log(x[is_inf_one]/2.0))

        is_sup_one = (x > 1.0).nonzero()
        result[is_sup_one] = 4.0/x[is_sup_one]**2.0\
            * (2.0/np.sqrt(x[is_sup_one]**2.0-1.0)\
            * np.arctan(np.sqrt((x[is_sup_one]-1.0)/(1.0+x[is_sup_one]))) + np.log(x[is_sup_one]/2.0))

        return r_s*rho_s*result

    if compute == "bruteForce":

        logz  = np.arange(np.log(1.e-3), np.log(1.e2), 0.1)
        z     = np.exp(logz)
        dlogz = np.log(z[1]/z[0])

        for i in range(len(x)):
            logxp  = np.arange(np.log(1.e-3), np.log(x[i]), 0.1)
            xp     = np.exp(logxp)
            dlogxp = np.log(xp[1]/xp[0])
            Sigma_x = np.zeros(len(xp))

            for j in range(len(xp)):
                Sigma_x[j] = np.sum(NFW_x(np.sqrt(xp[j]**2 + z**2)) * z * dlogz)


            result[i] = 2.0/(x[i]*x[i]) * np.sum(xp * Sigma_x * xp * dlogxp)

        return 2.0*r_s*rho_s*result


    if compute == "quad":

        for i in range(len(x)):

            logxmin = np.log(1.e-3)
            logxmax = np.log(x[i])

            result[i]= 2.0/x[i]**2 * integrate.quad(intForSigmaMean, logxmin, logxmax, args=(r_s, rho_s), epsrel=1.e-3)[0]

        return result


def intForSigmaMean_DEPRECATED(logx, r_s, rho_s):

    x = np.exp(logx)
    return x * Sigma(np.array([x]), r_s, rho_s)[0] * x


def SigmaMeanSat_DEPRECATED(x, r_s, rho_s, r_h):

    result = np.zeros(len(x))

    for i in range(len(x)):

        logxmin = np.log(1.e-3)
        logxmax = np.log(x[i])

        result[i]= 2.0/x[i]**2 * integrate.quad(intForSigmaSatMean, logxmin, logxmax, args=(r_s, rho_s, r_h), epsrel=1.e-1)[0]

    return result


def intForSigmaSatMean_DEPRECATED(logx, r_s, rho_s, r_h):

    x = np.exp(logx)
    return x * SigmaSat(np.array([x]), r_s, rho_s, r_h)[0] * x

def SigmaSat_DEPRECATED(x, r_s, rho_s, r_h):

    result = np.zeros(len(x))

    theta  = np.linspace(0.0, 2.0*np.pi, 50)
    dtheta = theta[1] - theta[0]

    logRoff    = np.linspace(np.log(1.e-3), np.log(r_h), 50)
    Roff       = np.exp(logRoff)
    Sigma_Roff = np.zeros(len(Roff))

    R = x * r_s

    for i in range(len(R)):
        for j in range(len(Roff)):
            Sigma_Roff[j] = 1.0/(2.0*np.pi) * np.trapz(Sigma(np.sqrt(R[i]**2 + Roff[j]**2 - 2.0*R[i]*Roff[j]*np.cos(theta))/r_s, r_s, rho_s), theta)
        result[i] = np.trapz(Psat(Roff, r_s, rho_s, r_h) * Sigma_Roff, Roff)

    return result


def Psat_DEPRECATED(Roff, r_s, rho_s, r_h):

    logR    = np.linspace(np.log(1.e-3), np.log(r_h), 50)
    R       = np.exp(logR)
    tot     = R * Sigma(R/r_s, r_s, rho_s)
    norm    = np.trapz(tot, R)

    result = Roff * Sigma(Roff/r_s, r_s, rho_s)/ norm
    result[Roff > r_h] = 0.0

    return result
