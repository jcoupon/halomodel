#!/usr/bin/env python

"""
Jean coupon - 2016 - 2018
python wrapper for halomodel routines in c

IMPORTANT:

1. halomodel.py is exclusively in
hubble units:
halo mass: [h^-1 Msun]
galaxy stellar mass: [h^-2 Msun]
distances: [h^-1 Mpc]

2. halomodel is exclusively in
comoving units (concerns the stellar
mass function and the galaxy-galaxy
lensing).

Required librairies:

for c (if not in /usr/local/, set path in Makefile):
- fftw3 3.3.4 (http://www.fftw.org/)
- gsl 2.1 (https://www.gnu.org/software/gsl/)
- nicaea 2.7 (http://www.cosmostat.org/software/nicaea/)

for python:
- numpy 1.10.2 (http://www.numpy.org/)
- scipy 0.17.1 (https://www.scipy.org/scipylib/download.html)
- astropy 1.2.1 (http://www.astropy.org/)

"""

# see http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
# __version__ = "1.0.2"

from __future__ import print_function, division

import os
import numpy as np
import sys
import inspect
import collections
import ctypes
import traceback
import types

import scipy.interpolate as interpolate
from astropy.io import fits,ascii
from astropy.table import Table, Column
import matplotlib.pyplot as plt


"""

-------------------------------------------------------------
path to c library
-------------------------------------------------------------

"""

# script directory
HALOMODEL_DIRNAME = os.path.dirname(
        os.path.realpath(inspect.getfile(inspect.currentframe())))

# c library
C_HALOMODEL = ctypes.cdll.LoadLibrary(HALOMODEL_DIRNAME+"/lib/libhalomodel.so")


"""

-------------------------------------------------------------
classes
-------------------------------------------------------------

"""

class Model(ctypes.Structure):
    """ Structure to serve as the interface between
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
        ("h", ctypes.c_double),
        ("Omega_b", ctypes.c_double),
        ("sigma_8", ctypes.c_double),
        ("n_s", ctypes.c_double),
        ("log10h", ctypes.c_double),
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
        ("haloExcl", ctypes.c_int),

        # if using hod model
        ("hod", ctypes.c_int),

        # for X-ray binaries
        ("IxXB_Re", ctypes.c_double),
        ("IxXB_CR", ctypes.c_double),

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

        # for tx(Mh) - Temperature-Mass relationship
        ("gas_TGasMh_N", ctypes.c_int),
        ("gas_TGasMh_log10Mh", ctypes.POINTER(ctypes.c_double)),
        ("gas_TGasMh_log10TGas", ctypes.POINTER(ctypes.c_double)),

        # for ZGas(Mh) - Metallicity-Mass relationship
        ("gas_ZGasMh_N", ctypes.c_int),
        ("gas_ZGasMh_log10Mh", ctypes.POINTER(ctypes.c_double)),
        ("gas_ZGasMh_ZGas", ctypes.POINTER(ctypes.c_double)),

        # for Lx to CR conversion flux [CR] = Lx * fac
        ("gas_LxToCR_NZGas", ctypes.c_int),
        ("gas_LxToCR_NTGas", ctypes.c_int),
        ("gas_LxToCR_ZGas", ctypes.POINTER(ctypes.c_double)),
        ("gas_LxToCR_log10TGas", ctypes.POINTER(ctypes.c_double)),
        ("gas_LxToCR_log10fac", ctypes.POINTER(ctypes.c_double)),

        # for gg lensing, if hod = 0
        ("ggl_pi_max", ctypes.c_double),
        ("ggl_log10c", ctypes.c_double),
        ("ggl_log10Mh", ctypes.c_double),
        ("ggl_log10Mstar", ctypes.c_double),

        # for wtheta, if hod = 1
        ("wtheta_nz_N", ctypes.c_int),
        ("wtheta_nz_z", ctypes.POINTER(ctypes.c_double)),
        ("wtheta_nz", ctypes.POINTER(ctypes.c_double)),

        # if hod = 1, one may input non-parametric HOD - centrals
        ("HOD_cen_N", ctypes.c_int),
        ("HOD_cen_log10Mh", ctypes.POINTER(ctypes.c_double)),
        ("HOD_cen_Ngal", ctypes.POINTER(ctypes.c_double)),

        # if hod = 1, one may input non-parametric HOD - satellites
        ("HOD_sat_N", ctypes.c_int),
        ("HOD_sat_log10Mh", ctypes.POINTER(ctypes.c_double)),
        ("HOD_sat_Ngal", ctypes.POINTER(ctypes.c_double)),

        # XMM PSF, King function parameters
        # ("XMM_PSF_A", ctypes.c_double),
        ("XMM_PSF_rc_deg", ctypes.c_double),
        ("XMM_PSF_alpha", ctypes.c_double)

        ]



    _defaults_ = {

        # cosmology
        'Omega_m' : 0.258,
        'Omega_de' : 0.742,
        'H0' : 72.0,
        'Omega_b' : 0.0441,
        'sigma_8' : 0.796,
        'n_s' : 0.963,

        # halo mass definition: M500c, M500m, M200c, M200m, Mvir, MvirC15
        'massDef' : 'M500c',

        # mass/concentration relation: D11, M11, TJ03, B12_F, B12_R, B01
        'concenDef' : 'TJ03',

        # halo mass defintion: PS74, ST99, ST02, J01, T08
        'hmfDef' : 'T08',

        # mass/bias relation:  PS74, ST99, ST02, J01, T08
        'biasDef' : 'T08',

        # halo model / HOD parameters
        'log10M1' : 12.5, # in Msun h^-1
        'log10Mstar0' : 10.6, # in Msun h^-2
        'beta' : 0.3,
        'delta' : 0.7,
        'gamma' : 1.0,
        'log10Mstar_min' : 10.00,
        'log10Mstar_max' : 11.00,
        'sigma_log_M0' : 0.2,
        'sigma_lambda' : 0.0,
        'B_cut' : 1.50,
        'B_sat' : 10.0,
        'beta_cut' : 1.0,
        'beta_sat' : 0.8,
        'alpha' : 1.0,
        'fcen1' : -1,
        'fcen2' : -1,
        'haloExcl' : 1,

        # if using hod model
        'hod' : 0,

        # for X-ray binaries
        'IxXB_Re' : 0.01196, # in h^-1 Mpc
        'IxXB_CR' : 0.0, # in CR

        # X-ray, if hod' :0
        'gas_log10n0' : -3.0,
        'gas_log10beta' : -1.0,
        'gas_log10rc' : -1.0,

        # X-ray, if hod' :1
        # log10n0' :gas_log10n0_1  + gas_log10n0_2 * (log10Mh-14.0)
        # n0 in [h^3 Mpc^-3], Mpc in comoving coordinate.
        'gas_log10n0_1' : -2.5,
        'gas_log10n0_2' : 1.0,
        'gas_log10n0_3' : np.nan, # not used
        'gas_log10n0_4' : np.nan, # not used

        # spline function
        'gas_log10beta_1' : np.log10(2.0),
        'gas_log10beta_2' : np.log10(0.35),
        'gas_log10beta_3' : np.log10(0.5),
        'gas_log10beta_4' : np.log10(0.5),

        # spline function
        'gas_log10rc_1' : np.log10(0.3),
        'gas_log10rc_2' : np.log10(0.04),
        'gas_log10rc_3' : np.log10(0.08),
        'gas_log10rc_4' : np.log10(0.08),

        # for tx(Mh) - Temperature-Mass relationship
        'gas_TGasMh_N' : 0,
        'gas_TGasMh_log10Mh' : None,
        'gas_TGasMh_log10TGas' : None,

        # for ZGas(Mh) - Metallicity-Mass relationship
        'gas_ZGasMh_N' : 0,
        'gas_ZGasMh_log10Mh' : None,
        'gas_ZGasMh_ZGas' : None,

        # for Lx to CR conversion flux [CR]' :Lx * fac
        'gas_LxToCR_NZGas' : 0,
        'gas_LxToCR_NTGas' : 0,
        'gas_LxToCR_ZGas' : None,
        'gas_LxToCR_log10TGas' : None,
        'gas_LxToCR_log10fac' : None,

        # for gg lensing
        'ggl_pi_max' : 60.0,
        'ggl_log10c' : np.nan,
        'ggl_log10Mh' : 14.0,
        'ggl_log10Mstar' : np.nan,

        # n(z) for wtheta, if hod' : 1
        'wtheta_nz_N' : 0,
        'wtheta_nz_z' : None,
        'wtheta_nz' : None,

        # if hod' : 1, one may input non-parametric HOD - centrals
        'HOD_cen_N' : 0,
        'HOD_cen_log10Mh' : None,
        'HOD_cen_Ngal' : None,

        # if hod' : 1, one may input non-parametric HOD - satellites
        'HOD_sat_N' : 0,
        'HOD_sat_log10Mh' : None,
        'HOD_sat_Ngal' : None,

        # XMM PSF
        # ATTENTION: rc is in degrees
        # 'XMM_PSF_A' : np.nan
        'XMM_PSF_rc_deg' : np.nan,
        'XMM_PSF_alpha' : np.nan
    }


    def print_default(self):
        """ Print the default values
        """

        for k in self._defaults_:
            print('{} {}'.format(k, self._defaults_[k]))


    def print_current(self):
        """ Print the current values
        """

        for k in self._defaults_:
            print('{} {}'.format(k, getattr(self, k)))

    def __init__(self, *args, **kwargs):
        """ Ctypes.Structure with integrated default values.
        https://www.programcreek.com/python/example/105644/ctypes.Structure.__init__
        https://stackoverflow.com/questions/7946519/default-values-in-a-ctypes-structure/25892189#25892189

        :param kwargs: values different to defaults
        :type kwargs: dict
        """

        # sanity checks
        defaults = type(self)._defaults_
        assert type(defaults) is types.DictionaryType

        # use defaults
        values = defaults.copy()
        for k in values:
            setattr(self, k, values[k])

        # set attributes passed during init
        for k in kwargs:
            values[k] = kwargs[k]
            setattr(self, k, kwargs[k])

        self.h = self.H0/100.0
        self.log10h = np.log10(self.h)

        # appropriately initialize ctypes.Structure
        # super().__init__(**values)                       # Python 3 syntax
        return ctypes.Structure.__init__(self, **values)  # Python 2 syntax

        # TODO
        # for each redshift:
        # add toLxBolo: Zgas log10TGas fac
        # add TGasMh: log10Mh TGas
        # add ZGasMh: log10Mh log10Zgas

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


C_HALOMODEL.xi_gg.argtypes = [ctypes.POINTER(Model), np.ctypeslib.ndpointer(dtype = np.float64), ctypes.c_int, ctypes.c_double, ctypes.c_int, np.ctypeslib.ndpointer(dtype = np.float64)]
C_HALOMODEL.SigmaIx.argtypes = [ctypes.POINTER(Model), np.ctypeslib.ndpointer(dtype = np.float64), ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, np.ctypeslib.ndpointer(dtype = np.float64)]
C_HALOMODEL.rh.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double, ctypes.c_double]
C_HALOMODEL.rh.restype = ctypes.c_double
C_HALOMODEL.bias_h.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double]
C_HALOMODEL.bias_h.restype = ctypes.c_double
C_HALOMODEL.concentration.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double, ctypes.c_char_p]
C_HALOMODEL.concentration.restype = ctypes.c_double
C_HALOMODEL.M1_to_M2.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
C_HALOMODEL.Omega_m_z.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double]
C_HALOMODEL.Omega_m_z.restype = ctypes.c_double
C_HALOMODEL.Delta.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_char_p]
C_HALOMODEL.Delta.restype = ctypes.c_double
C_HALOMODEL.r_vir.argtypes = [ ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double, ctypes.c_double ]
C_HALOMODEL.r_vir.restype = ctypes.c_double
C_HALOMODEL.Delta_vir.argtypes = [ctypes.POINTER(Model), ctypes.c_double]
C_HALOMODEL.Delta_vir.restype = ctypes.c_double
C_HALOMODEL.msmh_log10Mh.argtypes = [ctypes.POINTER(Model), ctypes.c_double]
C_HALOMODEL.msmh_log10Mh.restype = ctypes.c_double
C_HALOMODEL.log10M_sat.argtypes = [ctypes.POINTER(Model), ctypes.c_double]
C_HALOMODEL.log10M_sat.restype = ctypes.c_double
C_HALOMODEL.log10M_cut.argtypes = [ctypes.POINTER(Model), ctypes.c_double]
C_HALOMODEL.log10M_cut.restype = ctypes.c_double
C_HALOMODEL.nGas.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
C_HALOMODEL.nGas.restype = ctypes.c_double
C_HALOMODEL.inter_gas_log10n0.argtypes = [ctypes.POINTER(Model), ctypes.c_double]
C_HALOMODEL.inter_gas_log10n0.restype = ctypes.c_double
C_HALOMODEL.inter_gas_log10beta.argtypes = [ctypes.POINTER(Model), ctypes.c_double]
C_HALOMODEL.inter_gas_log10beta.restype = ctypes.c_double
C_HALOMODEL.inter_gas_log10rc.argtypes = [ctypes.POINTER(Model), ctypes.c_double]
C_HALOMODEL.inter_gas_log10rc.restype = ctypes.c_double
C_HALOMODEL.lookbackTimeInv.argtypes = [ctypes.POINTER(Model), ctypes.c_double]
C_HALOMODEL.lookbackTimeInv.restype = ctypes.c_double
C_HALOMODEL.lookbackTime.argtypes = [ctypes.POINTER(Model), ctypes.c_double]
C_HALOMODEL.lookbackTime.restype = ctypes.c_double
C_HALOMODEL.DA.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_int]
C_HALOMODEL.DA.restype = ctypes.c_double
C_HALOMODEL.DM.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_int]
C_HALOMODEL.DM.restype = ctypes.c_double
C_HALOMODEL.DL.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_int]
C_HALOMODEL.DL.restype = ctypes.c_double
C_HALOMODEL.msmh_log10Mstar.argtypes = [ctypes.POINTER(Model), ctypes.c_double]
C_HALOMODEL.msmh_log10Mstar.restype = ctypes.c_double
C_HALOMODEL.LambdaBolo.argtypes = [ctypes.c_double, ctypes.c_double]
C_HALOMODEL.LambdaBolo.restype = ctypes.c_double
C_HALOMODEL.Lambda0p5_2p0.argtypes = [ctypes.c_double, ctypes.c_double]
C_HALOMODEL.Lambda0p5_2p0.restype = ctypes.c_double
C_HALOMODEL.LxToCR.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double, ctypes.c_double]
C_HALOMODEL.LxToCR.restype = ctypes.c_double
C_HALOMODEL.phi_c.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double]
C_HALOMODEL.phi_c.restype = ctypes.c_double
C_HALOMODEL.phi_s.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double]
C_HALOMODEL.phi_s.restype = ctypes.c_double
C_HALOMODEL.ngal_den.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
C_HALOMODEL.ngal_den.restype = ctypes.c_double
C_HALOMODEL.Ngal_s.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double, ctypes.c_double]
C_HALOMODEL.Ngal_s.restype = ctypes.c_double
C_HALOMODEL.Ngal_c.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double, ctypes.c_double]
C_HALOMODEL.Ngal_c.restype = ctypes.c_double
C_HALOMODEL.Ngal.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double, ctypes.c_double]
C_HALOMODEL.Ngal.restype = ctypes.c_double
C_HALOMODEL.shmr_s.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double, ctypes.c_double]
C_HALOMODEL.shmr_s.restype = ctypes.c_double
C_HALOMODEL.shmr_c.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double, ctypes.c_double]
C_HALOMODEL.shmr_c.restype = ctypes.c_double
C_HALOMODEL.shmr.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double, ctypes.c_double]
C_HALOMODEL.shmr.restype = ctypes.c_double
C_HALOMODEL.rho_crit.argtypes  = [ctypes.POINTER(Model), ctypes.c_double]
C_HALOMODEL.rho_crit.restype   = ctypes.c_double
C_HALOMODEL.MhToTGas.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double]
C_HALOMODEL.MhToTGas.restype = ctypes.c_double
C_HALOMODEL.MhToZGas.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double]
C_HALOMODEL.MhToZGas.restype = ctypes.c_double
C_HALOMODEL.changeModelHOD.argtypes = [ctypes.POINTER(Model), ctypes.POINTER(Model)]
C_HALOMODEL.changeModelHOD.restype = ctypes.c_int
C_HALOMODEL.uHalo.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
C_HALOMODEL.uHalo.restype  = ctypes.c_double
C_HALOMODEL.uHaloClosedFormula.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
C_HALOMODEL.uHaloClosedFormula.restype  = ctypes.c_double
C_HALOMODEL.uIx.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
C_HALOMODEL.uIx.restype = ctypes.c_double



"""

-------------------------------------------------------------
test
-------------------------------------------------------------

"""

def test():
    """ Perform basic tests.
    """

    """ computeRef = True will compute and write
    the reference quantities whereas
    computeRef = False will compare the current
    computation with the reference quantities
    """
    computeRef = False
    printModelChanges = False

    # list of quantities to compute/check
    actions = [
        'satContrib', 'lookbackTime', 'dist', 'change_HOD',
        'Ngal', 'MsMh', 'concen', 'mass_conv', 'xi_dm',
        'uHalo', 'smf', 'ggl_HOD', 'wtheta_HOD', 'populate'
        ]

    # TODO
    """
    actions += [
        'ggl', 'Lambda', 'LxToCR', 'uIx', 'SigmaIx_HOD',
        'SigmaIx', 'SigmaIx_HOD_nonPara', 'populate'
        ]
    """

    # cosmological model
    model = Model(
        Omega_m = 0.258, Omega_de = 0.742, H0 = 72.0,
        Omega_b = 0.0441, sigma_8 = 0.796, n_s = 0.963,
        hod = 1, massDef = "M200m", concenDef = "TJ03",
        hmfDef = "T08", biasDef = "T08"
        )

    # redshift
    z = 0.308898

    # HOD model
    model.log10M1 = 12.35 # in Msun h^-1
    model.log10Mstar0 = 10.30 # in Msun h^-2
    model.beta = 0.43
    model.delta = 0.76
    model.gamma = 0.0
    model.sigma_log_M0 = 0.19
    model.sigma_lambda = 0.0
    model.B_cut = 2.10
    model.B_sat = 8.70
    model.beta_cut = 0.47
    model.beta_sat = 0.69
    model.alpha = 1.0
    model.fcen1 = -1
    model.fcen2 = -1

    # Stellar mass bins in log10(Mstar/[h^-2 Msun])
    model.log10Mstar_min = 11.00
    model.log10Mstar_max = 11.30

    # record current model
    m1 =  dumpModel(model)

    # astropy for compararison
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=model.H0, Om0=model.Omega_m)

    # satellite HOD times the stellar mass
    if 'satContrib' in actions:

        result = collections.OrderedDict()

        result['log10Mstar'] = np.linspace(7.0, 12.0, 100)
        result['satContrib'] = pow(
            10.0, result['log10Mstar'])*dndlog10Mstar(model,
            result['log10Mstar'], z, obs_type="sat")
        _write_or_check(model, 'satContrib', result, computeRef)

    # look back time
    if 'lookbackTime' in actions:

        result = collections.OrderedDict()

        result['lookbackTime'] = C_HALOMODEL.lookbackTime(model, z)
        result['lookbackTimeInv'] = C_HALOMODEL.lookbackTimeInv(
            model, result['lookbackTime'])
        result['lookbackTimeAstropy'] = cosmo.lookback_time([z]).value
        _write_or_check(model, 'lookbackTime', result, computeRef)

    # angular diameter distance
    if 'dist' in actions:

        result = collections.OrderedDict()

        result['dist'] =  C_HALOMODEL.DA(model, z, 0)/model.h
        result['distAstropy'] = cosmo.angular_diameter_distance([z]).value
        _write_or_check(model, 'dist', result, computeRef)

    # check whether the HOD
    if 'change_HOD' in actions:

        result = collections.OrderedDict()

        model2 = Model(
            Omega_m=0.258, Omega_de=0.742, H0=72.0, hod=1,
            massDef="M200m", concenDef="TJ03", hmfDef="T08", biasDef="T08")
        model2.hod = 0
        result['change_HOD'] = C_HALOMODEL.changeModelHOD(model, model2)
        _write_or_check(model2, 'change_HOD', result, computeRef)

        del model2

    # HOD's N(Mh)
    if 'Ngal' in actions:

        result = collections.OrderedDict()

        result['log10Mh'] = np.linspace(10.0, 15.0, 100)
        result['N'] = Ngal(model, result['log10Mh'], 10.0, 11.0, obs_type='all')
        _write_or_check(model, 'Ngal', result, computeRef, qty_to_check='N')

    # Stellar mass halo mass
    if 'MsMh' in actions:

        result = collections.OrderedDict()

        result['log10Mh'] = np.linspace(10.0, 15.0, 100)
        result['log10Mstar'] = msmh_log10Mstar(model, result['log10Mh'])
        _write_or_check(
            model, 'MsMh', result, computeRef, qty_to_check='log10Mstar')

    # halo concentration relationship
    if 'concen' in actions:

        result = collections.OrderedDict()

        result['concentration'] = concentration(
            model, 1.e14, z, concenDef="TJ03")
        _write_or_check(model, 'concentration', result, computeRef)

    # mass conversion
    if 'mass_conv' in actions:

        result = collections.OrderedDict()

        result['mass_conv'] = log10M1_to_log10M2(
            model, 13.0, None, "M200m", "M500c", z)
        _write_or_check(model, 'mass_conv', result, computeRef)

    # matter two-point
    if 'xi_dm' in actions:

        result = collections.OrderedDict()

        result['r'] = pow(10.0, np.linspace(
            np.log10(2.e-3), np.log10(2.0e2), 100))
        result['xi_dm'] = xi_dm(model, result['r'], z)
        _write_or_check(model, 'xi_dm', result, computeRef)

    # Fourrier transform of halo profile
    if 'uHalo' in actions:

        result = collections.OrderedDict()

        result['k'] =  pow(10.0, np.linspace(
            np.log10(2.e-3), np.log10(1.e4), 100))
        result['uHalo'] = np.asarray(
            np.zeros(len(result['k'])), dtype=np.float64)
        result['uHaloAnalytic'] = np.asarray(
            np.zeros(len(result['k'])), dtype=np.float64)
        for i in range(len(result['k'])):
            result['uHalo'][i] = C_HALOMODEL.uHalo(
                model, result['k'][i], 1.e14, np.nan, z)
            result['uHaloAnalytic'][i]  = C_HALOMODEL.uHaloClosedFormula(
                model, result['k'][i], 1.e14, np.nan, z)
        _write_or_check(model, 'uHalo', result, computeRef)

    # stellar mass function
    if 'smf' in actions:

        result = collections.OrderedDict()

        result['log10Mstar'] = np.linspace(9.0, 12.0, 100)
        result['smf'] = dndlog10Mstar(
            model, result['log10Mstar'], z, obs_type="all")
        _write_or_check(model, 'smf', result, computeRef)

    # Galaxy-galaxy lensing, HOD model
    if 'ggl_HOD' in actions:

        result = collections.OrderedDict()

        result['R'] = pow(10.0, np.linspace(3.0, 2.0, 100))
        result['ggl_HOD'] = DeltaSigma(model, result['R'], z, obs_type="all")
        result['star'] = DeltaSigma(model, result['R'], z, obs_type="star")
        result['cen'] = DeltaSigma(model, result['R'], z, obs_type="cen")
        result['sat'] = DeltaSigma(model, result['R'], z, obs_type="sat")
        result['twohalo'] = DeltaSigma(
            model, result['R'], z, obs_type="twohalo")
        _write_or_check(model, 'ggl_HOD', result, computeRef)

    # Galaxy-galaxy lensing, no HOD
    if "ggl" in actions:

        result = collections.OrderedDict()

        modelNoHOD = Model(
            Omega_m=0.258, Omega_de=0.742, H0=72.0,
            Omega_b = 0.0441, sigma_8 = 0.796, n_s = 0.963, hod=0,
            massDef="M200m", concenDef="TJ03", hmfDef="T08", biasDef="T08")
        modelNoHOD.ggl_log10Mh = 13.4
        modelNoHOD.ggl_log10c = 0.69
        modelNoHOD.ggl_log10Mstar = 11.0

        result['R'] = pow(10.0, np.linspace(3.0, 2.0, 100))
        result['ggl'] = DeltaSigma(model, result['R'], z, obs_type="all")
        result['star'] = DeltaSigma(model, result['R'], z, obs_type="star")
        result['cen'] = DeltaSigma(model, result['R'], z, obs_type="cen")
        result['sat'] = DeltaSigma(model, result['R'], z, obs_type="sat")
        result['twohalo'] = DeltaSigma(
            model, result['R'], z, obs_type="twohalo")
        _write_or_check(modelNoHOD, 'ggl', result, computeRef)

        del modelNoHOD

    # W(theta) HOD model
    if "wtheta_HOD" in actions:

        result = collections.OrderedDict()

        loadWtheta_nz(model, HALOMODEL_DIRNAME+"/data/wtheta_nz.ascii")
        result['theta'] = pow(10.0, np.linspace(-3.0, 2.0, 100))
        result['wtheta_HOD'] = wOfTheta(
            model, result['theta'], z, obs_type="all")
        result['censat'] = wOfTheta(
            model, result['theta'], z, obs_type="censat")
        result['satsat'] = wOfTheta(
            model, result['theta'], z, obs_type="satsat")
        result['twohalo'] = wOfTheta(
            model, result['theta'], z, obs_type="twohalo")
        _write_or_check(model, 'wtheta_HOD', result, computeRef)

    # populate halos with halo catalogue
    if 'populate' in actions:

        # DEBUGGING
        """
        func_cum_prob_HOD(
            model, log10Mstar_min = 5.0, obs_type="cen", plot=True)
        func_cum_prob_rho_halo(model, z, log10Mstar_min = 5.0, plot=True)
        """

        # options
        log10Mstar_min = 7.0
        np.random.seed(seed = 2009182)

        halo_file_name = HALOMODEL_DIRNAME+'/data/halos_z0.38.fits'

        with fits.open(halo_file_name) as tbhdu:
            halos = tbhdu[1].data[:10]
        result = populate(model, log10Mstar_min, halos, z, verbose=False)
        _write_or_check(
            model, 'populate', result, computeRef, qty_to_check='log10Mstar')

    # X-ray cooling function
    if "Lambda" in actions:

        result = collections.OrderedDict()

        result['TGas'] = pow(10.0, np.linspace(
            np.log10(1.01e-1), np.log10(0.8e1), 100))
        result['Lambda'] = LambdaBolo(result['TGas'], 0.15)
        result['Lambda_0_00'] = LambdaBolo(result['TGas'], 0.00)
        result['Lambda_0_40'] = LambdaBolo(result['TGas'], 0.40)

        _write_or_check(model, 'Lambda', result, computeRef)

    # CR to Lx conversion
    if "LxToCR" in actions:

        result = collections.OrderedDict()

        result['TGas'] = pow(10.0, np.linspace(
            np.log10(1.01e-1), np.log10(0.8e1), 100))
        TGas = pow(10.0, np.linspace(np.log10(1.01e-1), np.log10(1.e1), 100))
        result['LxToCR'] = LxToCR(model, result['TGas'], 0.15)
        result['LxToCR_0_00'] = LxToCR(model, result['TGas'], 0.00)
        result['LxToCR_0_40'] = LxToCR(model, result['TGas'], 0.40)

        _write_or_check(model, 'LxToCR', result, computeRef)
        # formats={'LxToCR_0_00':'%.8g', 'LxToCR_0_15':'%.8g', 'LxToCR_0_40':'%.8g'}

    # Fourrier transform of X-ray profile
    if "uIx" in actions:

        result = collections.OrderedDict()

        result['k'] = pow(10.0, np.linspace(
            np.log10(2.e-3), np.log10(1.e4), 100))
        result['uIx'] = np.asarray(np.zeros(len(result['k'])), dtype=np.float64)
        for i in range(len(k)):
            result['uIx'][i] = C_HALOMODEL.uIx(model, k[i], 1.e14, np.nan, z)

        _write_or_check(model, 'uIx', result, computeRef)

    # X-ray projected profile, HOD model
    if "SigmaIx_HOD" in actions:

        result = collections.OrderedDict()

        model.IxXB_Re = 0.01196
        model.IxXB_CR = 6.56997872802e-05

        result['theta'] = pow(10.0, np.linspace(-4.0, 5.0, 100))
        result['SigmaIx_HOD'] = SigmaIx(
            model, result['theta'], np.nan, np.nan, z, obs_type="all", PSF=None)
        result['cen'] = SigmaIx(
            model, result['theta'], np.nan, np.nan, z, obs_type="cen", PSF=None)
        result['sat'] = SigmaIx(
            model, result['theta'], np.nan, np.nan, z, obs_type="sat", PSF=None)
        result['XB'] = SigmaIx(
            model, result['theta'] , np.nan, np.nan, z, obs_type="XB", PSF=None)
        result['twohalo'] = SigmaIx(
            model, result['theta'] , np.nan, np.nan, z,
            obs_type="twohalo", PSF=None)

        _write_or_check(model, 'SigmaIx_HOD', result, computeRef)

    # X-ray projected profile no HOD model
    if "SigmaIx" in actions:

        result = collections.OrderedDict()

        Mh = 1.e14
        c = np.nan
        PSF = [0.00211586211541, 1.851542]

        modelNoHOD = Model(
            Omega_m=0.258, Omega_de=0.742, H0=72.0, Omega_b = 0.0441,
            sigma_8 = 0.796, n_s = 0.963, hod=0, massDef="M200m",
            concenDef="TJ03", hmfDef="T08", biasDef="T08")

        result['R500'] = C_HALOMODEL.rh(
            modelNoHOD, Mh, Delta(model, z, "M500c"), z)

        modelNoHOD.gas_log10n0 = np.log10(5.3e-3)
        modelNoHOD.gas_log10beta = np.log10(0.40)
        modelNoHOD.gas_log10rc = np.log10(0.03*result['R500'])
        modelNoHOD.IxXB_Re = 0.01196
        modelNoHOD.IxXB_CR = 6.56997872802e-05

        result['theta'] = pow(10.0, np.linspace(-4.0, 2,0, 100))
        result['SigmaIx'] = SigmaIx(
            model, result['theta'], Mh, c, z, obs_type="all", PSF=None)
        result['cen'] = SigmaIx(
            model, result['theta'], Mh, c, z, obs_type="cen", PSF=PSF)
        result['sat'] = SigmaIx(
            model, result['theta'], Mh, c, z, obs_type="sat", PSF=None)
        result['XB'] = SigmaIx(
            model, result['theta'], Mh, c, z, obs_type="XB", PSF=None)
        result['twohalo'] = SigmaIx(
            model, result['theta'], Mh, c, z, obs_type="twohalo", PSF=None)

        _write_or_check(modelHOD_nonPara, 'SigmaIx', result, computeRef)

    # X-ray projected profile, HOD model
    if "SigmaIx_HOD_nonPara" in actions:

        result = collections.OrderedDict()

        modelHOD_nonPara = Model(Omega_m=0.258, Omega_de=0.742, H0=72.0,
            Omega_b = 0.0441, sigma_8 = 0.796, n_s = 0.963,
            hod=0, massDef="M200m", concenDef="TJ03",
            hmfDef="T08", biasDef="T08")

        """ set non parametric HODs
        """
        cen = ascii.read(HALOMODEL_DIRNAME
            +"/data/HOD_0.20_0.35_cen_M200m_Mstar_11.30_11.45.ascii",
            format="no_header")

        # TODO correct below
        # cen["col1"] += 1.0*model.log10h

        cen["col1"] += 2.0*model.log10h
        modelHOD_nonPara.HOD_cen_N = len(cen)
        modelHOD_nonPara.HOD_cen_log10Mh = np.ctypeslib.as_ctypes(cen["col1"])
        modelHOD_nonPara.HOD_cen_Ngal = np.ctypeslib.as_ctypes(cen["col2"])

        sat = ascii.read(HALOMODEL_DIRNAME
            +"/data/HOD_0.20_0.35_sat_M200m_Mstar_11.30_11.45.ascii",
            format="no_header")

        # TODO correct below
        # sat["col1"] += 1.0*model.log10h

        sat["col1"] += 2.0*model.log10h
        modelHOD_nonPara.HOD_sat_N = len(sat)
        modelHOD_nonPara.HOD_sat_log10Mh = np.ctypeslib.as_ctypes(sat["col1"])
        modelHOD_nonPara.HOD_sat_Ngal = np.ctypeslib.as_ctypes(sat["col2"])

        modelHOD_nonPara.IxXB_Re = -1.0
        modelHOD_nonPara.IxXB_CR = -1.0

        result['theta'] = pow(10.0, np.linspace(-4.0, 5.0, 100))
        result['SigmaIx_HOD'] = SigmaIx(
            modelHOD_nonPara, result['theta'], np.nan,
            np.nan, z, obs_type="all", PSF=None)
        result['cen'] = SigmaIx(
            modelHOD_nonPara, result['theta'], np.nan,
            np.nan, z, obs_type="cen", PSF=None)
        result['sat'] = SigmaIx(
            modelHOD_nonPara, result['theta'], np.nan,
            np.nan, z, obs_type="sat", PSF=None)
        result['XB'] = SigmaIx(
            modelHOD_nonPara, result['theta'], np.nan,
            np.nan, z, obs_type="XB", PSF=None)
        result['twohalo'] = SigmaIx(modelHOD_nonPara,
            result['theta'], np.nan, np.nan, z, obs_type="twohalo", PSF=None)

        _write_or_check(modelHOD_nonPara, 'SigmaIx_HOD', result, computeRef)

        del modelHOD_nonPara


    # sanity check: the model should not have changed
    m2 = dumpModel(model)
    if printModelChanges:
        if m1 != m2:
            sys.stderr.write("Changes in the model:\n")
            import difflib
            m1 = m1.splitlines(1)
            m2 = m2.splitlines(1)
            diff = difflib.unified_diff(m1, m2)
            sys.stderr.write(''.join(diff))

    return

def _write_or_check(
        model, action, result, computeRef, decimal=6, qty_to_check=None):
    """ Write or check outputs of the test() function.
    """

    if qty_to_check is None:
        qty_to_check = action

    # how the messages should appear on screen
    OK_MESSAGE = "OK\n"
    FAIL_MESSAGE = "FAILED\n"
    DONE_MESSAGE = "DONE\n"

    for k in result:
        if isinstance(result[k], (int, float)):
            result[k] = [result[k]]

    fileOutName = HALOMODEL_DIRNAME+'/data/tests/'+action+'_ref.ascii'
    if computeRef:
        sys.stderr.write('Computing reference for '+action+':')
        ascii.write(
            result, fileOutName, format="commented_header", overwrite=True)
        dumpModel(model, fileOutName=fileOutName)
        sys.stderr.write(bcolors.OKGREEN+DONE_MESSAGE+bcolors.ENDC)
    else:
        sys.stderr.write(action+':')
        ref = ascii.read(
            fileOutName, format="commented_header", header_start=-1)
        try:
            np.testing.assert_array_almost_equal(
                result[qty_to_check], ref[qty_to_check], err_msg='in '+action,
                decimal=decimal)
        except:
            sys.stderr.write(bcolors.FAIL+FAIL_MESSAGE+bcolors.ENDC)
            traceback.print_exc()
        else:
            sys.stderr.write(bcolors.OKGREEN+OK_MESSAGE+bcolors.ENDC)

    return



# def fitBetaPara(args):
#
#
#     from astropy.io import ascii
#     from scipy.optimize import curve_fit
#
#     C_HALOMODEL.MhToTGas.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double]
#     C_HALOMODEL.MhToTGas.restype = ctypes.c_double
#
#     C_HALOMODEL.TGasToMh.argtypes = [ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double]
#     C_HALOMODEL.TGasToMh.restype = ctypes.c_double
#
#     if False:
#         data = ascii.read("/Users/coupon/projects/Stacked_X_ray/info/betaProfilesEckert2015.ascii", header_start=-1)
#
#         x = [ np.log10(C_HALOMODEL.TGasToMh(model, TGas, z)) for TGas in data['TGas'] ]
#
#         print x
#
#         R500 = [C_HALOMODEL.rh(model, pow(10.0, log10Mh), np.nan, z) for log10Mh in x]
#
#         data["rc"] = data["RcOverR500"]*R500
#         data["rc_err"] = data["RcOverR500_err"]*R500
#
#         para = 'rc'
#         y    = [ np.log10(p) for p in data[para] ]
#         yerr = [ p_err/p / np.log(10.0) for (p, p_err) in zip(data[para],data[para+'_err']) ]
#
#         p, pCov = curve_fit(powerLaw,x, y, sigma=yerr )
#         print p
#
#         return
#
#
# def powerLaw(x, a, b):
#         return a + b*(x-14.0)
#

"""

-------------------------------------------------------------
main functions
-------------------------------------------------------------

"""


def populate(
        model, halos, redshift, verbose = False, Mh_limits=None, sat=True,
        log10Mh_name='log10Mh', log10Mstar_low=5.0, log10Mstar_high=12.5):

    """ Populate halos with galaxies from input
    halo catalogue and HOD (set in model).

    INPUT
    model: cosmological and HOD model
    halos: dictionary with the following keys:
        - 'log10Mh': log halo mass in h^-1 Msun
        - 'x': x coordinate
        - 'y': y coordinate
        - 'z': z coordinate
    redshift: mean redshift
    Mh_limits: [log10Mh_min, log10Mh_max], halo
    mass limits over which HOD is populated
    log10Mstar_low: lowest stellar mass limit

    OUTPUT
    dictionary with coordinates, masses and types
    (central or not) of galaxies: x,y,z,log10Mstar,cen

    log10Mstar in log10(Mstar/[h^-2 Msun]]
    log10Mh in log10(Mstar/[h^-1 Msun]]
    """

    # dictionary to be output
    galaxies = collections.OrderedDict({
        'log10Mstar': [], 'x': [], 'y': [], 'z': [],
        'cen': [], 'log10Mh': []
        })

    # interpolate DM profiles (for satellites)
    cum_prob_rho_Halo = func_cum_prob_rho_halo(
        model, redshift, log10Mstar_low = log10Mstar_low)

    Nh = len(halos['log10Mh'])

    if verbose:
        sys.stdout.write('\rPopulated {0:d}/{1:d} halos'.format(0, Nh))
        sys.stdout.flush()

    # loop over halos
    for count, (log10Mh, x, y, z) in enumerate(
        zip(halos[log10Mh_name], halos['x'], halos['y'], halos['z'])):

        # centrals
        galaxies['log10Mstar'].append(
            draw_log10Mstar(model, log10Mh, obs_type='cen')[0])
        galaxies['x'].append(x)
        galaxies['y'].append(y)
        galaxies['z'].append(z)
        galaxies['cen'].append(1)
        galaxies['log10Mh'].append(log10Mh)

        # number of satellites
        if sat:
            log10Mstar = draw_log10Mstar(
                model, log10Mh, obs_type='sat',
                log10Mstar_low=log10Mstar_low,
                log10Mstar_high=log10Mstar_high)
            Nsat = len(log10Mstar)
            if Nsat > 0:

                # print(Nsat)

                # radius and angle
                r = pow(10.0, cum_prob_rho_Halo(
                    log10Mh, np.random.rand(Nsat))).flatten()
                ra = np.random.rand(Nsat)*2.0*np.pi
                dec = np.random.rand(Nsat)*2.0-1.0
                dec = np.arcsin(dec)

                # set galaxies
                for i in range(Nsat):
                    galaxies['log10Mstar'].append(log10Mstar[i])

                    galaxies['x'].append(x+r[i]*np.cos(ra[i])*np.cos(dec[i]))
                    galaxies['y'].append(y+r[i]*np.sin(ra[i])*np.cos(dec[i]))
                    galaxies['z'].append(z+r[i]*np.sin(-dec[i]))
                    galaxies['cen'].append(0)
                    galaxies['log10Mh'].append(log10Mh)

        if verbose:
            if (count+1)%1000 == 0:
                sys.stdout.write(
                    '\rPopulated {0:d}/{1:d} halos'.format(count+1, Nh))
                sys.stdout.flush()

    if verbose:
        sys.stdout.write('\rPopulated {0:d}/{1:d} halos\n'.format(Nh, Nh))

    # convert to numpy arrays
    for k in galaxies:
        galaxies[k] = np.array(galaxies[k])

    return galaxies




def populate2(
        model, halos, redshift, verbose = False, log10Mstar_low = 5.0,
        Mh_limits=None, log10Mh_name='log10Mh', sat=True):

    """ Populate halos with galaxies from input
    halo catalogue and HOD (set in model).

    INPUT
    model: cosmological and HOD model
    halos: dictionary with the following keys:
        - 'log10Mh': log halo mass in h^-1 Msun
        - 'x': x coordinate
        - 'y': y coordinate
        - 'z': z coordinate
    redshift: mean redshift
    Mh_limits: [log10Mh_min, log10Mh_max], halo
    mass limits over which HOD is populated
    log10Mstar_low: lowest stellar mass limit

    OUTPUT
    dictionary with coordinates, masses and types
    (central or not) of galaxies: x,y,z,log10Mstar,cen

    log10Mstar in log10(Mstar/[h^-2 Msun]]
    log10Mh in log10(Mstar/[h^-1 Msun]]
    """

    # dictionary to be output
    galaxies = collections.OrderedDict({
        'log10Mstar': [], 'x': [], 'y': [], 'z': [],
        'cen': [], 'log10Mh': []
        })

    # interpolate phi(log10Mstar|log10Mh)
    cum_prob_HOD_cen = func_cum_prob_HOD(
        model, log10Mstar_low = log10Mstar_low,
        obs_type="cen", Mh_limits=Mh_limits)
    cum_prob_HOD_sat = func_cum_prob_HOD(
        model, log10Mstar_low = log10Mstar_low,
        obs_type="sat", Mh_limits=Mh_limits)

    # interpolate DM profiles (for satellites)
    cum_prob_rho_Halo = func_cum_prob_rho_halo(
        model, redshift, log10Mstar_low = log10Mstar_low)

    Nh = len(halos['log10Mh'])

    if verbose:
        sys.stdout.write('\rPopulated {0:d}/{1:d} halos'.format(0, Nh))
        sys.stdout.flush()

    # loop over halos
    for count, (log10Mh, x, y, z) in enumerate(
        zip(halos[log10Mh_name], halos['x'], halos['y'], halos['z'])):

        # centrals
        galaxies['log10Mstar'].append(cum_prob_HOD_cen(
            log10Mh, np.random.rand())[0])
        galaxies['x'].append(x)
        galaxies['y'].append(y)
        galaxies['z'].append(z)
        galaxies['cen'].append(1)
        galaxies['log10Mh'].append(log10Mh)

        # number of satellites
        if sat:
            Nsat = np.random.poisson(
                Ngal(model, log10Mh, log10Mstar_low, -1.0, obs_type='sat'))
            if Nsat > 0:

                # Mstar distribution within halo
                log10Mstar = cum_prob_HOD_sat(
                    log10Mh, np.random.rand(Nsat)).flatten()

                # radius and angle
                r = pow(10.0, cum_prob_rho_Halo(
                    log10Mh, np.random.rand(Nsat))).flatten()
                ra = np.random.rand(Nsat)*2.0*np.pi
                dec = np.random.rand(Nsat)*2.0-1.0
                dec = np.arcsin(dec)

                # set galaxies
                for i in range(Nsat):
                    galaxies['log10Mstar'].append(log10Mstar[i])
                    galaxies['x'].append(x+r[i]*np.cos(ra[i])*np.cos(dec[i]))
                    galaxies['y'].append(y+r[i]*np.sin(ra[i])*np.cos(dec[i]))
                    galaxies['z'].append(z+r[i]*np.sin(-dec[i]))
                    galaxies['cen'].append(0)
                    galaxies['log10Mh'].append(log10Mh)

        if verbose:
            if (count+1)%1000 == 0:
                sys.stdout.write(
                    '\rPopulated {0:d}/{1:d} halos'.format(count+1, Nh))
                sys.stdout.flush()

    if verbose:
        sys.stdout.write('\rPopulated {0:d}/{1:d} halos\n'.format(Nh, Nh))

    # convert to numpy arrays
    for k in galaxies:
        galaxies[k] = np.array(galaxies[k])

    return galaxies


def func_cum_prob_HOD(
        model, obs_type='cen',
        plot=None, Mh_limits=None, return_prob=False):

    """ Return a function to interpolate
    cumulative HOD (Mstar_inv).

    For a given log10Mh and random number
    number between 0 and 1, this function then
    returns a log10Mstar with the probability
    given by the HOD model.

    The Mh_limits is not given, the minimum halo mass
    is deduced from the Mstar-Mh relation (with +0.2
    added to avoid truncated gaussian in Mstar at
    given Mh)

    return_prob: if True, returns the probability
    instead of the cumulative one.

    """

    # Mstar grid
    log10Mstar_low = 5.0
    log10Mstar = np.concatenate((
        np.linspace(log10Mstar_low, 11.00, 200),
        np.linspace(11.00, 11.50, 200),
        np.linspace(11.50, 12.50, 200)))

    # DEBUGGING
    log10Mstar = np.linspace(log10Mstar_low, 12.50, 200)

    Nlog10Mstar = len(log10Mstar)

    # Mh grid
    Nlog10Mh = 100

    if Mh_limits is not None:
        log10Mh = np.linspace(Mh_limits[0], Mh_limits[1], Nlog10Mh)
    else:
        log10Mh = np.linspace(
            msmh_log10Mh(model, log10Mstar[0])+0.2, 16.0, Nlog10Mh)

    uniform = np.linspace(0.0, 1.0, Nlog10Mstar)

    # cumulative sums and 2D probability containers
    cs_2D = np.zeros(Nlog10Mh*Nlog10Mstar)
    cs_2D_array = np.zeros((Nlog10Mh,Nlog10Mstar))
    prob_2D = np.zeros(Nlog10Mh*Nlog10Mstar)

    prob_2D_array = np.zeros((Nlog10Mh,Nlog10Mstar))
    cs = np.zeros(Nlog10Mstar)

    # loop over halo masses
    for i,m in enumerate(log10Mh):

        # probability of Mstar given Mh
        prob = phi(model, log10Mstar, m, obs_type=obs_type)

        sum_prob = np.trapz(prob, log10Mstar)

        # skip those halos with too few satellites
        # or too low probability
        if sum_prob > 0.0:

            # commulative function
            cs[1:] = np.cumsum(np.diff(log10Mstar)*(prob[:-1]+prob[1:])/2.0)

            # normalise
            cs /= max(cs)
            prob /= sum_prob

            cs_2D[i*Nlog10Mstar:(i+1)*Nlog10Mstar] = np.interp(
                uniform, cs, log10Mstar)

            cs_2D_array[i, :] = np.interp(uniform, cs, log10Mstar)

            prob_2D[i*Nlog10Mstar:(i+1)*Nlog10Mstar] = prob

            prob_2D_array[i,:] = prob

        # DEBUGGING
        """
        prob_2D[i,:] = cs/max(cs)
        prob_2D[i:] = np.interp(
            uniform, cs/max(cs), log10Mstar,  left=0.0, right=1.0)
        prob_2D[i,:] = prob
        """

    if plot is not None:

        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        im = ax.imshow(
            prob_2D_array, cmap=plt.cm.viridis,
            interpolation='nearest', origin='lower')

        x_int = Nlog10Mstar//10
        y_int = Nlog10Mh//10

        xticks = np.arange(0, Nlog10Mstar, x_int)
        xticklabels = ["{0:4.2f}".format(b) for b in log10Mstar[xticks]]

        yticks = np.arange(0, Nlog10Mh, y_int)
        yticklabels = ["{0:4.2f}".format(b) for b in log10Mh[yticks]]

        ax.xaxis.set_ticks(xticks)
        ax.xaxis.set_ticklabels(xticklabels, rotation=45)

        ax.yaxis.set_ticks(yticks)
        ax.yaxis.set_ticklabels(yticklabels)

        ax.set_xlabel('log10Mstar')
        ax.set_ylabel('log10Mh')

        fig.set_tight_layout(True)
        fig.savefig(plot)

    if return_prob:
        # return log10Mh, log10Mstar, prob_2D

        X, Y = np.meshgrid(log10Mh, log10Mstar)
        return X.flatten(), Y.flatten(), prob_2D.flatten()
        # return X, Y, prob_2D

        #func = interpolate.interp2d(X, Y, prob_2D, kind='linear')

        # func = interpolate.interp2d(log10Mh, log10Mstar, prob_2D, kind='linear')
    else:
        return interpolate.interp2d(log10Mh, uniform, cs_2D, kind='linear')

def func_cum_prob_rho_halo(model, z, log10Mstar_low = 5.0, plot=None):
    """ Return function to interpolate
    cumulative halo profile probability
    to populate satellites in halos.

    For a given log10Mh and random number
    number between 0 and 1, this function then
    returns a position with the probability
    given by the dark matter halo profile.

    This routine needs a redshift for the
    calculation of the maximum radius
    of the halos
    """

    # dimensions in both directions
    Nlog10r = 1000
    Nlog10Mh = 100

    # grids
    log10Mh = np.linspace(msmh_log10Mh(model, log10Mstar_low), 16.0, Nlog10Mh)
    log10r = np.linspace(
        -3.0, np.log10(rh(model, pow(10.0, log10Mh[-1]), z)), Nlog10r)
    log10r_inv = np.linspace(0.0, 1.0, Nlog10r)

    r = pow(10.0, log10r)

    cs_2D = np.zeros(Nlog10Mh*Nlog10r)
    prob_2D = np.zeros((Nlog10Mh,Nlog10r))
    cs = np.zeros(Nlog10r)

    for i,m in enumerate(log10Mh):

        # normalised probability of position
        # (= normalised NFW profile x r^2 x 4 pi x r
        # [because log scale])
        prob = rhoHalo(model, r, m, z) \
            /pow(10.0, m) * r**2 * 4.0 * np.pi * r * np.log(10.0)

        # commulative function
        cs[1:] = np.cumsum(np.diff(log10r)*(prob[:-1]+prob[1:])/2.0)

        # renormalise in case
        # probability goes beyond limits
        cs /= max(cs)

        # inverse cumulative function
        select = prob > 1.e-8
        cs_2D[i*Nlog10r:(i+1)*Nlog10r] = np.interp(
            log10r_inv, cs[select], log10r[select])

        prob_2D[i,:] = cs # /max(cs)

        # DEBUGGING
        """
        prob_2D[i,:] = prob
        prob_2D[i:] = np.interp(
            uniform, cs/max(cs), log10Mstar,  left=0.0, right=1.0)
        """

    if plot is not None:
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        im = ax.imshow(
            prob_2D, cmap=plt.cm.viridis,
            interpolation='nearest', origin='lower')

        x_int = Nlog10r//10
        y_int = Nlog10Mh//10

        xticks = np.arange(0, Nlog10r, x_int)
        xticklabels = ["{0:4.2f}".format(b) for b in log10r[xticks]]

        yticks = np.arange(0, Nlog10Mh, y_int)
        yticklabels = ["{0:4.2f}".format(b) for b in log10Mh[yticks]]

        ax.xaxis.set_ticks(xticks)
        ax.xaxis.set_ticklabels(xticklabels, rotation=45)

        ax.yaxis.set_ticks(yticks)
        ax.yaxis.set_ticklabels(yticklabels)

        ax.set_xlabel('log10r')
        ax.set_ylabel('log10Mh')

        fig.set_tight_layout(True)
        fig.savefig(plot)

    return interpolate.interp2d(log10Mh, log10r_inv, cs_2D, kind='linear')

def draw_log10Mstar(
        model, log10Mh, obs_type='cen',
        log10Mstar_low=5.0, log10Mstar_high=12.5):
    """ Draw log10(Mstar) value at given halo mass
    """

    if log10Mstar_low > 11.0:
        raise ValueError(
            'draw_log10Mstar: log10Mstar_low must be larger than 11.0')

    if log10Mstar_high < 11.5:
        raise ValueError(
            'draw_log10Mstar: log10Mstar_high must be higher than 11.5')

    # split log10Mstar range so that high-mass values are well represented
    log10Mstar = np.concatenate((
        np.linspace(log10Mstar_low, 11.0, 20),
        np.linspace(11.0, 11.5, 20),
        np.linspace(11.5, 12.50, 20)))
    Nlog10Mstar = len(log10Mstar)

    # phi(Mh) = probability of Mstar given a halo mass
    P = phi(model, log10Mstar, log10Mh, obs_type=obs_type)

    # number of values to draw
    if obs_type == 'cen':
        N = 1
    elif obs_type == 'sat':
        N = np.random.poisson(Ngal(
            model, log10Mh, log10Mstar_low, log10Mstar_high, obs_type='sat'))
    else:
        raise ValueError(
            "draw_log10Mstar: obs_type \"{0:s}\" is not recognised".format(
                obs_type))

    return draw_dist(log10Mstar, P, N)


def draw_dist(x, P, N):
    """ Draw values at random from distribution
    """

    # return empty array
    if N == 0:
        return np.array([])

    N_bins = len(x)

    # cumulative sum
    cs = np.zeros(N_bins)
    # cs = np.cumsum(phi_sat)
    cs[1:] = np.cumsum(np.diff(x)*(P[:-1]+P[1:])/2.0)

    # normalise
    cs /= max(cs)

    # inverse distribution
    x_inv = interpolate.interp1d(cs, x)

    return x_inv(np.random.rand(N))


# c-function prototype
C_HALOMODEL.rhoHalo.argtypes = [
        ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double
        ]
C_HALOMODEL.rhoHalo.restype = ctypes.c_double
def rhoHalo(model, r, log10Mh, z, c=None):
    """ Wrapper for c-function rhoHalo

    Returns the dark matter density profile
    (default: NFW profile)

    ** Mpc in comoving units **

    INPUT
    r: distance in [h^-1 Mpc]
    log10Mh: log halo mass. Halo mass in h^-1 Mpc
    c: concentration, default: None (will assume
    mass-concentration relationship)

    OUTPUT
    rho Halo evaluated at r
    """

    if c is None:
        c = np.nan

    Mh = pow(10.0, log10Mh)

    if isinstance(r, (list, tuple, np.ndarray)):
        r = np.asarray(r, dtype=np.float64)
        result = np.asarray(np.zeros(len(r)), dtype=np.float64)
        for i in range(len(r)):
            result[i] = C_HALOMODEL.rhoHalo(model, r[i], Mh, c, z)
    else:
        result = C_HALOMODEL.rhoHalo(model, r, Mh, c, z)

    return result


def nGas(model, r, log10Mh, c, z):
    """ Wrapper for c-function nGas()

    Returns the gas profile density

    ** Mpc in comoving units **

    INPUT
    r: distance in h^-1 Mpc

    OUTPUT
    nGas evaluated at r
    """

    Mh = pow(10.0, log10Mh)

    if isinstance(r, (list, tuple, np.ndarray)):
        r = np.asarray(r, dtype=np.float64)
        result = np.asarray(np.zeros(len(r)), dtype=np.float64)
        for i in range(len(r)):
            result[i] = C_HALOMODEL.nGas(model, r[i], Mh, c, z)
    else:
        result = C_HALOMODEL.nGas(model, r, Mh, c, z)

    return result


C_HALOMODEL.xi_m.argtypes = [
    ctypes.POINTER(Model), np.ctypeslib.ndpointer(dtype = np.float64),
    ctypes.c_int, ctypes.c_double,  np.ctypeslib.ndpointer(dtype = np.float64)
    ]
def xi_dm(model, r, z):
    """ Wrapper for c-function xi_dm()

    Return the dark matter two-point correlation function.

    ** Mpc in comoving units **

    INPUT
    r: distance in h^-1 Mpc

    OUTPUT
    xi_dm evaluated at r
    """

    r = np.asarray(r, dtype=np.float64)
    result = np.asarray(np.zeros(len(r)), dtype=np.float64)

    C_HALOMODEL.xi_m(model, r, len(r), z, result)

    return result


C_HALOMODEL.xi_m_lin.argtypes = [
    ctypes.POINTER(Model), np.ctypeslib.ndpointer(dtype = np.float64),
    ctypes.c_int, ctypes.c_double,  np.ctypeslib.ndpointer(dtype = np.float64)
    ]
def xi_dm_lin(model, r, z):
    """ Wrapper for c-function xi_dm_lin()

    Return the linear dark matter two-point correlation function.

    ** Mpc in comoving units **

    INPUT
    r: distance in h^-1 Mpc

    OUTPUT
    xi_dm evaluated at r
    """

    r = np.asarray(r, dtype=np.float64)
    result = np.asarray(np.zeros(len(r)), dtype=np.float64)

    C_HALOMODEL.xi_m_lin(model, r, len(r), z, result)

    return result


C_HALOMODEL.dndlog10Mstar.argtypes = [
    ctypes.POINTER(Model), np.ctypeslib.ndpointer(dtype = np.float64),
    ctypes.c_int, ctypes.c_double, ctypes.c_int,
    np.ctypeslib.ndpointer(dtype = np.float64)
    ]
def dndlog10Mstar(model, log10Mstar, z, obs_type="all"):
    """ Wrapper for c-function dndlog10Mstar()

    Returns the stellar mass function in units of (Mpc/h)^-3 dex^-1
    Mstar in [h^-2 Msun]

    ** volume in comoving units **

    INPUT
    log10Mstar: log10(Mstar) array or single value) in log10 Msun/h units
    z: redshift of the sample
    obs_type: [cen, sat, all]

    OUTPUT
    dndlog10Mstar evaluated at log10Mstar
    """

    log10Mstar = np.asarray(log10Mstar, dtype=np.float64)
    result = np.asarray(np.zeros(len(log10Mstar)), dtype=np.float64)

    if obs_type == "cen":
        obs_type = 1
    elif obs_type == "sat":
        obs_type = 2
    elif obs_type == "all":
        obs_type = 3
    else:
        raise ValueError(
            "dndlog10Mstar: obs_type \"{0:s}\" is not recognised".format(obs_type))

    C_HALOMODEL.dndlog10Mstar(model, log10Mstar, len(log10Mstar), z, obs_type, result)

    return result


C_HALOMODEL.dndlnMh.argtypes = [
    ctypes.POINTER(Model), ctypes.c_double, ctypes.c_double
    ]
C_HALOMODEL.dndlnMh.restype = ctypes.c_double
def dndlnMh(model, log10Mh, z):
    """ Wrapper for c-function dndlnMh().

    Return the stellar mass function in units of (Mpc/h)^-3 dex^-1
    Mh in [h^-1 Msun]

    ** volume in comoving units **

    INPUT
    log10Mh: log10(Mh) array or single value) in log10 Msun/h units
    z: redshift of the sample

    OUTPUT
    dndlnMstar evaluated at log10Mh. Multiply by ln(10)
    to get dndlog10Mstar

    """

    Mh = np.asarray(pow(10.0, log10Mh), dtype=np.float64)
    result = np.asarray(np.zeros(len(Mh)), dtype=np.float64)

    for i, m in enumerate(Mh):
        result[i] = C_HALOMODEL.dndlnMh(model, m,  z)

    return result


C_HALOMODEL.DeltaSigma.argtypes = [
    ctypes.POINTER(Model), np.ctypeslib.ndpointer(dtype = np.float64),
    ctypes.c_int, ctypes.c_double, ctypes.c_int,
    np.ctypeslib.ndpointer(dtype = np.float64)
    ]
def DeltaSigma(model, R, zl, obs_type="all"):
    """ Return DeltaSigma for NFW halo mass profile (in h Msun/pc^2) -
    PHYSICAL UNITS, unless como=True set.

    ** DS and R in comoving units **

    R_como  = R_phys * (1+z)
    DS_como = DS_phys / (1+z)^2

    INPUT
    R: (array or single value) in comoving units [h^-1 Mpc]
    zl: redshift of the lens
    obs_type: [cen, sat, twohalo, star, all], for "star"

    OUTPUT
    Delta Sigma (R)

    REFS: Sec. 7.5.1 of Mo, van den Bosh & White's Galaxy
    formation and evolution Wright & Brainerd (200) after
    correcting the typo in Eq. (7.141)
    """

    R = np.asarray(R, dtype=np.float64)
    result = np.asarray(np.zeros(len(R)), dtype=np.float64)

    if obs_type == "star":
        obs_type = 0
    elif obs_type == "cen":
        obs_type = 1
    elif obs_type == "sat":
        obs_type = 2
    elif obs_type == "twohalo":
        obs_type = 33
    elif obs_type == "all":
        obs_type = 3
    else:
        raise ValueError(
            "DeltaSigma: obs_type \"{0:s}\" is not recognised".format(obs_type))

    C_HALOMODEL.DeltaSigma(model, R, len(R), zl, obs_type, result)

    return result


def xi_gg(model, r, z, obs_type="all"):
    """ Returns xi_gg(r) for NFW halo mass profile -

    INPUT PARAMETERS:
    r: (array or single value) in Mpc/h
    z: mean redshift of the population

    OUTPUT
    xi(r)

    obs_type: [censat, satsat, twohalo, all]
    """

    r = np.asarray(r, dtype=np.float64)
    result = np.asarray(np.zeros(len(r)), dtype=np.float64)

    if obs_type == "censat":
        obs_type = 12
    elif obs_type == "satsat":
        obs_type = 22
    elif obs_type == "twohalo":
        obs_type = 33
    elif obs_type == "all":
        obs_type = 3
    else:
        raise ValueError(
            'xi(r): obs_type \"{0:s}\" is not recognised'.format(obs_type))

    C_HALOMODEL.xi_gg(model, r, len(r), z, obs_type, result)

    return result


C_HALOMODEL.wOfTheta.argtypes = [ctypes.POINTER(Model),
    np.ctypeslib.ndpointer(dtype = np.float64), ctypes.c_int,
    ctypes.c_double, ctypes.c_int, np.ctypeslib.ndpointer(dtype = np.float64)
    ]
def wOfTheta(model, theta, z, obs_type="all"):
    """ Return w(theta) for NFW halo mass profile.

    INPUT PARAMETERS:
    theta: (array or single value) in degree
    z: mean redshift of the population

    OUTPUT
    w(theta)

    obs_type: [censat, satsat, twohalo, all]
    """

    theta = np.asarray(theta, dtype=np.float64)
    result = np.asarray(np.zeros(len(theta)), dtype=np.float64)

    if obs_type == 'censat':
        obs_type = 12
    elif obs_type == 'satsat':
        obs_type = 22
    elif obs_type == 'twohalo':
        obs_type = 33
    elif obs_type == "all":
        obs_type = 3
    else:
        raise ValueError(
            'wOfTheta: obs_type "{0:s}" is not recognised'.format(obs_type))

    C_HALOMODEL.wOfTheta(model, theta, len(theta), z, obs_type, result)

    return result


C_HALOMODEL.wOfThetaFromXi.argtypes = [ctypes.POINTER(Model),
    np.ctypeslib.ndpointer(dtype = np.float64),
    ctypes.c_int, ctypes.c_double,
    np.ctypeslib.ndpointer(dtype = np.float64), ctypes.c_int,
    np.ctypeslib.ndpointer(dtype = np.float64),
    np.ctypeslib.ndpointer(dtype = np.float64)
    ]
def wOfThetaFromXi(model, theta, z, r, xi):
    """ Returns w(theta) from input xi and n(z)

    INPUT PARAMETERS:
    theta: (array or single value) in degree
    r: (array)in h^-1 Mpc
    xi: (array) 3D clustering

    OUTPUT
    w(theta)
    """

    theta = np.asarray(theta, dtype=np.float64)
    result = np.asarray(np.zeros(len(theta)), dtype=np.float64)

    r = np.asarray(r, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)

    C_HALOMODEL.wOfThetaFromXi(model, theta, len(theta), z, r, len(r), xi, result)

    return result




def SigmaIx(model, theta, Mh, c, z, obs_type="all", PSF=None):
    """ Wrapper for c-function SigmaIx()

    Returns the X-ray brightness profile in CR s^-1 deg^-2

    INPUT
    theta: (array or single value) in degree
    Mh: mass of the host halo (not used if HOD set)
    c: concentration of the host halo (not used if HOD set)
    z: redshift
    obs_type: [cen, sat, XB, all]
    PSF: normalised King's profile parameters

    OUTPUT
    SigmaIx(R)
    """

    theta = np.asarray(theta, dtype=np.float64)
    result = np.asarray(np.zeros(len(theta)), dtype=np.float64)

    if obs_type == "cen":
        obs_type = 1
    elif obs_type == "sat":
        obs_type = 2
    elif obs_type == "XB":
        obs_type = 4
    elif obs_type == "twohalo":
        obs_type = 33
    elif obs_type == "all":
        obs_type = 3
    else:
        raise ValueError("SigmaIx: obs_type \"{0:s}\" is not recognised".format(obs_type))

    # PSF in model
    if PSF is not None:
        # model.XMM_PSF_A = PSF[0]
        model.XMM_PSF_rc_deg = PSF[0]
        model.XMM_PSF_alpha = PSF[1]

    C_HALOMODEL.SigmaIx(model, theta, len(theta), Mh, c, z, obs_type, result)

    return result


def phi(model, log10Mstar, log10Mh, obs_type="cen"):
    """ Wrapper for c-functions "phi_c" and "phi_s"

    Returns the probability of Mstar at
    fixed Mh, phi(Mstar|Mh)

    INPUT
    log10Mh: log10(mass/[h^-1 Msun]) of the host halo
    log10Mstar: log10(mass/[h^-2 Msun]) stellar mass
    obs_type: [cen, sat]

    OUTPUT
    phi(log10Mstar|log10Mh), normalised to one
    """

    if isinstance(log10Mstar, (list, tuple, np.ndarray)):
        log10Mstar = np.asarray(log10Mstar, dtype=np.float64)
        result = np.asarray(np.zeros(len(log10Mstar)), dtype=np.float64)

        if obs_type == "cen":
            for i, m in enumerate(log10Mstar):
                result[i] = C_HALOMODEL.phi_c(model, m, log10Mh)
        elif obs_type == "sat":
            for i, m in enumerate(log10Mstar):
                result[i] = C_HALOMODEL.phi_s(model, m, log10Mh)
        else:
            raise ValueError("phi: obs_type \"{0:s}\" is not recognised".format(obs_type))

    else:
        if obs_type == "cen":
            result = C_HALOMODEL.phi_c(model, log10Mstar, log10Mh)
        elif obs_type == "sat":
            result = C_HALOMODEL.phi_s(model, log10Mstar, log10Mh)
        else:
            raise ValueError("phi: obs_type \"{0:s}\" is not recognised".format(obs_type))


    return result


def ngal_den(model, log10Mstar_min, log10Mstar_max, z, obs_type='all'):
    """  Wrapper for c-function ngal_den()

    Returns ngal_den = f(log10Mstar_min, log10Mstar_max) for a given Mstar-Mh relation.

    INPUT
    log10Mstar_min: lower stellar mass limit in units of log10[Mstar/[h^-2 Msun]
    log10Mstar_max: upper stellar mass limit in units of log10[Mstar/[h^-2 Msun]

    OUPUT
    ngal_den

    """

    lnMh_max = 37.99

    if obs_type == "cen":
        result = C_HALOMODEL.ngal_den(model, lnMh_max, log10Mstar_min, log10Mstar_max, z, 1)
    elif obs_type == "sat":
        result = C_HALOMODEL.ngal_den(model, lnMh_max, log10Mstar_min, log10Mstar_max, z, 2)
    elif obs_type == "all":
        result = C_HALOMODEL.ngal_den(model, lnMh_max, log10Mstar_min, log10Mstar_max, z, 3)
    else:
        raise ValueError("ngal_den: obs_type \"{0:s}\" is not recognised".format(obs_type))

    return result



def Ngal(model, log10Mh, log10Mstar_min, log10Mstar_max, obs_type="all"):
    """ Wrapper for c-functions "Ngal"

    Returns the HOD Ngal(Mh)

    INPUT
    log10Mh: log10(mass) of the host halo
    log10Mstar_min: lower stellar mass bin
    log10Mstar_max: upper stellar mass bin
    obs_type: [cen, sat, all]

    OUTPUT
    N(Mh)
    """

    if isinstance(log10Mh, (list, tuple, np.ndarray)):

        Mh = np.asarray(pow(10.0, log10Mh), dtype=np.float64)
        result = np.asarray(np.zeros(len(Mh)), dtype=np.float64)

        if obs_type == "cen":
            for i, m in enumerate(Mh):
                result[i] = C_HALOMODEL.Ngal_c(model, m,  log10Mstar_min, log10Mstar_max)
        elif obs_type == "sat":
            for i, m in enumerate(Mh):
                result[i] = C_HALOMODEL.Ngal_s(model, m,  log10Mstar_min, log10Mstar_max)
        elif obs_type == "all":
            for i, m in enumerate(Mh):
                result[i] = C_HALOMODEL.Ngal(model, m,  log10Mstar_min, log10Mstar_max)
        else:
            raise ValueError("Ngal: obs_type \"{0:s}\" is not recognised".format(obs_type))

    else:

        Mh = pow(10.0, log10Mh)

        if obs_type == "cen":
            result = C_HALOMODEL.Ngal_c(model, Mh,  log10Mstar_min, log10Mstar_max)
        elif obs_type == "sat":
            result = C_HALOMODEL.Ngal_s(model, Mh,  log10Mstar_min, log10Mstar_max)
        elif obs_type == "all":
            result = C_HALOMODEL.Ngal(model, Mh, log10Mstar_min, log10Mstar_max)
        else:
            raise ValueError("Ngal: obs_type \"{0:s}\" is not recognised".format(obs_type))


    return result


def shmr(model, log10Mh, log10Mstar_min, log10Mstar_max, obs_type="all"):
    """ Wrapper for shmr c-function

    Returns the stellar-to-halo mass ratio

    INPUT
    log10Mh: log10(mass) of the host halo
    log10Mstar_min: lower stellar mass bin
    log10Mstar_max: upper stellar mass bin
    obs_type: [cen, sat, all]

    OUTPUT
    ratio(Mh)
    """
    if isinstance(log10Mh, (list, tuple, np.ndarray)):

        Mh = np.asarray(pow(10.0, log10Mh), dtype=np.float64)
        result = np.asarray(np.zeros(len(Mh)), dtype=np.float64)

        if obs_type == "cen":
            for i, m in enumerate(Mh):
                result[i] = C_HALOMODEL.shmr_c(model, m,  log10Mstar_min, log10Mstar_max)
        elif obs_type == "sat":
            for i, m in enumerate(Mh):
                result[i] = C_HALOMODEL.shmr_s(model, m,  log10Mstar_min, log10Mstar_max)
        elif obs_type == "all":
            for i, m in enumerate(Mh):
                result[i] = C_HALOMODEL.shmr(model, m,  log10Mstar_min, log10Mstar_max)
        else:
            raise ValueError("shmr: obs_type \"{0:s}\" is not recognised".format(obs_type))

    else:

        if obs_type == "cen":
            result = C_HALOMODEL.shmr_c(model, pow(10.0, log10Mh),  log10Mstar_min, log10Mstar_max)
        elif obs_type == "sat":
            result = C_HALOMODEL.shmr_s(model, pow(10.0, log10Mh),  log10Mstar_min, log10Mstar_max)
        elif obs_type == "all":
            result = C_HALOMODEL.shmr(model, pow(10.0, log10Mh),  log10Mstar_min, log10Mstar_max)
        else:
            raise ValueError("shmr: obs_type \"{0:s}\" is not recognised".format(obs_type))

    return result



"""

-------------------------------------------------------------
Utils
-------------------------------------------------------------

"""

def loadWtheta_nz(model, fileInName):
    """ load n(z) into model

    INPUT PARAMETERS:
    model: halomodel structre
    fileInName: n(z) input file, format: z nz
    """
    data = ascii.read(fileInName, format="no_header")

    """ load n(z) in model object """
    model.wtheta_nz_N = len(data)
    model.wtheta_nz_z = np.ctypeslib.as_ctypes(data["col1"])
    model.wtheta_nz = np.ctypeslib.as_ctypes(data["col2"])

    return



def MhToTGas(model, Mh):
    """  Wrapper for c-function MhToTGas

    Returns TGas = f(Mh) for a given TGas-Mh relation.

    INPUT
    log10Mh: log10(Mh) array or single value) in log10 Msun/h units
    z: NOT USED

    OUPUT
    TGas

    """

    if isinstance(Mh, (list, tuple, np.ndarray)):
        result = np.zeros(len(Mh))
        for i, m in enumerate(Mh):
            result[i] = C_HALOMODEL.MhToTGas(model, m, np.nan)

    else:
        result = C_HALOMODEL.MhToTGas(model, Mh, np.nan)

    return pow(10.0, result)



def loadGas_TGasMh(model, fileInName):
    """ load temparature-mass relationship into model

    INPUT PARAMETERS:
    model: halomodel structre
    fileInName: txMh input file, format: log10Mh log10TGas

    WARNING: mass defintion must match the adopted definition (massDef)

    """
    data = ascii.read(fileInName, format="no_header")
    log10Mh = np.log10(data["col1"]*model.h)
    log10TGas = np.log10(data["col2"])

    """ load TGas in model object """
    model.gas_TGasMh_N = len(data)
    model.gas_TGasMh_log10Mh = np.ctypeslib.as_ctypes(log10Mh)
    model.gas_TGasMh_log10TGas = np.ctypeslib.as_ctypes(log10TGas)


    return


def MhToZGas(model, Mh):
    """  Wrapper for c-function MhToZGas

    Returns ZGas = f(Mh) for a given ZGas-Mh relation.

    INPUT
    log10Mh: log10(Mh) array or single value) in log10 Msun/h units
    z: NOT USED

    OUPUT
    ZGas

    """

    if isinstance(Mh, (list, tuple, np.ndarray)):
        result = np.zeros(len(Mh))
        for i, m in enumerate(Mh):
            result[i] = C_HALOMODEL.MhToZGas(model, m, np.nan)

    else:
        result = C_HALOMODEL.MhToZGas(model, Mh, np.nan)

    return result


def loadGas_ZGasMh(model, fileInName):
    """ load temparature-mass relationship into model

    INPUT PARAMETERS:
    model: halomodel structre
    fileInName: ZGasMh input file, format: log10Mh ZGas

    WARNING: mass defintion must match the adopted definition (massDef)
    """

    data = ascii.read(fileInName, format="no_header")
    log10Mh = np.log10(data["col1"]*model.h)
    ZGas = data["col2"]

    """ load ZGas in model object """
    model.gas_ZGasMh_N = len(data)
    model.gas_ZGasMh_log10Mh = np.ctypeslib.as_ctypes(log10Mh)
    model.gas_ZGasMh_ZGas = np.ctypeslib.as_ctypes(ZGas)

    return


def LxToCR(model, TGas, ZGas):
    """  Wrapper for c-function LxToCR()

    Returns  = LxToCR(z, TGas, ZGas) the coefficient to
    transform CR into luminosity

    INPUT
    z: redshift (not used)
    TGas: temperature (float or array)
    ZGas: metallicity

    OUPUT
    LxToCR
    """

    if isinstance(TGas, (list, tuple, np.ndarray)):
        result = np.zeros(len(TGas))
        for i, t in enumerate(TGas):
            result[i] = C_HALOMODEL.LxToCR(model, np.nan, t, ZGas)
    else:
        result = C_HALOMODEL.LxToCR(model, np.nan, TGas, ZGas)

    return result



def loadGas_LxToCR(model, fileInName):
    """ load Lx to CR conversion factor at a given
    redhsift. It depends on ZGas and TGas.

    INPUT PARAMETERS:
    model: halomodel structre
    fileInName: LXToCR input file, format: ZGas TGas fac
    """

    data = ascii.read(fileInName, format="no_header")
    ZGas = np.unique(data["col1"])
    log10TGas = np.log10(np.unique(data["col2"]))
    log10fac = np.log10(data["col3"])

    """ load LxToCR in model object """
    model.gas_LxToCR_NZGas = len(ZGas)
    model.gas_LxToCR_NTGas = len(log10TGas)
    model.gas_LxToCR_ZGas = np.ctypeslib.as_ctypes(ZGas)
    model.gas_LxToCR_log10TGas = np.ctypeslib.as_ctypes(log10TGas)
    model.gas_LxToCR_log10fac = np.ctypeslib.as_ctypes(log10fac)

    return


# def getCRtoLx_bremss(fileNameIn, redshift):
#     from scipy import interpolate
#     from   astropy.io import ascii
#
#     # CR to Lx_bolo conversion file:
#     data = ascii.read(fileNameIn, format="commented_header", header_start=-1)
#     x    = data['z']
#     y    = np.log10(data['Lx_bolo']/data['CR'])
#
#     return pow(10.0, interpolate.griddata(x, y, redshift, method='linear'))
#

def lookbackTimeInv(model, tL):
    """ Returns the redshift for a given
    lookback time
    Assumes OmegaR = 0.0
    """

    if isinstance(tL, (list, tuple, np.ndarray)):
        result = np.zeros(len(tL))
        for i, tt in enumerate(tL):
            result[i] =  C_HALOMODEL.lookbackTimeInv(model, tt)
    else:
            result = C_HALOMODEL.lookbackTimeInv(model, tL)

    return result


def lookbackTime(model, z):
    """ Returns the look back time.
    Assumes OmegaR = 0.0
    """

    if isinstance(z, (list, tuple, np.ndarray)):
        result = np.zeros(len(z))
        for i, zz in enumerate(z):
            result[i] =  C_HALOMODEL.lookbackTime(model, zz)
    else:
            result = C_HALOMODEL.lookbackTime(model, z)

    return result


def DA(model, z):
    """ Returns the angular diameter distance
    Assumes OmegaR = 0.0
    """

    if isinstance(z, (list, tuple, np.ndarray)):
        result = np.zeros(len(z))
        for i, zz in enumerate(z):
            result[i] =  C_HALOMODEL.DA(model, zz, 0)
    else:
            result = C_HALOMODEL.DA(model, z, 0)

    return result


def DM(model, z):
    """ Returns the transverse comoving distance
    Assumes OmegaR = 0.0
    """

    if isinstance(z, (list, tuple, np.ndarray)):
        result = np.zeros(len(z))
        for i, zz in enumerate(z):
            result[i] =  C_HALOMODEL.DM(model, zz, 0)
    else:
            result =  C_HALOMODEL.DM(model, z, 0)

    return result

def DL(model, z):
    """ Returns the luminosity distance
    Assumes OmegaR = 0.0
    """

    if isinstance(z, (list, tuple, np.ndarray)):
        result = np.zeros(len(z))
        for i, zz in enumerate(z):
            result[i] =  C_HALOMODEL.DL(model, zz, 0)
    else:
            result =  C_HALOMODEL.DL(model, z, 0)

    return result


def rh(model, Mh, z, D=None):
    """ Returns the radius rh enclosing Delta (D) times the CRITICAL
    density of the Universe at redshift z. If Delta = Delta_vir,
    and Mh virial mass, this is the virial radius.

    INPUT
    z: redshift
    Mh: halo mass in h^-1 Msun
    D: overdensity with respect to the critical density
        ("M200c", "M200m", "M500c" or "M500m"),
        default: mass definition in model

    OUTPUT
    rh in h^-1 Mpc and in comoving coordinates

    """

    if D is None:
        D = Delta(model, z, model.massDef)
    else:
        D = Delta(model, z, D)

    if isinstance(Mh, (list, tuple, np.ndarray)):
        result = np.asarray(np.zeros(len(Mh)), dtype=np.float64)
        for i, m in enumerate(Mh):
            result[i] = C_HALOMODEL.rh(model, Mh[i], D, z)
    else :
        result = C_HALOMODEL.rh(model, Mh, D, z)

    return result

def bias_h(model, Mh, z):
    """ Returns halo bias """
    return C_HALOMODEL.bias_h(model, Mh, z)

def concentration(model, Mh, z, concenDef="TJ03"):
    """ Returns the concentration """
    return C_HALOMODEL.concentration(model, Mh, z, concenDef)


def Delta(model, z, massDef):
    """ Returns Delta according to mass definition.
    Matches Delta() in C_HALOMODEL

    Delta defined wrt critical density
    """

    return C_HALOMODEL.Delta(model, z, massDef)


def r_vir(model, Mh, c, z):
    """ Returns r_vir """


    if c is None:
        c = np.nan

    return C_HALOMODEL.r_vir(model, Mh, c, z)


def msmh_log10Mstar(model, log10Mh):
    """  Wrapper for c-function msmh_log10Mstar()

    Returns Mstar = f(Mh) for a given Mstar-Mh relation.

    INPUT
    log10Mh: log10(Mh) array or single value) in log10 Msun/h units

    OUPUT
    log10Mstar evaluated at log10Mh

    """

    if isinstance(log10Mh, (list, tuple, np.ndarray)):
        result = np.zeros(len(log10Mh))
        for i, m in enumerate(log10Mh):
            result[i] = C_HALOMODEL.msmh_log10Mstar(model, m)
    else:
        result = C_HALOMODEL.msmh_log10Mstar(model, log10Mh)

    return result




def msmh_log10Mh(model, log10Mstar):
    """  Wrapper for c-function msmh_log10Mh()

    Returns Mh = f(Mstar) for a given Mstar-Mh relation.

    INPUT
    log10Mstar: log10(Mstar) array or single value) in log10 Msun/h units

    OUPUT
    log10Mh evaluated at log10Mstar

    """

    if isinstance(log10Mstar, (list, tuple, np.ndarray)):
        result = np.zeros(len(log10Mstar))
        for i, m in enumerate(log10Mstar):
            result[i] = C_HALOMODEL.msmh_log10Mh(model, m)
    else:
        result = C_HALOMODEL.msmh_log10Mh(model, log10Mstar)

    return result

def log10M_sat(model, log10Mstar):
    """  Wrapper for c-function msmh_log10M_sat()

    Returns Msat = f(Mstar) for a given Mstar-Mh relation.

    INPUT
    log10Mstar: log10(Mstar) array or single value) in log10 h^-2 Msun units

    OUPUT
    log10Mh evaluated at log10Mstar Mh in h^-1 Msun units

    """

    if isinstance(log10Mstar, (list, tuple, np.ndarray)):
        result = np.zeros(len(log10Mstar))
        for i, m in enumerate(log10Mstar):
            result[i] = C_HALOMODEL.log10M_sat(model, m)
    else:
        result = C_HALOMODEL.log10M_sat(model, log10Mstar)

    return result


def log10M_cut(model, log10Mstar):
    """  Wrapper for c-function log10M_cut()

    Returns Mh = f(Mstar) for a given Mstar-Mh relation.

    INPUT
    log10Mstar: log10(Mstar) array or single value) in log10 Msun/h units

    OUPUT
    log10Mh evaluated at log10Mstar

    """

    if isinstance(log10Mstar, (list, tuple, np.ndarray)):
        result = np.zeros(len(log10Mstar))
        for i, m in enumerate(log10Mstar):
            result[i] = C_HALOMODEL.log10M_cut(model, m)
    else:
        result = C_HALOMODEL.log10M_cut(model, log10Mstar)

    return result





def LambdaBolo(TGas, ZGas):
    """  Wrapper for c-function Lambda()

    Returns  = f(TGas, ZGas) the cooling function
    for a given temperature and metallicity

    INPUT
    TGas: temperature (float or array)
    ZGas: metallicity

    OUPUT
    Lambda

    """

    if isinstance(TGas, (list, tuple, np.ndarray)):
        result = np.zeros(len(TGas))
        for i, t in enumerate(TGas):
            result[i] = C_HALOMODEL.LambdaBolo(t, ZGas)
    else:
        result = C_HALOMODEL.LambdaBolo(TGas, ZGas)

    return result


def Lambda0p5_2p0(TGas, ZGas):
    """  Wrapper for c-function Lambda()

    Returns  = f(TGas, ZGas) the cooling function
    for a given temperature and metallicity

    INPUT
    TGas: temperature (float or array)
    ZGas: metallicity

    OUPUT
    Lambda

    """

    if isinstance(TGas, (list, tuple, np.ndarray)):
        result = np.zeros(len(TGas))
        for i, t in enumerate(TGas):
            result[i] = C_HALOMODEL.Lambda0p5_2p0(t, ZGas)
    else:
        result = C_HALOMODEL.Lambda0p5_2p0(TGas, ZGas)

    return result




def M1_to_M2(model, M1, c1, Delta1, Delta2, z):

    Delta1 = Delta(model, z, Delta1)
    Delta2 = Delta(model, z, Delta2)

    if c1 is None:
        c1 = np.nan

    M2 = ctypes.c_double()
    c2 = ctypes.c_double()

    C_HALOMODEL.M1_to_M2(model, M1, c1, Delta1, Delta2, z, ctypes.pointer(M2), ctypes.pointer(c2))

    return M2.value, c2.value

def log10M1_to_log10M2(model, log10M1, log10c1, Delta1, Delta2, z):

    if log10c1 is None:
        c1 = None
    else:
        c1 = pow(10.0, log10c1)

    M2, c2 = M1_to_M2(model, pow(10.0, log10M1), c1, Delta1, Delta2, z)

    return np.log10(M2), np.log10(c2)


def getInterParaGas(model, log10Mh):
    """ get log10n0, log10beta and log10rc in model
    from interpolated values, and returns
    the values
    """

    if isinstance(log10Mh, np.ndarray):

        N = len(log10Mh)

        gas_log10n0 = np.zeros(N)
        gas_log10beta = np.zeros(N)
        gas_log10rc  = np.zeros(N)
        for i, m in enumerate(log10Mh):
            gas_log10n0[i] = C_HALOMODEL.inter_gas_log10n0(model, m)
            gas_log10beta[i] = C_HALOMODEL.inter_gas_log10beta(model, m)
            gas_log10rc[i] = C_HALOMODEL.inter_gas_log10rc(model, m)

    else:

        gas_log10n0 = C_HALOMODEL.inter_gas_log10n0(model, log10Mh)
        gas_log10beta = C_HALOMODEL.inter_gas_log10beta(model, log10Mh)
        gas_log10rc  = C_HALOMODEL.inter_gas_log10rc(model, log10Mh)


    return [gas_log10n0, gas_log10beta, gas_log10rc]



def setInterParaGas(model, log10Mh):
    """ Set log10n0, log10beta and log10rc in model
    from interpolated values, and returns
    the values
    """

    if isinstance(log10Mh, np.ndarray):

        gas_log10n0 = []
        gas_log10beta = []
        gas_log10rc  = []
        for m in log10Mh:
            gas_log10n0.append(C_HALOMODEL.inter_gas_log10n0(model, m))
            gas_log10beta.append(C_HALOMODEL.inter_gas_log10beta(model, m))
            gas_log10rc.append(C_HALOMODEL.inter_gas_log10rc(model, m))

        return [gas_log10n0, gas_log10beta, gas_log10rc]

    else:

        model.gas_log10n0 = C_HALOMODEL.inter_gas_log10n0(model, log10Mh)
        model.gas_log10beta = C_HALOMODEL.inter_gas_log10beta(model, log10Mh)
        model.gas_log10rc  = C_HALOMODEL.inter_gas_log10rc(model, log10Mh)


        return [model.gas_log10n0 , model.gas_log10beta, model.gas_log10rc]

def dumpModel(model, fileOutName=None):
    """ Returns model parameters as a string
    If a file name is given, it will prepend
    the string as a header.
    """

    string = ""
    for a in model._fields_:
        string+= "# {0:s} = {1}\n".format(a[0], getattr(model, a[0]))

    if fileOutName is not None:
        fileIn = file(fileOutName, 'r')
        new = string+fileIn.read()
        fileIn.close()

        fileOut = open(fileOutName, 'w')
        fileOut.write(new)

    return string
