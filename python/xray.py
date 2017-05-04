
def CRToLx(fileInName, z, Zgas):
    """ function to return CR to Lx conversion
    as a function of temperature.

    Takes the redshift and the metallicity as input.
    Returns log(tx) values and convertion factor.
    """

    from astropy.io import ascii
    from scipy import interpolate
    from numpy import linspace, log

    N = 128

    data = ascii.read(fileInName)

    z_ZGas_logTGas = (data['z'], data['ZGas'], log(data['TGas']))
    logTGas  = linspace(log(min(data['TGas'])), log(max(data['TGas'])), N)

    z_ZGas_logTGas_eval = ([z]*N, [Zgas]*N, logTGas )

    logCR = interpolate.griddata(z_ZGas_logTGas, log(data["CR_pn"]+2.0*data["CR_MOS"]), z_ZGas_logTGas_eval, method='linear')
    logLx = interpolate.griddata(z_ZGas_logTGas, log(data["Lx_bolo"]), z_ZGas_logTGas_eval, method='linear')

    return list(logTGas), list(logLx-logCR)
