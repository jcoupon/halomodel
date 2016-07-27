
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

    z_ZGas_logTx = (data['z'], data['ZGas'], log(data['Tx']))
    logTx  = linspace(log(min(data['Tx'])), log(max(data['Tx'])), N)

    z_ZGas_logTx_eval = ([z]*N, [Zgas]*N, logTx )

    logCR = interpolate.griddata(z_ZGas_logTx, log(data["CR_pn"]+2.0*data["CR_MOS"]), z_ZGas_logTx_eval, method='linear')
    logLx = interpolate.griddata(z_ZGas_logTx, log(data["Lx_bolo"]), z_ZGas_logTx_eval, method='linear')

    return list(logTx), list(logLx-logCR)
