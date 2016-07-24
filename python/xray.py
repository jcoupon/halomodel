
def CRToLx(filein, z):

    from   astropy.io import ascii
    from scipy import interpolate

    data = ascii.read(filein)

    ZGas_Tx_z = (data['ZGas'], data['Tx'], data['z'])

    CR = interpolate.griddata(ZGas_Tx_z, data["CR_pn"]+2.0*data["CR_MOS"], (0.25, 0.5, 0.2), method='linear')
    Lx = interpolate.griddata(ZGas_Tx_z, data["Lx_bolo"], (0.25, 0.5, 0.2), method='linear')

    return Lx
