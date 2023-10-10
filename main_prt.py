import numpy as np
import glob
import os
from opac_mixer.read import ReadOpacChubb

import matplotlib.pyplot as plt

from opac_mixer.patches.prt import PatchedRadtrans
from petitRADTRANS import nat_cst as nc

from petitRADTRANS.poor_mans_nonequ_chem import interpol_abundances
from petitRADTRANS.physics import guillot_global

pressures = np.logspace(-6, 2, 10)
R='S1'
mixmethod = 'aee_jit'
mixmethod_orig = 'aee'

base = f'{os.environ["pRT_input_data_path"]}/opacities/lines/corr_k'
files = glob.glob(os.path.join(base,f'*_R_{R}/*.h5'))

opac = ReadOpacChubb(files)
opac.read_opac()

linespecies = [f.split('/')[-1].split('.')[0] for f in files]
atmosphere_orig = PatchedRadtrans(line_species=linespecies, pressures=pressures, wlen_bords_micron=[(1e4/opac.bin_edges).min(), (1e4/opac.bin_edges).max()])
atmosphere = PatchedRadtrans(line_species=linespecies, pressures=pressures, wlen_bords_micron=[(1e4/opac.bin_edges).min(), (1e4/opac.bin_edges).max()])
atmosphere.setup_mixing(mixmethod=mixmethod)
atmosphere_orig.setup_mixing(mixmethod=mixmethod_orig)

fig_fl, ax_fl = plt.subplots(2,1, constrained_layout=True)


def plot_flux_toa(gamma, T_int, T_equ):
    for axi in ax_fl:
        axi.cla()

    kappa_IR = 0.05
    Tstar=6000
    Rstar=1.0*nc.r_sun
    semimajoraxis=0.01*nc.AU
    gravity = 1e1**2.45
    geometry='dayside_ave'

    # temperature = chubb.Tr
    temperature = guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)

    COs = 0.55 * np.ones_like(pressures)
    FeHs = 0. * np.ones_like(pressures)

    mass_fractions = interpol_abundances(COs, \
                                         FeHs, \
                                         temperature, \
                                         pressures)

    for sp in linespecies:
        mass_fractions[sp] = mass_fractions.pop(sp.split('_')[0])

    atmosphere_orig.calc_flux(temperature, mass_fractions, gravity, mass_fractions['MMW'], Rstar=Rstar, Tstar=Tstar, semimajoraxis=semimajoraxis, geometry=geometry)
    atmosphere.calc_flux(temperature, mass_fractions, gravity, mass_fractions['MMW'], Rstar=Rstar, Tstar=Tstar, semimajoraxis=semimajoraxis, geometry=geometry)

    bol_orig =np.sum(atmosphere_orig.flux*np.diff(atmosphere_orig.border_freqs))
    bol = np.sum(atmosphere.flux*np.diff(atmosphere.border_freqs))
    print('relative bolometric error on calculated flux:', (bol_orig-bol)/bol)

    ax_fl[0].plot(nc.c/atmosphere_orig.freq*1e4, atmosphere_orig.flux, label ='original')
    ax_fl[0].plot(nc.c/atmosphere.freq*1e4, atmosphere.flux, label ='emulator')
    ax_fl[0].set_xscale('log')
    ax_fl[0].set_xlabel(r'$\lambda$ / $\mu$m')
    ax_fl[0].set_ylabel(r'$F_\nu$ / cgs')
    ax_fl[1].loglog(temperature, pressures)
    ax_fl[1].set_ylabel(r'pressure / bar')
    ax_fl[1].set_xlabel('temperature')
    ax_fl[1].invert_yaxis()
    ax_fl[0].legend()


if __name__ == "__main__":
    plot_flux_toa(0.4, 100, 1000)